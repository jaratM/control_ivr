from .types import ClassificationResult, ClassificationInput
import time
import os
import boto3
import json
import re
import random
from typing import Dict, Tuple
from botocore.exceptions import ClientError
from botocore.config import Config
from loguru import logger
# INSERT_YOUR_CODE
from dotenv import load_dotenv
load_dotenv()

class Classifier:
    """
    Classifies call outcomes using Claude via AWS Bedrock.
    Returns deterministic two-digit output: call type (0-8) + technician behavior (0-1).
    """

    CATEGORY_MAP = {
        0: "Silence",
        1: "Client refuse installation",
        2: "Client reporte RDV",
        3: "CLIENT INJOIGNABLE ",
        4: "autre",
        5: "Attente retour client",
        6: "Client Absent",
        7: "Absence routeur client",
        8: "local ferme",
    }
    
    TECHNICIAN_BEHAVIOR_MAP = {
        1: "Bien",
        0: "Mauvais"
    }

    def __init__(
        self,
        category: str,
        aws_region: str = "us-west-2",
        model_id: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        log: bool = True,
        config: dict = None
    ):
        self.model_id = model_id
        self.log = log
        self.config = config
        # Configure boto3 with reduced retries so our custom retry logic can handle throttling
        boto_config = Config(
            retries={
                'max_attempts': 2,  # Reduce boto3's internal retries
                'mode': 'adaptive'
            }
        )
        
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_session_token = os.getenv("AWS_SESSION_TOKEN")
        if aws_access_key and aws_secret_key:
            self.bedrock = boto3.client(
                "bedrock-runtime",
                region_name=aws_region,
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                aws_session_token=aws_session_token,
                config=boto_config,
            )
        else:
            self.bedrock = boto3.client("bedrock-runtime", region_name=aws_region, config=boto_config)

        if self.log:
            logger.info(f"\n[Classifier] Initialized with {model_id} (Claude)")
        
        # Load prompt from external file
        try:
            config_classification = self.config.get('classification', {}) if self.config else {}
            prompt_file = config_classification.get(category, 'config/sav.txt')
            with open(prompt_file, "r", encoding="utf-8") as f:
                self.system_prompt = f.read().strip()
            if self.log:
                logger.info(f"[Classifier] Loaded prompt from {prompt_file}")
        except Exception as e:
            if self.log:
                logger.warning(f"[Classifier] Warning: Could not load prompt.txt: {e}")
            self.system_prompt = None 

    def classify_full_text(self, full_text: str, file_id: str) -> Dict:
            """Classify call outcome."""
            if self.log:
                logger.info(f"\n[Classification] Processing...")

            if not full_text or len(full_text) < 5:
                return self._make_result(0, 1, file_id)

            system_prompt, user_prompt = self._build_prompts(full_text)
            try:
                raw = self._call_llm(system_prompt, user_prompt)
                call_type, tech_behavior, reasoning = self._parse_response(raw)
                
                logger.info(f"Classification result: Call Type={call_type} ({self.CATEGORY_MAP.get(call_type, 'Unknown')}), Tech Behavior={tech_behavior} ({self.TECHNICIAN_BEHAVIOR_MAP.get(tech_behavior, 'Unknown')})")
                return self._make_result(call_type, tech_behavior, file_id)
            
            except Exception as e:
                if self.log:
                    logger.error(f"Classification error: {str(e)}")
                # Default to (Other, Good Behavior) on error
                return self._make_result(4, 1, file_id)

    def _make_result(self, call_type: int, tech_behavior: int, file_id: str) -> Dict:
            """Helper to build the result dictionary with both classification digits."""
            
            # Default to 1 (Good) if an invalid value is somehow passed
            if tech_behavior not in self.TECHNICIAN_BEHAVIOR_MAP:
                tech_behavior = 1
                
            return ClassificationResult(
                status=self.CATEGORY_MAP[call_type],
                behavior=self.TECHNICIAN_BEHAVIOR_MAP[tech_behavior], 
                file_id=file_id
            )

    def _build_prompts(self, text: str) -> Tuple[str, str]:
        # Use the system prompt loaded from prompt.txt
        system_prompt = self.system_prompt if self.system_prompt else "You are a call classifier."

        fewshots = (
                    'Transcription: « بغيت نلغيه الطلب » → Answer: 11\n'
                    'Transcription: « غنأنوليها أخويا » → Answer: 11\n'
                    'Transcription: « صافي راه خدام نعم خدام » → Answer: 11\n'
                    'Transcription: « زكاو معيا نهار الجمعة » → Answer: 21\n'
                    'Transcription: « الوقت لي غاتكون في الدار » → Answer: 21\n'
                    'Transcription: « نديرو الجمعة صباح » → Answer: 21\n'
                    'Transcription: « Orange vous remercie, votre correspondant n\'est pas joignable » → Answer: 31\n'
                )

        user_prompt = (
            f"{fewshots}\n"
            f'Now classify this transcription:\n'
            f'Transcription: « {text} »\n\n'
            f"Return only two digit:"
        )
        return system_prompt, user_prompt

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call Claude LLM via Bedrock with exponential backoff and jitter."""
        max_attempts = self.config.get('max_attempts', 10)  # Increased default
        base_delay = 2.0  # Increased base delay
        max_delay = 300.0  # Cap at 5 minutes

        for attempt in range(1, max_attempts + 1):
            try:
                return self._call_claude(system_prompt, user_prompt)
                    
            except ClientError as e:
                error_response = e.response.get("Error", {})
                code = error_response.get("Code", "")
                message = error_response.get("Message", "")
                
                # Handle throttling and rate limiting errors
                is_throttle = code in ["ThrottlingException", "TooManyRequestsException"] or "throttl" in message.lower() or "too many requests" in message.lower()
                
                if not is_throttle or attempt == max_attempts:
                    # Non-throttling error or final attempt - raise
                    if attempt == max_attempts and is_throttle:
                        logger.error(f"[Classifier] Max retries ({max_attempts}) reached for throttling. Last error: {code} - {message}")
                    raise
                
                # Exponential backoff with jitter
                exponential_delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                jitter = random.uniform(0, exponential_delay * 0.3)  # Add up to 30% jitter
                sleep_time = exponential_delay + jitter
                
                if self.log:
                    logger.warning(f"[Classifier] Throttled (attempt {attempt}/{max_attempts}), "
                          f"sleeping {sleep_time:.1f}s (exponential: {exponential_delay:.1f}s + jitter: {jitter:.1f}s)")
                time.sleep(sleep_time)

    def _call_claude(self, system_prompt: str, user_prompt: str) -> str:
        """Claude-specific API format."""
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "system": system_prompt,
            "temperature": 0.0,
            "max_tokens": 200,
            "stop_sequences": [".", "–", "-", "→"],
            "messages": [{"role": "user", "content": user_prompt}],
        }
        
        response = self.bedrock.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body).encode("utf-8"),
            accept="application/json",
            contentType="application/json",
        )
        payload = json.loads(response["body"].read())
        return payload["content"][0]["text"].strip()

    @staticmethod
    def _parse_response(response: str) -> Tuple[int, int, str]:
        """
        Extract two digits (call_type 0-4, tech_behavior 0-1) from the
        model response.
        """
        if not response:
            # Default to (Other, Good Behavior)
            return 4, 1, "Empty model response; defaulting to 41."

        # Regex to find the first digit [0-8] followed by the second [0-1]
        m = re.search(r"([0-8])\s*([0-1])", response) 

        if not m:
            # Fallback: if only one digit is found, default tech behavior to 1
            m_single = re.search(r"[0-8]", response)
            if m_single:
                call_type = int(m_single.group())
                return call_type, 1, f"Only found one digit: {response!r}. Defaulting tech behavior to 1."
            
            # Default to (Other, Good Behavior) if no valid digits are found
            return 4, 1, f"No valid digits in model response: {response!r}. Defaulting to 41."
        
        call_type = int(m.group(1))
        tech_behavior = int(m.group(2))
        
        return call_type, tech_behavior, f"Model returned: {response!r}"
   