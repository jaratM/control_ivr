from .types import ClassificationResult, ClassificationInput
import time
import os
import requests
import json
import re
import random
from typing import Dict, Tuple
from loguru import logger
# INSERT_YOUR_CODE
from dotenv import load_dotenv
load_dotenv()

class Classifier:
    """
    Classifies call outcomes using Claude via an external API.
    Returns deterministic two-digit output: call type (0-8) + technician behavior (0-1).
    """

    CATEGORY_MAP = {
        0: "Silence",
        1: "Client refuse installation",
        2: "Client reporte RDV",
        3: "CLIENT INJOIGNABLE",
        4: "autre",
        5: "Attente retour client",
        6: "Client Absent",
        7: "Absence routeur client",
        8: "local ferme",
        9: "Non classifié"
    }
    
    TECHNICIAN_BEHAVIOR_MAP = {
        1: "Bien",
        0: "Mauvais",
        2: "Non classifié"
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
        
        self.api_url = os.getenv("BEDROCK_API_URL")
        # Fallback to a constructed URL if API_ID is present
        if not self.api_url:
            api_id = os.getenv("BEDROCK_API_ID")
            if api_id:
                self.api_url = f"https://{api_id}.execute-api.eu-west-3.amazonaws.com/dev/chat"
            elif self.log:
                logger.warning("[Classifier] BEDROCK_API_URL or BEDROCK_API_ID not set in environment.")

        if self.log:
            logger.info(f"\n[Classifier] Initialized (API Mode) for {category}")
        
        # Load prompt from external file
        try:
            config_classification = self.config.get('classification', {}) if self.config else {}
            prompt_file = config_classification.get(category, 'config/sav.txt')
            logger.info(f"[Classifier] config_classification: {config_classification}, prompt_file: {prompt_file}")
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

            if not full_text or len(full_text) < 4:
                # Default to (Silence, Good Behavior) for empty text? 
                # Original code used 0, 1
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
                return self._make_result(9, 2, file_id)

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
                    'Transcription: « انكم في العلبة » → Answer: 31\n'
                )

        user_prompt = (
            f"{fewshots}\n"
            f'Now classify this transcription:\n'
            f'Transcription: « {text} »\n\n'
            f"Return only two digit:"
        )
        return system_prompt, user_prompt

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call Claude via requests with retries."""
        if not self.api_url:
            raise ValueError("BEDROCK_API_URL is not set.")

        max_attempts = self.config.get('max_attempts', 10)
        base_delay = 2.0
        max_delay = 300.0

        full_message = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt

        payload = {
            "message": full_message,
            "history": [] # No history for single-turn classification
        }

        for attempt in range(1, max_attempts + 1):
            try:
                response = requests.post(
                    self.api_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30 
                )

                if response.status_code == 200:
                    resp_json = response.json()
                    if resp_json.get("status") == "success":
                        return resp_json["data"]["response"]
                    else:
                        raise ValueError(f"API returned non-success status: {resp_json}")

                # Handle Errors
                if response.status_code == 429:
                    error_msg = response.json().get("message", "Unknown 429 error")
                    error = response.json().get('error', '')
                    logger.error(f"[Classifier] 429 Error ({error}): {error_msg}")
                    raise Exception(f"API 429 Error ({error}): {error_msg}")
                
                if response.status_code == 400:
                    error_msg = response.json().get("message", "Bad request")
                    raise ValueError(f"API 400 Bad Request: {error_msg}")
                
                response.raise_for_status()

            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.HTTPError) as e:
                # Retry on network/server errors
                is_server_error = False
                if isinstance(e, requests.exceptions.HTTPError) and e.response is not None:
                     if e.response.status_code >= 500:
                         is_server_error = True
                
                should_retry = isinstance(e, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)) or is_server_error

                if should_retry and attempt < max_attempts:
                    exponential_delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    jitter = random.uniform(0, exponential_delay * 0.3)
                    sleep_time = exponential_delay + jitter
                    
                    if self.log:
                        logger.warning(f"[Classifier] API Request failed (attempt {attempt}/{max_attempts}): {e}. Sleeping {sleep_time:.1f}s")
                    time.sleep(sleep_time)
                else:
                    if self.log and attempt == max_attempts:
                        logger.error(f"[Classifier] Max retries reached. Last error: {e}")
                    raise e
            
            except Exception as e:
                # Non-retryable
                raise e

    @staticmethod
    def _parse_response(response: str) -> Tuple[int, int, str]:
        """
        Extract two digits (call_type 0-4, tech_behavior 0-1) from the
        model response.
        """
        if not response:
            return 4, 1, "Empty model response; defaulting to 41."

        m = re.search(r"([0-8])\s*([0-1])", response) 

        if not m:
            m_single = re.search(r"[0-8]", response)
            if m_single:
                call_type = int(m_single.group())
                return call_type, 1, f"Only found one digit: {response!r}. Defaulting tech behavior to 1."
            
            return 4, 1, f"No valid digits in model response: {response!r}. Defaulting to 41."
        
        call_type = int(m.group(1))
        tech_behavior = int(m.group(2))
        
        return call_type, tech_behavior, f"Model returned: {response!r}"