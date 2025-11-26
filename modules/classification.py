from .types import ClassificationResult, ClassificationInput
import time

class Classifier:
    def __init__(self, device: str = "cpu"):
        # Device param is kept for compatibility but this is now an API client
        pass

    def classify_full_text(self, text: str, file_id: str) -> ClassificationResult:
        # Simulate AWS Bedrock API Latency (network I/O)
        # Real implementation would use boto3.client('bedrock-runtime')
        time.sleep(0.5) 
        
        # Mock logic based on text content
        is_polite = "please" in text.lower() or "thank" in text.lower()
        tags = ["professional"] if is_polite else ["neutral"]
        
        return ClassificationResult(
            file_id=file_id,
            status="clean",
            behavior_tags=tags
        )
