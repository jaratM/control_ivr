import json
import os
import time
import soundfile as sf
from .types import AudioMetadata
from loguru import logger

class MetadataExtractor:
    def extract(self, audio_path: str) -> AudioMetadata:
        """
        Loads business metadata from the companion JSON file and 
        technical metadata from the audio file header.
        
        Expects: /path/to/file.ogg -> /path/to/file.json
        """
        # 1. Derive JSON path
        base_path, _ = os.path.splitext(audio_path)
        json_path = f"{base_path}.json"
        
        # Defaults
        file_id = os.path.basename(base_path)
        created_at = time.ctime()
        
        # 2. Load Business Metadata from JSON
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    # Map JSON fields to our schema
                    file_id = data.get("file_id", file_id)
                    created_at = data.get("created_at", created_at)
                    # You can extend this to capture other business fields
            except Exception as e:
                logger.error(f"Failed to parse JSON metadata for {audio_path}: {e}")
        else:
            logger.warning(f"Metadata JSON not found for {audio_path}, using defaults.")

      

        return AudioMetadata(
            file_path=audio_path,
            file_id=file_id,
            created_at=created_at
        )
