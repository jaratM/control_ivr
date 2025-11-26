from typing import List, Dict
from .types import AudioSegment, TranscriptionResult
import time
import torch
from transformers import Wav2Vec2BertForCTC, Wav2Vec2BertProcessor
import numpy as np
from loguru import logger

class Transcriber:
    def __init__(self, device: str = "cpu", config: Dict = None):
        self.device = device
        self.config = config or {}
        
        model_name = self.config.get('transcription_model', 'facebook/w2v-bert-2.0') # Fallback if not in config
        
        try:
            self.model = Wav2Vec2BertForCTC.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                attn_implementation="eager",
                low_cpu_mem_usage=True
            ).to(self.device)
            
            self.processor = Wav2Vec2BertProcessor.from_pretrained(model_name)
            self.model.eval()
            
            # Optimization: Compile the model for faster inference (PyTorch 2.0+)
            # We use 'reduce-overhead' which is great for inference loops
            if hasattr(torch, 'compile'):
                logger.info(f"Compiling model on {self.device}...")
                self.model = torch.compile(self.model, mode="reduce-overhead")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def transcribe_batch(self, batch: List[AudioSegment]) -> List[TranscriptionResult]:
        if not batch:
            return []
            
        # 1. Prepare Inputs
        # Convert list of AudioSegments (numpy arrays) to list of arrays
        # Note: AudioSegment.audio_data is expected to be 1D array from previous step
        
        # Extract audio arrays
        audio_arrays = [seg.audio_data for seg in batch]
        
        # 2. Processor Call
        try:
            inputs = self.processor(
                audio_arrays, 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=True
            )
            
            input_features = inputs["input_features"].to(self.device)
            
            # 3. Inference
            with torch.no_grad():
                logits = self.model(input_features=input_features).logits
                
            # 4. Decode
            predicted_ids = torch.argmax(logits, dim=-1)
            transcripts = self.processor.batch_decode(predicted_ids)
            
            # 5. Post-processing & Result Creation
            results = []
            for i, text in enumerate(transcripts):
                seg = batch[i]
                results.append(TranscriptionResult(
                    file_id=seg.file_id,
                    segment_index=seg.segment_index,
                    text=text,
                ))
                
            # Cleanup GPU memory for this batch
            del input_features, logits, predicted_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return results
            
        except Exception as e:
            logger.error(f"Batch transcription failed: {e}")
            # Return empty or error results to avoid crashing pipeline
            # Ideally, we should propagate error
            return []
