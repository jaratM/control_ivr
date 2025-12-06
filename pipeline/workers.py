import time
import queue
import os
import tempfile

from numpy import log10
import soundfile as sf
from loguru import logger
from modules.frequency import FrequencyAnalyzer
from modules.transcription import Transcriber
from modules.classification import Classifier
from modules.compliance import ComplianceVerifier
from modules.types import AudioSegment, ClassificationInput, ComplianceInput, AudioMetadata
from storage.minio_client import MinioStorage
import json
import pandas as pd

# --- Ingestion Worker ---
def ingestion_worker(input_queue, segment_queue, assembly_queue, config, ingestion_output_file):
    """
    Ingestion worker.
    - Get the input item from the input queue
    - Download the audio file from the S3 bucket
    - Analyze the audio file
    - Notify the assembler (start of file)
    - Send the segments to the batcher
    - Save the ingestion output to a file
    """
    freq_analyzer = FrequencyAnalyzer()
    
    # Initialize Minio
    minio_config = config.get('storage', {})
    minio = MinioStorage(
        endpoint=minio_config.get('endpoint'),
        access_key=minio_config.get('access_key'),
        secret_key=minio_config.get('secret_key'),
        bucket_name=minio_config.get('bucket_name'),
        secure=minio_config.get('secure', False)
    )
        
    while True:
        try:
            input_item = input_queue.get(timeout=1.0)
            if input_item is None:
                segment_queue.put(None) # Propagate stop
                break

            file_id = None
            local_file_path = None
            
            try:
                # Handle dictionary input (from ManifestProcessor)
                if isinstance(input_item, dict):
                    call_id = input_item.get('call_id')
                    s3_path = input_item.get('s3_path_audio')
                    start_time = input_item.get('start_time')
                    file_id = call_id
                    
                    if not s3_path:
                        logger.error(f"No s3_path_audio for call {call_id}")
                        continue

                    # Download to temp file
                    # Determine extension
                    _, ext = os.path.splitext(s3_path)
                    if not ext:
                        ext = ".ogg" # Default
                        
                    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_file:
                        local_file_path = tmp_file.name
                    
                    # Download using Minio client
                    # s3_path might be like "19-11-2025/filename.ogg"
                    # minio.download_file expects object_name, local_path
                    if not minio.download_file(s3_path, local_file_path):
                         logger.error(f"Failed to download {s3_path}")
                         continue
                    
                    # Create Metadata
                    try:
                        info = sf.info(local_file_path)
                        duration = info.duration
                        sample_rate = info.samplerate
                    except Exception as e:
                        logger.warning(f"Could not get audio info with soundfile: {e}")
                        duration = 0.0
                        sample_rate = 0
                    
                    metadata = AudioMetadata(
                        file_path=s3_path,
                        file_id=file_id,
                        duration=duration,
                        sample_rate=sample_rate,
                        created_at=str(start_time)
                    )
                    
                
                # 2. Frequency Analysis
                processing_config = config.get('processing', {})
                number_high_beeps, number_low_bips, segments = freq_analyzer.process(local_file_path, file_id, processing_config)
                
                # logger.info(f"Frequency analysis completed for file {file_id}. High beeps: {high_beeps}, Low beeps: {low_beeps}, Segments: {len(segments)}")
                
                
                # 3. Notify Assembler (Start of file)
                assembly_queue.put({
                    "type": "FILE_INIT",
                    "file_id": file_id,
                    "metadata": metadata,
                    "beep_count": number_low_bips,
                    "total_segments": len(segments),
                    "high_beeps": number_high_beeps
                })
                
                # Save ingestion output to file
                with open(ingestion_output_file, "a") as f:
                    f.write(json.dumps({
                        "file_id": file_id,
                        "high_beeps": number_high_beeps,
                        "low_beeps": number_low_bips,
                        "segments": len(segments)
                    }) + "\n")
                # 4. Send segments to batcher
                for seg in segments:
                    segment_queue.put(seg)
            
            except Exception as e:
                logger.error(f"Error processing {file_id}: {e}")
            
            finally:
                # Cleanup temp file if we created one
                if isinstance(input_item, dict) and local_file_path and os.path.exists(local_file_path):
                    os.remove(local_file_path)

        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Error in ingestion worker loop: {e}")

# --- Batcher Worker ---
def batcher_worker(segment_queue, transcription_queue, config):
    batch_size = config['pipeline']['batch_size']
    timeout = config['pipeline']['batch_timeout_ms'] / 1000.0
    
    current_batch = []
    last_flush = time.time()
    
    logger.info("Batcher worker started")
    
    num_ingestion = config['pipeline']['num_ingestion_workers']
    stop_signals_received = 0
    
    while True:
        try:
            # Short timeout to check time-based flushing
            item = segment_queue.get(timeout=0.05)
            
            if item is None:
                stop_signals_received += 1
                if stop_signals_received >= num_ingestion:
                    # Flush remaining
                    if current_batch:
                        transcription_queue.put(current_batch)
                    # We let the Orchestrator handle downstream shutdown
                    break
                continue
                
            current_batch.append(item)
            
            if len(current_batch) >= batch_size:
                logger.info(f"Putting batch of {len(current_batch)} segments to transcription queue")
                transcription_queue.put(current_batch)
                current_batch = []
                last_flush = time.time()
                
        except queue.Empty:
            # Check timeout flush
            if current_batch and (time.time() - last_flush > timeout):
                transcription_queue.put(current_batch)
                current_batch = []
                last_flush = time.time()
                
# --- GPU Worker ---
def gpu_worker(transcription_queue, assembly_queue, config, gpu_id):
    # Initialize Models on specific GPU
    
    logger.info(f"GPU Worker started on device {gpu_id}")
    device = f"cuda:{gpu_id}" if gpu_id != "cpu" else "cpu"
    
    transcriber = Transcriber(device=device, config=config)
    
    while True:
        try:
            batch = transcription_queue.get(timeout=1.0)
            if batch is None:
                # We don't signal assembly queue here, orchestrator handles shutdown flow
                break
            
            # Transcribe
            transcripts = transcriber.transcribe_batch(batch)
            
            # Send individual results to Assembler
            for t_result in transcripts:
                assembly_queue.put({
                    "type": "SEGMENT_RESULT",
                    "file_id": t_result.file_id,
                    "data": t_result
                })
                
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Error in GPU worker: {e}")

# --- Assembler Worker ---
def assembler_worker(assembly_queue, classification_queue, config, assembler_output_file):
    # State: file_id -> { metadata, beep_count, total_segments, received_count, text_segments: {index: text} }
    state_store = {}
    
    logger.info("Assembler worker started")
    
    while True:
        try:
            msg = assembly_queue.get(timeout=1.0)
            
            if msg is None:
                break
                
            msg_type = msg["type"]
            file_id = msg["file_id"]
            
            if msg_type == "FILE_INIT":
                state_store[file_id] = {
                    "metadata": msg["metadata"],
                    "beep_count": msg["beep_count"],
                    "total_segments": msg["total_segments"],
                    "text_segments": {},
                    "received_count": 0
                }
                logger.info(f"FILE_INIT: {file_id} initiated at Assembler worker")
            elif msg_type == "SEGMENT_RESULT":
                if file_id not in state_store:
                    # This can happen if FILE_INIT is delayed or lost, but in local MP queues order is usually preserved 
                    # from same producer. But Ingestion produces INIT, GPU produces RESULT. Race condition?
                    # Ingestion sends INIT -> Assembler. Ingestion sends Seg -> Batcher -> GPU -> Assembler.
                    # INIT path is direct. SEG path is long. INIT should arrive first almost always.
                    # If not, we might need to buffer. For now, assume strict order or handle robustly.
                    # Real prod systems use Redis/DB for this state.
                    logger.warning(f"Received segment for unknown file {file_id} - caching temporarily or dropping")
                    continue
                    
                store = state_store[file_id]
                t_result = msg["data"]
                store["text_segments"][t_result.segment_index] = t_result.text
                store["received_count"] += 1
                
                # Check completion
                if store["received_count"] == store["total_segments"]:
                    # Reassemble Full Text
                    sorted_indices = sorted(store["text_segments"].keys())
                    full_text = " ".join([store["text_segments"][i] for i in sorted_indices])
                    
                    # Create Classification Input
                    class_input = ClassificationInput(
                        file_id=file_id,
                        full_transcript=full_text,
                        metadata=store["metadata"],
                        beep_count=store["beep_count"]
                    )
                    # INSERT_YOUR_CODE
                    # Save class_input to a file for record-keeping or audit
                 
                    try:
                        df = pd.DataFrame([{
                            "file_id": class_input.file_id,
                            "beep_count": class_input.beep_count,
                            "full_transcript": class_input.full_transcript,
                        }])
                        
                        # Append to CSV, write header only if file doesn't exist
                        header = not os.path.exists(assembler_output_file)
                        df.to_csv(assembler_output_file, mode='a', header=header, index=False)
                    except Exception as e:
                        logger.error(f"Failed to save class_input for file {file_id}: {e}")
                    classification_queue.put(class_input)
                    
                    # Cleanup
                    del state_store[file_id]
                    logger.info(f"Finished assembling {file_id}")

        except queue.Empty:
            continue

# --- Classification & Compliance Worker ---
def classification_worker(classification_queue, result_queue, config):
    # This worker calls AWS Bedrock (simulated)
    # It should be run in multiple threads/processes to handle I/O latency
    classifier = Classifier(config=config) # API Client
    
    logger.info("Classification worker started")
    
    while True:
        try:
            input_data = classification_queue.get(timeout=1.0)
            if input_data is None:
                break
                
            # 1. Call AWS Bedrock (Simulated)
            class_result = classifier.classify_full_text(input_data.full_transcript, input_data.file_id)
            
            # 2. Verify Compliance
            compliance_input = ComplianceInput(
                metadata=input_data.metadata,
                beep_count=input_data.beep_count,
                classification=class_result
            )
            result_queue.put(compliance_input)
            
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Error in classification worker: {e}")
