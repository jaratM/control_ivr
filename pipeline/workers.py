import time
import queue
import os
import tempfile
import multiprocessing

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
def ingestion_worker(input_queue, segment_queue, assembly_queue, config, ingestion_output_file, metrics=None):
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
    worker_name = multiprocessing.current_process().name
    logger.info(f"[{worker_name}] Ingestion worker started")
    
    # Initialize Minio
    minio_config = config.get('storage', {})
    minio = MinioStorage(
        endpoint=minio_config.get('endpoint'),
        access_key=minio_config.get('access_key'),
        secret_key=minio_config.get('secret_key'),
        bucket_name=minio_config.get('bucket_name'),
        secure=minio_config.get('secure', False)
    )
    
    files_processed_local = 0
    files_failed_local = 0
        
    while True:
        try:
            input_item = input_queue.get(timeout=1.0)
            if input_item is None:
                segment_queue.put(None) # Propagate stop
                logger.info(f"[{worker_name}] Shutting down. Processed: {files_processed_local}, Failed: {files_failed_local}")
                break

            file_id = None
            local_file_path = None
            file_start_time = time.time()
            
            try:
                # Handle dictionary input (from ManifestProcessor)
                if isinstance(input_item, dict):
                    call_id = input_item.get('call_id')
                    s3_path = input_item.get('s3_path_audio')
                    start_time = input_item.get('start_time')
                    file_id = call_id
                    
                    if not s3_path:
                        logger.error(f"No s3_path_audio for call {call_id}")
                        files_failed_local += 1
                        if metrics:
                            metrics.increment("files_failed")
                            metrics.record_error("ingestion", call_id or "unknown", "No s3_path_audio")
                        continue

                    # Download to temp file
                    # Determine extension
                    _, ext = os.path.splitext(s3_path)
                    if not ext:
                        ext = ".ogg" # Default
                        
                    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_file:
                        local_file_path = tmp_file.name
                    
                    # Download using Minio client
                    download_start = time.time()
                    if not minio.download_file(s3_path, local_file_path):
                        logger.error(f"Failed to download {s3_path}")
                        files_failed_local += 1
                        if metrics:
                            metrics.increment("download_failed")
                            metrics.increment("files_failed")
                            metrics.record_error("download", file_id, f"Failed to download {s3_path}")
                        continue
                    
                    download_duration = time.time() - download_start
                    if metrics:
                        metrics.increment("files_downloaded")
                    logger.debug(f"[{worker_name}] Downloaded {file_id} in {download_duration:.2f}s")
                    
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
                
                
                # 3. Notify Assembler (Start of file)
                assembly_queue.put({
                    "type": "FILE_INIT",
                    "file_id": file_id,
                    "numero_commande": input_item.get('numero_commande'),
                    "metadata": metadata,
                    "beep_count": number_low_bips,
                    "total_segments": len(segments),
                    "high_beeps": number_high_beeps
                })
                
                # Save ingestion output to file
                with open(ingestion_output_file, "a") as f:
                    f.write(json.dumps({
                        "file_id": file_id,
                        "numero_commande": input_item.get('numero_commande'),
                        "high_beeps": number_high_beeps,
                        "low_beeps": number_low_bips,
                        "segments": len(segments)
                    }) + "\n")
                    
                # 4. Send segments to batcher
                for seg in segments:
                    segment_queue.put(seg)
                
                # Record metrics for successful processing
                file_duration = time.time() - file_start_time
                files_processed_local += 1
                if metrics:
                    metrics.increment("files_processed")
                    metrics.increment("segments_created", len(segments))
                    metrics.record_time("ingestion", file_id, file_duration)
            
            except Exception as e:
                logger.error(f"Error processing {file_id}: {e}")
                files_failed_local += 1
                if metrics:
                    metrics.increment("files_failed")
                    metrics.record_error("ingestion", file_id or "unknown", str(e))
            
            finally:
                # Cleanup temp file if we created one
                if isinstance(input_item, dict) and local_file_path and os.path.exists(local_file_path):
                    os.remove(local_file_path)

        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Error in ingestion worker loop: {e}")

# --- Batcher Worker ---
def batcher_worker(segment_queue, transcription_queue, config, metrics=None):
    batch_size = config['pipeline']['batch_size']
    timeout = config['pipeline']['batch_timeout_ms'] / 1000.0
    
    current_batch = []
    last_flush = time.time()
    batches_created = 0
    segments_received = 0
        
    num_ingestion = config['pipeline']['num_ingestion_workers']
    stop_signals_received = 0
    
    logger.info(f"Batcher worker started (batch_size={batch_size}, timeout={timeout}s)")
    
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
                        batches_created += 1
                        if metrics:
                            metrics.increment("batches_created")
                        logger.info(f"Final batch flushed with {len(current_batch)} segments")
                    logger.info(f"Batcher shutting down. Total batches: {batches_created}, Segments: {segments_received}")
                    break
                continue
            
            segments_received += 1
            current_batch.append(item)
            
            if len(current_batch) >= batch_size:
                logger.info(f"Putting batch of {len(current_batch)} segments to transcription queue")
                transcription_queue.put(current_batch)
                batches_created += 1
                if metrics:
                    metrics.increment("batches_created")
                current_batch = []
                last_flush = time.time()
                
        except queue.Empty:
            # Check timeout flush
            if current_batch and (time.time() - last_flush > timeout):
                logger.debug(f"Timeout flush: batch of {len(current_batch)} segments")
                transcription_queue.put(current_batch)
                batches_created += 1
                if metrics:
                    metrics.increment("batches_created")
                current_batch = []
                last_flush = time.time()
                
# --- GPU Worker ---
def gpu_worker(transcription_queue, assembly_queue, config, gpu_id, metrics=None):
    # Initialize Models on specific GPU
    worker_name = multiprocessing.current_process().name
    
    logger.info(f"[{worker_name}] GPU Worker starting on device {gpu_id}")
    device = f"cuda:{gpu_id}" if gpu_id != "cpu" else "cpu"
    
    model_load_start = time.time()
    transcriber = Transcriber(device=device, config=config)
    model_load_time = time.time() - model_load_start
    logger.info(f"[{worker_name}] Model loaded in {model_load_time:.2f}s")
    
    batches_processed = 0
    segments_transcribed = 0
    total_transcription_time = 0.0
    
    while True:
        try:
            batch = transcription_queue.get(timeout=1.0)
            if batch is None:
                # We don't signal assembly queue here, orchestrator handles shutdown flow
                logger.info(f"[{worker_name}] Shutting down. Batches: {batches_processed}, "
                           f"Segments: {segments_transcribed}, "
                           f"Total time: {total_transcription_time:.2f}s")
                break
            
            # Transcribe with timing
            batch_start = time.time()
            transcripts = transcriber.transcribe_batch(batch)
            batch_duration = time.time() - batch_start
            
            batches_processed += 1
            segments_transcribed += len(transcripts)
            total_transcription_time += batch_duration
            
            if metrics:
                metrics.increment("transcriptions_completed", len(transcripts))
                for t_result in transcripts:
                    metrics.record_time("transcription", t_result.file_id, batch_duration / len(batch))
            
            # Log batch performance periodically
            if batches_processed % 10 == 0:
                avg_per_segment = batch_duration / len(batch) if batch else 0
                logger.info(f"[{worker_name}] Batch {batches_processed}: {len(batch)} segments in {batch_duration:.2f}s "
                           f"({avg_per_segment:.3f}s/segment)")
            
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
            logger.error(f"[{worker_name}] Error in GPU worker: {e}")
            if metrics:
                metrics.increment("transcription_errors")
                metrics.record_error("transcription", "batch", str(e))

# --- Assembler Worker ---
def assembler_worker(assembly_queue, classification_queue, config, assembler_output_file, metrics=None):
    # State: file_id -> { metadata, beep_count, total_segments, received_count, text_segments: {index: text}, start_time }
    state_store = {}
    assemblies_completed = 0
    segments_received = 0
    
    logger.info("Assembler worker started")
    
    while True:
        try:
            msg = assembly_queue.get(timeout=1.0)
            
            if msg is None:
                logger.info(f"Assembler shutting down. Assemblies completed: {assemblies_completed}, "
                           f"Segments received: {segments_received}, "
                           f"Files in progress: {len(state_store)}")
                break
                
            msg_type = msg["type"]
            file_id = msg["file_id"]
            
            if msg_type == "FILE_INIT":
                state_store[file_id] = {
                    "metadata": msg["metadata"],
                    "beep_count": msg["beep_count"],
                    "total_segments": msg["total_segments"],
                    "text_segments": {},
                    "received_count": 0,
                    "numero_commande": msg.get("numero_commande"),
                    "start_time": time.time()  # Track assembly start time
                }
                logger.debug(f"FILE_INIT: {file_id} ({msg['total_segments']} segments expected)")
            elif msg_type == "SEGMENT_RESULT":
                segments_received += 1
                
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
                    assembly_duration = time.time() - store["start_time"]
                    
                    # Reassemble Full Text
                    sorted_indices = sorted(store["text_segments"].keys())
                    full_text = " ".join([store["text_segments"][i] for i in sorted_indices])
                    
                    # Create Classification Input
                    class_input = ClassificationInput(
                        numero_commande=store.get("numero_commande"),
                        file_id=file_id,
                        full_transcript=full_text,
                        metadata=store["metadata"],
                        beep_count=store["beep_count"]
                    )
                 
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
                    
                    # Record metrics
                    assemblies_completed += 1
                    if metrics:
                        metrics.increment("assemblies_completed")
                        metrics.record_time("assembly", file_id, assembly_duration)
                    
                    # Cleanup
                    del state_store[file_id]
                    logger.info(f"Assembled {file_id} ({store['total_segments']} segments, {assembly_duration:.2f}s)")

        except queue.Empty:
            continue

# --- Classification & Compliance Worker ---
def classification_worker(classification_queue, result_queue, config, metrics=None):
    # This worker calls AWS Bedrock (simulated)
    # It should be run in multiple threads/processes to handle I/O latency
    # classifier = Classifier(config=config) # API Client
    
    worker_name = multiprocessing.current_process().name
    logger.info(f"[{worker_name}] Classification worker started")
    
    classifications_completed = 0
    total_classification_time = 0.0
    
    while True:
        try:
            input_data = classification_queue.get(timeout=1.0)
            if input_data is None:
                logger.info(f"[{worker_name}] Shutting down. Classifications: {classifications_completed}, "
                           f"Total time: {total_classification_time:.2f}s")
                break
            
            classification_start = time.time()
                
            # 1. Call AWS Bedrock (Simulated)
            # class_result = classifier.classify_full_text(input_data.full_transcript, input_data.file_id)
            class_result = {
                "status": "Silence",
                "behavior": "Bien",
                "file_id": input_data.file_id
            }
            
            classification_duration = time.time() - classification_start
            
            # 2. Verify Compliance
            compliance_input = ComplianceInput(
                numero_commande=input_data.numero_commande,
                metadata=input_data.metadata,
                beep_count=input_data.beep_count,
                classification=class_result
            )
            result_queue.put(compliance_input)
            
            # Record metrics
            classifications_completed += 1
            total_classification_time += classification_duration
            if metrics:
                metrics.increment("classifications_completed")
                metrics.record_time("classification", input_data.file_id, classification_duration)
            
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"[{worker_name}] Error in classification worker: {e}")
            if metrics:
                metrics.increment("classification_errors")
                metrics.record_error("classification", 
                                    input_data.file_id if 'input_data' in dir() else "unknown", 
                                    str(e))
