import time
import queue
from loguru import logger
from modules.frequency import FrequencyAnalyzer
from modules.metadata import MetadataExtractor
from modules.transcription import Transcriber
from modules.classification import Classifier
from modules.compliance import ComplianceVerifier, ComplianceInput
from modules.types import AudioSegment, ClassificationInput

# --- Ingestion Worker ---
def ingestion_worker(input_queue, segment_queue, assembly_queue, config):
    freq_analyzer = FrequencyAnalyzer()
    meta_extractor = MetadataExtractor()
    
    logger.info("Ingestion worker started")
    
    while True:
        try:
            file_path = input_queue.get(timeout=1.0)
            if file_path is None:
                segment_queue.put(None) # Propagate stop
                break
                
            logger.info(f"Processing file: {file_path}")
            
            # 1. Metadata
            metadata = meta_extractor.extract(file_path)
            
            # 2. Frequency Analysis
            # Now returns 3 values: high_beeps, low_beeps, segments
            # We pass the processing config
            processing_config = config.get('processing', {})
            high_beeps, low_beeps, segments = freq_analyzer.process(file_path, metadata.file_id, processing_config)
            
            # Use low_beeps (voicemail beeps) as the primary count for compliance, or high_beeps depending on rule
            # Assuming low_beeps is relevant for start-of-call compliance
            beep_count = high_beeps 
            
            # 3. Notify Assembler (Start of file)
            assembly_queue.put({
                "type": "FILE_INIT",
                "file_id": metadata.file_id,
                "metadata": metadata,
                "beep_count": beep_count,
                "total_segments": len(segments)
            })
            
            # 4. Send segments to batcher
            for seg in segments:
                segment_queue.put(seg)
                
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Error in ingestion: {e}")

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
def assembler_worker(assembly_queue, classification_queue, config):
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
    
    classifier = Classifier() # API Client
    verifier = ComplianceVerifier()
    
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
            
            result = verifier.verify(compliance_input)
            result_queue.put(result)
            
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Error in classification worker: {e}")
