import multiprocessing
import yaml
import glob
import time
import pandas as pd
from typing import List
from loguru import logger
from .workers import ingestion_worker, batcher_worker, gpu_worker, assembler_worker, classification_worker
from .utils import setup_logging
# from modules.compliance import ComplianceVerifier, ComplianceInput
from modules.compliance import ComplianceVerifier

class PipelineOrchestrator:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        setup_logging(self.config)
        self.manager = multiprocessing.Manager()
        self.verifier = ComplianceVerifier()
        
    def run(self, df: pd.DataFrame, calls_metadata: List, manifest_type: str):
        logger.info("Starting Pipeline Orchestration")
        
        # 1. Setup Queues
        path_queue = self.manager.Queue()
        segment_queue = self.manager.Queue(maxsize=self.config['pipeline']['max_queue_size'])
        transcription_queue = self.manager.Queue(maxsize=self.config['pipeline']['max_queue_size'])
        assembly_queue = self.manager.Queue()
        classification_queue = self.manager.Queue()
        result_queue = self.manager.Queue()
        
        # 2. Discover Files
        logger.info(f"Found {len(calls_metadata)} calls")
        
        compliance_df = self.verifier.verify_compliance(df, calls_metadata, manifest_type)
        compliance_df.to_csv(f"data/compliance_df{manifest_type}.csv", index=False)
        
        for call in calls_metadata:
            if call and len(call) > 0:
                path_queue.put(call[0])

        # Poison pills for Ingestion Workers
        num_ingestion = self.config['pipeline']['num_ingestion_workers']
        for _ in range(num_ingestion):
            path_queue.put(None)
            
        # 3. Start Workers
        ingestion_procs = []
        gpu_procs = []
        class_procs = []
        
        # Ingestion Workers
        for _ in range(num_ingestion):
            p = multiprocessing.Process(
                target=ingestion_worker,
                args=(path_queue, segment_queue, assembly_queue, self.config)
            )
            p.start()
            ingestion_procs.append(p)
            
        # Batcher Worker
        batcher = multiprocessing.Process(
            target=batcher_worker,
            args=(segment_queue, transcription_queue, self.config)
        )
        batcher.start()
        
        # GPU Workers
        num_gpus = 2 if self.config['gpu']['use_multi_gpu'] else 0
        gpu_worker_count = num_gpus if num_gpus > 0 else 1
        device_list = list(range(num_gpus)) if num_gpus > 0 else ["cpu"]
        
        for i in range(gpu_worker_count):
            device_id = device_list[i]
            p = multiprocessing.Process(
                target=gpu_worker,
                args=(transcription_queue, assembly_queue, self.config, device_id)
            )
            p.start()
            gpu_procs.append(p)
            
        # Assembler Worker (Reassembles transcripts)
        assembler = multiprocessing.Process(
            target=assembler_worker,
            args=(assembly_queue, classification_queue, self.config)
        )
        assembler.start()
        
        # Classification Workers (AWS Bedrock - I/O Bound)
        # We can scale this higher than GPUs because it's just waiting on HTTP
        num_classifiers = 8 
        for _ in range(num_classifiers):
            p = multiprocessing.Process(
                target=classification_worker,
                args=(classification_queue, result_queue, self.config)
            )
            p.start()
            class_procs.append(p)
        
        # 4. Wait for processing to finish
        
        # Wait for Ingestion
        for p in ingestion_procs:
            p.join()
            
        # Wait for Batcher
        batcher.join()
        
        # Signal GPU workers to stop
        for _ in range(gpu_worker_count):
            transcription_queue.put(None)
            
        # Wait for GPU workers
        for p in gpu_procs:
            p.join()
            
        # Signal Assembler to stop
        # Since all GPU workers are done, they have sent all their SEGMENT_RESULTS.
        # Ingestion is done, so all FILE_INITs are sent.
        # Assembler just needs to clear the queue.
        assembly_queue.put(None)
        assembler.join()
        
        # Signal Classification Workers
        for _ in range(num_classifiers):
            classification_queue.put(None)
            
        for p in class_procs:
            p.join()
        
        # 5. Collect Results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
            
        logger.info(f"Pipeline finished. Generated {len(results)} results.")
        return results
