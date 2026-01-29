import multiprocessing
import yaml
import glob
import time
import pandas as pd
from typing import List
from loguru import logger
from sqlalchemy.orm import Session
from .workers import ingestion_worker, batcher_worker, gpu_worker, assembler_worker, classification_worker
from .utils import setup_logging
from .metrics import PipelineMetrics
from modules.compliance import ComplianceVerifier
from datetime import datetime
from database.database_manager import get_manifest_call, bulk_insert_manifest_calls
import torch
from services.email_service import EmailService
# from modules.results import ComplianceResults
class PipelineOrchestrator:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Setup logging and store the log file path for workers
        log_file_path = setup_logging(self.config)
        self.config['_log_file_path'] = log_file_path
        self.manager = multiprocessing.Manager()
        self.verifier = ComplianceVerifier()
        self.email_service = EmailService(self.config)
        # self.compliance_results = ComplianceResults(self.config) 
        
    def run(self, df_dict: List[dict], calls_metadata: List, manifest_type: str, category: str, output_path: str, metrics: PipelineMetrics, db: Session):
        """
        Run the pipeline orchestration.
        - Verify the compliance
        - Add the calls to the path queue
        - Start the workers
        - Wait for the workers to finish
        - Collect the results
        """
        logger.info(f"Starting Pipeline Orchestration for {manifest_type} {category}")
        
        # 1. Setup Queues
        path_queue = self.manager.Queue()
        segment_queue = self.manager.Queue(maxsize=self.config['pipeline']['max_queue_size'])
        transcription_queue = self.manager.Queue(maxsize=self.config['pipeline']['max_queue_size'])
        assembly_queue = self.manager.Queue()
        classification_queue = self.manager.Queue()
        result_queue = self.manager.Queue()
        
        # Initialize metrics tracking
        
        # df_dict = self.verifier.verify_compliance(df_dict, calls_metadata, category, manifest_type, self.config)
        # compliance_df = pd.DataFrame(df_dict)
        # logger.info(f"Saving compliance dataframe to {output_path}compliance_df_{manifest_type}_{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        # compliance_df.to_csv(f"{output_path}/compliance_df_{manifest_type}_{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)
        
        # Queue files and track count

        counter = 0
        for i,row in enumerate(df_dict):
            manifest_call = get_manifest_call(db, row['numero_commande'])
            processed = manifest_call.processed if manifest_call else False 
            if processed:
                logger.info(f"{row['numero_commande'] } already processed")
                continue
            if len(calls_metadata[i]) > 0:
                for call in calls_metadata[i]:
                    path_queue.put(call)
                    counter += 1
            else:
                df_dict[i]['conformite_IAM'] = 'Non Conforme'
                df_dict[i]['commentaire'] = 'Aucun appel trouvé'
        
        metrics.increment("files_queued", counter)
        logger.info(f"Adding {counter} calls to path queue")
        
        # Poison pills for Ingestion Workers
        num_ingestion = self.config['pipeline']['num_ingestion_workers']
        for _ in range(num_ingestion):
            path_queue.put(None)
            
        # 3. Start Workers
        ingestion_procs = []
        gpu_procs = []
        class_procs = []
        
        ingestion_output_file = f"{output_path}ingestion_output_{manifest_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        logger.info(f"Saving ingestion output to {ingestion_output_file}")
        # Ingestion Workers
        for i in range(num_ingestion):
            p = multiprocessing.Process(
                target=ingestion_worker,
                args=(path_queue, segment_queue, assembly_queue, self.config, ingestion_output_file, metrics),
                name=f"Ingestion-{i}"
            )
            p.start()
            ingestion_procs.append(p)
            
        # Batcher Worker
        logger.info(f"Starting Batcher Worker")
        batcher = multiprocessing.Process(
            target=batcher_worker,
            args=(segment_queue, transcription_queue, self.config, metrics),
            name="Batcher"
        )
        batcher.start()
        
        # GPU Workers
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if not self.config['gpu'].get('use_multi_gpu', False) and num_gpus > 0:
            num_gpus = 1
            
        gpu_worker_count = num_gpus if num_gpus > 0 else 1
        device_list = list(range(num_gpus)) if num_gpus > 0 else ["cpu"]
        
        for i in range(gpu_worker_count):
            device_id = device_list[i]
            p = multiprocessing.Process(
                target=gpu_worker,
                args=(transcription_queue, assembly_queue, self.config, device_id, metrics),
                name=f"GPU-{device_id}"
            )
            p.start()
            gpu_procs.append(p)
            
        # Assembler Worker (Reassembles transcripts)
        assembler_output_file = f"{output_path}assembler_output_{manifest_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        logger.info(f"Saving assembler output to {assembler_output_file}")      
        assembler = multiprocessing.Process(
            target=assembler_worker,
            args=(assembly_queue, classification_queue, self.config, assembler_output_file, metrics),
            name="Assembler"
        )
        assembler.start()
        
        # Classification Workers (AWS Bedrock - I/O Bound)
        # We can scale this higher than GPUs because it's just waiting on HTTP
        num_classifiers = self.config['pipeline'].get('num_classification_workers', 4)
        for i in range(num_classifiers):
            p = multiprocessing.Process(
                target=classification_worker,
                args=(classification_queue, result_queue, self.config, metrics, manifest_type),
                name=f"Classifier-{i}"
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

        compliance_list = self.verifier.verify_compliance_batch(df_dict, results)
        logger.info(f"Pipeline finished. Generated {len(results)} results.")
        
        # Bulk upsert all manifest calls (handles both new and existing records)
        logger.info(f"Upserting {len(compliance_list)} manifest calls")
        logger.debug(f'{compliance_list}')
        bulk_insert_manifest_calls(db, compliance_list)
        
        return metrics
