import os
import argparse
import tempfile
from contextlib import contextmanager
from typing import Optional, List

import yaml
from loguru import logger

from database.models import ManifestStatus
from database.connection import SessionLocal, init_db
from pipeline.orchestrator import PipelineOrchestrator
from services.ingest import IngestionService
from services.manifest import ManifestProcessor
from storage.minio_client import MinioStorage
from datetime import datetime
from pipeline.metrics import PipelineMetrics
from database.database_manager import  get_manifest, update_manifest_status
import uuid
from database.models import Manifest
from datetime import timedelta


def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        raise


def init_minio(config: dict) -> MinioStorage:
    """Initialize Minio storage client."""
    minio_config = config['storage']
    return MinioStorage(
        endpoint=minio_config['endpoint'],
        access_key=minio_config['access_key'],
        secret_key=minio_config['secret_key'],
        bucket_name=minio_config['bucket_name'],
        secure=minio_config['secure']
    )


@contextmanager
def get_db_session():
    """Context manager for database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def fetch_manifest_files(minio: MinioStorage, target_date: str = "", prefix: str = "servicenow") -> List[str]:
    """Fetch manifest files from Minio storage."""
    logger.info("=" * 20 + " Fetching manifest files from Minio for date: " + target_date + "=" * 20 + "\n")
    suffix = f'{target_date}'.replace('-','')
    all_objects = minio.list_objects(prefix=prefix)
    return [f for f in all_objects if f.endswith(f'.xlsx') or f.endswith(f'{suffix}.csv')]


def download_manifest(minio: MinioStorage, manifest_file: str, temp_dir: str) -> Optional[str]:
    """Download a manifest file from Minio to local temp directory."""
    local_path = os.path.join(temp_dir, manifest_file)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    try:
        minio.download_file(manifest_file, local_path)
        return local_path
    except Exception as e:
        logger.error(f"Failed to download {manifest_file}: {e}")
        return None


def process_single_manifest(
    processor: ManifestProcessor,
    orchestrator: PipelineOrchestrator,
    local_path: str,
    manifest_file: str,
    target_date: str,
    output_base_dir: str,
    db
) -> bool:
    """
    Process a single manifest file.
    
    Returns True if processing succeeded, False otherwise.
    """
    logger.info(f"Processing manifest: {manifest_file}")
    metrics = PipelineMetrics(orchestrator.manager)

    manifest_id = str(uuid.uuid4())
    filename = local_path.split('/')[-1]
    suffix = f'{target_date}'.replace('-','')
    manifest_record = get_manifest(db, filename)
    if manifest_record: 
        if manifest_record.status == ManifestStatus.COMPLETED:
            logger.info(f"Manifest {filename} already exists and is completed")
            return True
        else:
            manifest_record.status = ManifestStatus.PROCESSING
            manifest_record.received_at = datetime.now()
            db.add(manifest_record)
            db.commit()
        
    else:
        manifest_record = Manifest(
                id=manifest_id,
                filename=filename,
                status=ManifestStatus.PROCESSING,
                received_at=datetime.now()
        )
        db.add(manifest_record)
        db.commit()
        
    df_dict, calls_metadata, base_manifest_type, category, manifest_record = processor.process_manifest(
        local_path, target_date, manifest_record
    )

    if len(df_dict) == 0:
        logger.warning(f"No data processed for {manifest_file}")
        update_manifest_status(db, manifest_record.id, ManifestStatus.FAILED)
        return False        # No data processed
    
    logger.info(f"Manifest: {manifest_file} filtered and ready to be processed")
    
    metrics = orchestrator.run(df_dict, calls_metadata, base_manifest_type, category, output_base_dir, metrics, db)
    
    if not metrics:
        logger.error(f"Orchestration failed for {manifest_file}")
        update_manifest_status(db, manifest_record.id, ManifestStatus.FAILED)
        return False
    
    update_manifest_status(db, manifest_record.id, ManifestStatus.COMPLETED, processed_at=datetime.now())
    
    logger.info("=" * 48)
    logger.info(f"========== Orchestration completed for manifest: {manifest_file} ==========")
    logger.info("=" * 48)
    metrics.log_summary()
    return True


def process_manifests(config: dict, config_path: str, target_date: str, prefix: str = ""):
    """
    Main processing logic:
    1. Connect to Minio
    2. List manifests
    3. Process each manifest
    """
    orchestrator = PipelineOrchestrator(config_path)
    
    try:
        minio = init_minio(config)
    except Exception as e:
        logger.error(f"Failed to initialize Minio: {e}")
        return

    try:
        
        manifest_files = fetch_manifest_files(minio, target_date=target_date, prefix='servicenow')
    except Exception as e:
        logger.error(f"Error listing objects from Minio: {e}")
        return

    if not manifest_files:
        logger.info("No manifest files found in Minio.\n")
        return

    output_base_dir = f"output/{target_date}/"
    os.makedirs(output_base_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        with get_db_session() as db:
            processor = ManifestProcessor(db, config)
            
            for manifest_file in manifest_files:
                local_path = download_manifest(minio, manifest_file, temp_dir)
                if not local_path:
                    continue

                try:
                    process_single_manifest(
                        processor=processor,
                        orchestrator=orchestrator,
                        local_path=local_path,
                        manifest_file=manifest_file,
                        target_date=target_date,
                        output_base_dir=output_base_dir,
                        db=db
                    )
                except Exception as e:
                    logger.error(f"Error processing {manifest_file}: {e}")


def run_ingestion(config: dict, input_folder: str = None):
    """Run ingestion of the Minio bucket and index the calls in the database."""
    logger.info(f"Starting ingestion of the input folder: {input_folder.split('/')[-1] if input_folder else ''}")
    
    minio = init_minio(config)
    with get_db_session() as db:
        ingestion_service = IngestionService(db, minio)

        if input_folder:
            ingestion_service.ingest_folder(input_folder)   
    
    logger.info(f"Ingestion completed for the input folder: {input_folder.split('/')[-1] if input_folder else ''}")


def main():
    parser = argparse.ArgumentParser(description="Audio Compliance Verification Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix to filter manifest files in Minio"
    )
    
    args = parser.parse_args()

    config = load_config(args.config)
    
    try:
        init_db()
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
    
    input = config['ingestion']['input_folder']
    if input:
        # this for indexing the calls in the database json files
        run_ingestion(config, input_folder= input)

    # date = '2025-11-19'
    date = datetime.now() - timedelta(days=1)
    date = date.strftime('%Y-%m-%d')
    process_manifests(config, args.config, date, prefix=args.prefix)

if __name__ == "__main__":
    main()
