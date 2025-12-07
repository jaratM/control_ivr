import os
import argparse
import tempfile
from contextlib import contextmanager
from typing import Optional

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


def fetch_manifest_files(minio: MinioStorage) -> list[str]:
    """Fetch manifest files from Minio storage."""
    logger.info("=" * 20 + " Fetching manifest files from Minio " + "=" * 20 + "\n")
    all_objects = minio.list_objects()
    return [f for f in all_objects if f.endswith('.xlsx') or f.endswith('.csv')]


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

    df, calls_metadata, base_manifest_type, category, manifest_record = processor.process_manifest(
        local_path, target_date
    )
    
    if df.empty:
        logger.warning(f"No data processed for {manifest_file}")
        return False
    
    logger.info(f"Manifest: {manifest_file} filtered and ready to be processed")
    
    metrics = orchestrator.run(df, calls_metadata, base_manifest_type, category, output_base_dir, metrics)
    
    if not metrics:
        logger.error(f"Orchestration failed for {manifest_file}")
        manifest_record.status = ManifestStatus.FAILED
        manifest_record.processed_at = datetime.now()
        db.commit()
        return False
    
    manifest_record.status = ManifestStatus.COMPLETED
    manifest_record.processed_at = datetime.now()
    db.commit()
    
    logger.info("=" * 48)
    logger.info(f"========== Orchestration completed for manifest: {manifest_file} ==========")
    logger.info("=" * 48)
    metrics.log_summary()
    return True


def process_manifests(config: dict, config_path: str, target_date: str, process_all: bool = False):
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
        manifest_files = fetch_manifest_files(minio)
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

                if not process_all:
                    logger.info("Stopping after first manifest (use --all to process all).")
                    break


def run_ingestion(config: dict, ingest_minio: str):
    """Run ingestion of the Minio bucket and index the calls in the database."""
    logger.info(f"Starting ingestion of the Minio bucket: {ingest_minio}")
    
    minio = init_minio(config)
    
    with get_db_session() as db:
        ingestion_service = IngestionService(db, minio)
        ingestion_service.ingest_bucket(ingest_minio)
    
    logger.info(f"Ingestion completed for the Minio bucket: {ingest_minio}")


def main():
    parser = argparse.ArgumentParser(description="Audio Compliance Verification Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--date",
        type=str,
        default="19-11-2025",
        help="Date string for processing (format: DD-MM-YYYY or YYYY-MM-DD)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all found manifests instead of just the first one"
    )
    parser.add_argument(
        "--ingest-minio",
        type=str,
        default="",
        help="Ingest Minio bucket name (optional)"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config['ingest_minio'] = args.ingest_minio
    
    try:
        init_db()
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
    
    if args.ingest_minio:
        run_ingestion(config, args.ingest_minio)
    
    process_manifests(config, args.config, args.date, process_all=args.all)


if __name__ == "__main__":
    main()
