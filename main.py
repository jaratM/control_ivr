import os
import argparse
import json
import yaml
import tempfile
from datetime import datetime
from typing import Optional, List

from loguru import logger
from database.models import ManifestStatus
from pipeline.orchestrator import PipelineOrchestrator
from storage.minio_client import MinioStorage
from services.manifest import ManifestProcessor
from database.connection import SessionLocal
from services.ingest import IngestionService
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

    logger.info("="*20 + " Fetching manifest files from Minio " + "="*20 + "\n")
    try:
        all_objects = minio.list_objects()
        manifest_files = [f for f in all_objects if f.endswith('.xlsx') or f.endswith('.csv')]    
    except Exception as e:
        logger.error(f"Error listing objects from Minio: {e}")
        return

    if not manifest_files:
        logger.info("No manifest files found in Minio.\n")
        return

    # Create output directory for the target date
    output_base_dir = f"output/{target_date}/"
    os.makedirs(output_base_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        db = SessionLocal()
        try:
            processor = ManifestProcessor(db, config)
            
            for manifest_file in manifest_files:
                logger.info(f"Processing manifest: {manifest_file}")
                
                local_path = os.path.join(temp_dir, manifest_file)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                
                try:
                    minio.download_file(manifest_file, local_path)
                except Exception as e:
                    logger.error(f"Failed to download {manifest_file}: {e}")
                    continue

                try:
                    # Process manifest
                    # processor.process_manifest returns: df, calls_metadata, manifest_type, category
                    logger.info(f"Processing manifest: {manifest_file}")
                    df, calls_metadata, base_manifest_type, category, manifest_record = processor.process_manifest(local_path, target_date)
                    if df.empty:
                        logger.warning(f"No data processed for {manifest_file}")
                        continue
                    logger.info(f"Manifest: {manifest_file} filtered and ready to be processed")
                    
                    # Run orchestration
                    # Pass base_manifest_type (ACQUISITION or SAV) to ensure correct config lookup in ComplianceVerifier
                    success = orchestrator.run(df, calls_metadata, base_manifest_type, category, output_base_dir)
                    if not success:
                        logger.error(f"Orchestration failed for {manifest_file}")
                        manifest_record.status = ManifestStatus.FAILED
                        db.commit()
                        continue
                    manifest_record.status = ManifestStatus.COMPLETED
                    db.commit()
                    logger.info(f"================================================")
                    logger.info(f"========== Orchestration completed for manifest: {manifest_file} ==========")
                    logger.info(f"================================================")
                except Exception as e:
                    logger.error(f"Error processing {manifest_file}: {e}")
                    # Continue to next file

                if not process_all:
                    logger.info("Stopping after first manifest (use --all to process all).")
                    break

        finally:
            db.close()

def run_ingestion(config: dict, ingest_minio: str):
    """
    Run ingestion of the Minio bucket and index the calls in the database.
    """
    logger.info(f"Starting ingestion of the Minio bucket: {ingest_minio}")
    db = SessionLocal()
    minio = init_minio(config)
    ingestion_service = IngestionService(db, minio)
    ingestion_service.ingest_bucket(ingest_minio)
    db.close()
    logger.info(f"Ingestion completed for the Minio bucket: {ingest_minio}")

def main():
    parser = argparse.ArgumentParser(description="Audio Compliance Verification Pipeline")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration file")
    parser.add_argument("--date", type=str, default="19-11-2025", help="Date string for processing (format: DD-MM-YYYY or YYYY-MM-DD)")
    parser.add_argument("--all", action="store_true", help="Process all found manifests instead of just the first one")
    parser.add_argument("--ingest-minio", type=str, default="", help="Ingest Minio bucket name (optional)")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    config['ingest_minio'] = args.ingest_minio

    if args.ingest_minio:
        run_ingestion(config, args.ingest_minio)
    # Run processing
    process_manifests(config, args.config, args.date, process_all=args.all)

if __name__ == "__main__":
    main()
