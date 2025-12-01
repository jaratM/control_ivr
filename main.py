import os
import argparse
from pipeline.orchestrator import PipelineOrchestrator
from loguru import logger
import yaml
from storage.minio_client import MinioStorage
from datetime import datetime



def main():
    parser = argparse.ArgumentParser(description="Audio Compliance Verification Pipeline")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration file")
    
    args = parser.parse_args()
    

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    orchestrator = PipelineOrchestrator(args.config)
    # Initialize Minio and process manifests
    try:
        minio_config = config['storage']
        minio = MinioStorage(
            endpoint=minio_config['endpoint'],
            access_key=minio_config['access_key'],
            secret_key=minio_config['secret_key'],
            bucket_name=minio_config['bucket_name'],
            secure=minio_config['secure']
        )

        logger.info("Fetching manifest files from Minio...")
        manifest_files = [f for f in minio.list_objects() if f.endswith('.csv')]
        
        if manifest_files:
            import tempfile
            from services.manifest import ManifestProcessor
            from database.connection import SessionLocal

            with tempfile.TemporaryDirectory() as temp_dir:
                db = SessionLocal()
                try:
                    processor = ManifestProcessor(db, config)
                    
                    for manifest_file in manifest_files:
                        local_path = os.path.join(temp_dir, manifest_file)
                        minio.download_file(manifest_file, local_path)
                        logger.info(f"Processing manifest: {manifest_file}")
                        df, calls_metadata = processor.process_manifest(local_path, '2025-11-19')
                        results = orchestrator.run(df, calls_metadata)
                finally:
                    db.close()
        else:
            logger.info("No manifest files found in Minio.")

    except Exception as e:
        logger.error(f"Error processing manifests from Minio: {e}")

    return 0
    
    # Output results summary
    compliant_count = sum(1 for r in results if r.is_compliant)
    logger.info(f"Summary: {compliant_count}/{len(results)} files compliant.")
    
    for res in results:
        if not res.is_compliant:
            logger.warning(f"Non-compliant file {res.file_id}: {res.issues}")

if __name__ == "__main__":
    main()
