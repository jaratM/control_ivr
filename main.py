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
                        # Ensure subdirectories exist in temp_dir if manifest_file contains paths
                        os.makedirs(os.path.dirname(local_path), exist_ok=True)
                        
                        minio.download_file(manifest_file, local_path)
                        logger.info(f"Processing manifest: {manifest_file}")
                        df, calls_metadata = processor.process_manifest(local_path, '2025-11-19')
                        output_dir = "data/manifests/19-11-2025"
                        os.makedirs(output_dir, exist_ok=True)
                        output_filename = os.path.basename(manifest_file)
                        output_path = os.path.join(output_dir, output_filename)
                        # Save calls_metadata as a JSON file in the same output_dir as the manifest
                        import json
                        calls_metadata_path = os.path.splitext(output_path)[0] + "_calls_metadata.json"
                        with open(calls_metadata_path, "w", encoding="utf-8") as f:
                            json.dump(calls_metadata, f, ensure_ascii=False, indent=2, default=str)
                        manifest_type = 'ADSL_INSTALL' if 'crc_adsl' in manifest_file.lower() else 'FTTH_VULA_INSTALL'
                        # df.to_csv(output_path, index=False)
                        results = orchestrator.run(df, calls_metadata, manifest_type)
                finally:
                    db.close()
        else:
            logger.info("No manifest files found in Minio.")

    except Exception as e:
        logger.error(f"Error processing manifests from Minio: {e}")


if __name__ == "__main__":
    main()
