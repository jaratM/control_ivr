import os
import multiprocessing
import argparse
import tempfile
from io import BytesIO
from contextlib import contextmanager
from typing import Optional, List, Dict, Tuple

import yaml
import pandas as pd
from loguru import logger

from database.models import ManifestStatus
from database.connection import SessionLocal, init_db
from pipeline.orchestrator import PipelineOrchestrator
from services.ingest import IngestionService
from services.manifest import ManifestProcessor
from storage.minio_client import MinioStorage
from datetime import datetime
from pipeline.metrics import PipelineMetrics
from database.database_manager import get_manifest, update_manifest_status, get_manifest_types, get_manifest_calls
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


def get_column_mapping(config: dict, manifest_type: str, category: str) -> Dict[str, str]:
    """
    Get the column mapping from config for renaming DataFrame columns.
    Returns a dict mapping db_column_name -> display_name.
    
    Config format: result_columns.<manifest_type>.<category>.<db_column>: <display_name>
    """
    result_columns = config.get('result_columns', {})
    
    # Normalize manifest type key (config uses 'Acquisition' with capital A)
    manifest_key = 'Acquisition' if manifest_type == 'ACQUISITION' else manifest_type
    
    manifest_config = result_columns.get(manifest_key, {})
    category_config = manifest_config.get(category, {})
    
    if not category_config:
        logger.warning(f"No column mapping found for {manifest_type}/{category}")
        return {}
    
    # Config format is {db_column: display_name} - use directly
    column_mapping = {}
    for db_column, display_name in category_config.items():
        if display_name:
            column_mapping[str(db_column)] = str(display_name)
    
    return column_mapping


def create_result_dataframe(
    manifest_calls: List[dict],
    config: dict,
    manifest_type: str,
    category: str
) -> pd.DataFrame:
    """
    Create a DataFrame from manifest calls and rename columns based on config.
    """
    if not manifest_calls:
        logger.warning(f"No manifest calls to create DataFrame for {manifest_type}/{category}")
        return pd.DataFrame()
    
    # Create DataFrame from serialized data
    df = pd.DataFrame(manifest_calls)
    
    # Get column mapping and rename
    column_mapping = get_column_mapping(config, manifest_type, category)
    
    if column_mapping:
        # Only rename columns that exist in the DataFrame and have mappings
        rename_dict = {db_col: display_name for db_col, display_name in column_mapping.items() if db_col in df.columns}
        df = df.rename(columns=rename_dict)
        
        # Keep only columns from config, in the same order as defined
        ordered_columns = [display_name for display_name in column_mapping.values() if display_name in df.columns]
        df = df[ordered_columns]
    
    return df


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
    if 'sav' in filename.lower():
        filename = filename.replace('.xlsx', f'_{suffix}.xlsx')
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
        
    df_dict, calls_metadata, base_manifest_type, category = processor.process_manifest(
        local_path, target_date, manifest_record
    )

    if len(df_dict) == 0:
        logger.warning(f"No data to be processed for {manifest_file}. Skipping...")
        update_manifest_status(db, manifest_record.id, ManifestStatus.PROCESSING)
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
    logger.info(f"Processing {len(manifest_files)} manifests")
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
            return True
    return False

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
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
        
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

    # date = '2026-01-25'
    date = datetime.now() - timedelta(days=1)
    date = date.strftime('%Y-%m-%d')
    
    if process_manifests(config, args.config, date, prefix=args.prefix):
        output_dir = f"output/{date}"
        
        result_dataframes: List[Tuple[str, str, pd.DataFrame]] = []
        
        with get_db_session() as db:
            res = get_manifest_types(db, date)
            
            for categorie, filename, manifest_id in res:
                # Determine manifest type based on filename
                if 'crc_adsl' in filename.lower() or 'crc_vula' in filename.lower():
                    manifest_type = 'ACQUISITION'
                else:
                    manifest_type = 'SAV'
                logger.info(f"filename: {filename}, Manifest type: {manifest_type}, categorie: {categorie}")
                # Get manifest calls and create DataFrame
                manifest_calls = get_manifest_calls(db, manifest_id, categorie)
                
                if not manifest_calls:
                    logger.warning(f"No calls found for manifest_id={manifest_id}, categorie={categorie}")
                    continue
                
                # Create DataFrame with renamed columns
                df = create_result_dataframe(manifest_calls, config, manifest_type, categorie)
                
                if not df.empty:
                    result_dataframes.append((manifest_type, categorie, df))
                    logger.info(f"Created DataFrame for {manifest_type}/{categorie} with {len(df)} rows")
        
        # Save DataFrames to Minio
        minio = init_minio(config)
        for manifest_type, categorie, df in result_dataframes:
            object_name = f"{output_dir}/{manifest_type}_{categorie}_{date}.xlsx"
            
            # Write Excel to BytesIO buffer
            excel_buffer = BytesIO()
            df.to_excel(excel_buffer, index=False, engine='openpyxl',)
            excel_data = excel_buffer.getvalue()
            
            # Save results locally as well
            local_output_dir = f"{output_dir}"
            os.makedirs(local_output_dir, exist_ok=True)
            local_excel_path = os.path.join(local_output_dir, f"{manifest_type}_{categorie}_{date}.xlsx")
            with open(local_excel_path, "wb") as f:
                f.write(excel_data)
            logger.info(f"Saved results locally: {local_excel_path}")
            
            minio.upload_bytes(excel_data, object_name, content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            logger.info(f"Saved results to Minio: {object_name}")
                
if __name__ == "__main__":
    main()
