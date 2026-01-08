import os
import json
from datetime import datetime
from loguru import logger
from sqlalchemy.orm import Session
from database.models import Call, IngestionStatus
from storage.minio_client import MinioStorage

class IngestionService:
    def __init__(self, db_session: Session, minio_client: MinioStorage):
        self.db = db_session
        self.minio = minio_client


    def ingest_folder(self, input_folder: str):
        """
        Ingests a folder of files into the database.
        """
        for file_name in os.listdir(input_folder):
            file_path = os.path.join(input_folder, file_name)
            if file_name.endswith('.json') and os.path.isfile(file_path):
                self.process_json_file(file_path)
    
    def process_json_file(self, json_path: str):
        """
        Processes a local JSON file and indexes the calls in the database.
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not data:
                logger.warning(f"Empty or invalid JSON: {json_path}")
                return

            call_id = data.get("CALL_ID")
            if not call_id:
                logger.warning(f"Missing CALL_ID in {json_path}")
                return

            # Map fields
            date_fmt = "%Y-%m-%d %H:%M:%S"
            
            try:
                start_time = datetime.strptime(data.get("DATE_DEBUT", ""), date_fmt)
                end_time = datetime.strptime(data.get("DATE_FIN", ""), date_fmt)
            except ValueError:
                logger.warning(f"Invalid date format in {json_path}")
                return

            # Construct audio path based on convention (sibling file)
            audio_path = json_path.replace('.json', '.ogg')
            
            # Check if call exists
            existing_call = self.db.query(Call).filter(Call.call_id == call_id).first()
            if existing_call:
                # Update logic if needed
                existing_call.updated_at = datetime.utcnow()
                existing_call.ingestion_status = IngestionStatus.INDEXED
            else:
                new_call = Call(
                    call_id=call_id,
                    client_number=data.get("NUMERO_CLIENT"),
                    technician_number=data.get("NUMERO_TECHNICIEN"),
                    start_time=start_time,
                    end_time=end_time,
                    branch=data.get("BRANCHE"),
                    s3_path_audio=audio_path,
                    s3_path_json=json_path,
                    ingestion_status=IngestionStatus.INDEXED
                )
                self.db.add(new_call)
            
            self.db.commit()

        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to process {json_path}: {e}")
        

    # def ingest_bucket(self, prefix: str = ""):
    #     """
    #     Scans the bucket for JSON metadata files and indexes them in the DB.
    #     """
    #     # logger.info(f"Starting ingestion for prefix: '{prefix}'")
        
    #     # Iterate over all objects
    #     for object_name in self.minio.list_objects(prefix=prefix):
    #         if object_name.endswith('.json'):
    #             self.process_metadata_file(object_name)

    # def process_metadata_file(self, json_path: str):
    #     """
    #     Reads a JSON file from S3 and updates the database.
    #     """
    #     try:
    #         data = self.minio.get_json(json_path)
    #         if not data:
    #             logger.warning(f"Empty or invalid JSON: {json_path}")
    #             return

    #         call_id = data.get("CALL_ID")
    #         if not call_id:
    #             logger.warning(f"Missing CALL_ID in {json_path}")
    #             return

    #         # Map fields
    #         # "DATE_DEBUT": "2025-11-19 09:33:13"
    #         date_fmt = "%Y-%m-%d %H:%M:%S"
            
    #         try:
    #             start_time = datetime.strptime(data.get("DATE_DEBUT", ""), date_fmt)
    #             end_time = datetime.strptime(data.get("DATE_FIN", ""), date_fmt)
    #         except ValueError:
    #             logger.warning(f"Invalid date format in {json_path}")
    #             return

    #         # Construct audio path based on convention (sibling file)
    #         audio_path = json_path.replace('.json', '.ogg')
            
    #         # Check if call exists
    #         existing_call = self.db.query(Call).filter(Call.call_id == call_id).first()
            
    #         if existing_call:
    #             # Update logic if needed
    #             existing_call.updated_at = datetime.utcnow()
    #             existing_call.ingestion_status = IngestionStatus.INDEXED
    #             # ... update other fields if they might change
    #         else:
    #             new_call = Call(
    #                 call_id=call_id,
    #                 client_number=data.get("NUMERO_CLIENT"),
    #                 technician_number=data.get("NUMERO_TECHNICIEN"),
    #                 start_time=start_time,
    #                 end_time=end_time,
    #                 branch=data.get("BRANCHE"),
    #                 s3_path_audio=audio_path,
    #                 s3_path_json=json_path,
    #                 ingestion_status=IngestionStatus.INDEXED
    #             )
    #             self.db.add(new_call)
            
    #         self.db.commit()
    #         logger.debug(f"Indexed call {call_id}")

    #     except Exception as e:
    #         self.db.rollback()
    #         logger.error(f"Failed to process {json_path}: {e}")


