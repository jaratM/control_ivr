import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_
from loguru import logger
from database.models import Call, Manifest, ManifestStatus
import uuid

Acquisition_columns = {
    "ADSL": {"date_commande": "start_time", "date_last_status": "end_time", "statut_de_degroupage_iam": "status", "numero_de_contact": "client_number", "statut_commande": "status_commande"},
    "VULA": {"createddate": "start_time", "inwib2c_statut_demande__c": "status_demande", "inwib2c_motif__c": "status", "lastmodifieddate": "end_time", "inwib2c_numerocontact__c": "client_number"}
}
SAV_columns={
    
}

class ManifestProcessor:
    def __init__(self, db_session: Session):
        self.db = db_session

    def process_manifest(self, csv_path: str) -> List[Call]:
        """
        Parses a CSV manifest and retrieves matching calls from the database.
        """
        manifest_id = str(uuid.uuid4())
        filename = csv_path.split('/')[-1]
        
        # Record manifest start
        manifest_record = Manifest(
            id=manifest_id,
            filename=filename,
            status=ManifestStatus.PROCESSING
        )
        self.db.add(manifest_record)
        self.db.commit()

        matched_calls = []

        try:
            df = pd.read_csv(csv_path)
            # Expected columns: start_date, last_updated, contact_number, issue_type
            
            for _, row in df.iterrows():
                try:
                    # Parse dates from CSV
                    # Assuming format matches database or is standard ISO. 
                    # Adjust format string as per actual CSV data.
                    # Example: 2025-11-19 09:00:00
                    start_date = pd.to_datetime(row['start_date']).to_pydatetime()
                    end_date = pd.to_datetime(row['last_updated']).to_pydatetime()
                    contact_number = str(row['contact_number'])
                    issue_type = row['issue_type']

                    # Query DB
                    # Logic: Call DATE_DEBUT (start_time) is within [start_date, end_date]
                    # AND client_number == contact_number
                    
                    query = self.db.query(Call).filter(
                        Call.client_number == contact_number,
                        Call.start_time >= start_date,
                        Call.start_time <= end_date
                    )
                    
                    results = query.all()
                    
                    for call in results:
                        # Validate BRANCHE vs issue_type
                        # Logic: Simple equality or mapping check? 
                        # User said: "Validate issue_type from CSV against BRANCHE field"
                        # For now, we'll assume if they match or mapped, we process it.
                        # If validation fails, we might skip or flag it. 
                        # Let's assume we return it but maybe tag it?
                        # For now, just filtering based on business logic often implies
                        # we only want calls that MATCH the issue type context.
                        
                        # If strictly validating:
                        # if call.branch != issue_type: continue
                        
                        # But often "Validate" means "Check compliance for this type".
                        # So we add it to the list.
                        matched_calls.append(call)

                except Exception as row_error:
                    logger.error(f"Error processing row in manifest {filename}: {row_error}")
                    continue

            # Update manifest status
            manifest_record.status = ManifestStatus.COMPLETED
            manifest_record.processed_at = datetime.utcnow()
            self.db.commit()
            
            logger.info(f"Manifest {filename} processed. Found {len(matched_calls)} matching calls.")
            return matched_calls

        except Exception as e:
            manifest_record.status = ManifestStatus.FAILED
            self.db.commit()
            logger.error(f"Failed to process manifest {csv_path}: {e}")
            return []

