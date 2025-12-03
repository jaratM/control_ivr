from typing import List, Optional, Union
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc
from .models import Call, Manifest, ManifestStatus, VerificationResult, IngestionStatus, ManifestCall, ManifestCallStatus

def create_call(db: Session, call_data: dict) -> Call:
    """Create a new call record."""
    call = Call(**call_data)
    db.add(call)
    db.commit()
    db.refresh(call)
    return call

def get_call(db: Session, call_id: str) -> Optional[Call]:
    """Retrieve a call by ID."""
    return db.query(Call).filter(Call.call_id == call_id).first()

def get_calls(
    db: Session, 
    client_number: str, 
    start_date: datetime, 
    end_date: datetime, 
    strategy: str = 'all'
) -> Union[List[Call], Optional[Call]]:
    """
    Retrieve calls based on client number and date range.
    
    Args:
        db: Database session
        client_number: Client phone number
        start_date: Start of the search window
        end_date: End of the search window
        strategy: 'all' returns list of calls, 'first' returns the earliest call, 'last' returns latest.
    """
    query = db.query(Call).filter(
        Call.client_number == client_number,
        Call.start_time >= start_date,
        Call.start_time <= end_date
    )
    
    def serialize_call(call):
        if call is None:
            return None
        return {
            "call_id": call.call_id,
            "client_number": call.client_number,
            "start_time": call.start_time.isoformat() if call.start_time else None,
            "s3_path_audio": call.s3_path_audio,
            "branch": call.branch,
        }

    if strategy == 'first':
        db_obj = query.order_by(asc(Call.start_time)).first()
        return serialize_call(db_obj)
    elif strategy == 'last':
        db_obj = query.order_by(desc(Call.start_time)).first()
        return serialize_call(db_obj)
    else:
        db_objs = query.order_by(desc(Call.start_time)).all()
        return [serialize_call(obj) for obj in db_objs]

def update_call_ingestion_status(db: Session, call_id: str, status: IngestionStatus):
    """Update the ingestion status of a call."""
    call = get_call(db, call_id)
    if call:
        call.ingestion_status = status
        db.commit()

def create_manifest(db: Session, filename: str) -> Manifest:
    """Create a new manifest record."""
    manifest = Manifest(filename=filename)
    db.add(manifest)
    db.commit()
    db.refresh(manifest)
    return manifest

def update_manifest_status(db: Session, manifest_id: str, status: ManifestStatus, processed_at: Optional[datetime] = None):
    """Update manifest status."""
    manifest = db.query(Manifest).filter(Manifest.id == manifest_id).first()
    if manifest:
        manifest.status = status
        if processed_at:
            manifest.processed_at = processed_at
        db.commit()

def create_verification_result(db: Session, result_data: dict) -> VerificationResult:
    """Create a verification result."""
    result = VerificationResult(**result_data)
    db.add(result)
    db.commit()
    db.refresh(result)
    return result

def get_verification_results(db: Session, call_id: str) -> List[VerificationResult]:
    """Get verification results for a call."""
    return db.query(VerificationResult).filter(VerificationResult.call_id == call_id).all()

def bulk_create_calls(db: Session, calls_data: List[dict]):
    """Bulk insert calls."""
    db.bulk_insert_mappings(Call, calls_data)
    db.commit()

