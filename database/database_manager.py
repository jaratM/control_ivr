from typing import List, Optional, Union
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc
from sqlalchemy.dialects.postgresql import insert
from .models import Call, Manifest, ManifestStatus, IngestionStatus, ManifestCall, ManifestCallStatus
from loguru import logger

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
    numero_commande: str,
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
            "numero_commande": numero_commande,
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

def get_manifest(db: Session, filename: str) -> Optional[Manifest]:
    """Get a manifest by ID."""
    return db.query(Manifest).filter(Manifest.filename == filename).first()

def bulk_create_calls(db: Session, calls_data: List[dict]):
    """Bulk insert calls."""
    db.bulk_insert_mappings(Call, calls_data)
    db.commit()

def create_manifest_call(db: Session, manifest_call_data: dict) -> ManifestCall:
    """Create a new manifest call record."""
    manifest_call = ManifestCall(**manifest_call_data)
    db.add(manifest_call)
    db.commit()
    db.refresh(manifest_call)
    return manifest_call

def bulk_create_manifest_calls(db: Session, manifest_calls_data: List[dict]):
    """Bulk insert manifest calls."""
    db.bulk_insert_mappings(ManifestCall, manifest_calls_data)
    db.commit()

def bulk_update_manifest_calls(db: Session, manifest_calls_data: List[dict]):
    """Bulk update manifest calls."""
    db.bulk_update_mappings(ManifestCall, manifest_calls_data)
    db.commit()
    
def bulk_insert_manifest_calls(db: Session, manifest_calls_data: List[dict]):
    """Bulk insert manifest calls."""
    
    if not manifest_calls_data:
        return

    try:
        stmt = insert(ManifestCall).values(manifest_calls_data)
        
        # Update all columns except primary key and foreign key on conflict
        update_dict = {
            col.name: stmt.excluded[col.name] 
            for col in ManifestCall.__table__.columns 
            if col.name not in ['numero_commande', 'manifest_id']  # Keep FK stable
        }
        
        stmt = stmt.on_conflict_do_update(
            index_elements=['numero_commande'],
            set_=update_dict
        )
        
        result = db.execute(stmt)
        db.commit()
        
        return
        
    except Exception as e:
        db.rollback()
        raise

def get_manifest_call(db: Session, numero_commande: str) -> Optional[ManifestCall]:
    """Get a manifest call by numero_commande."""
    return db.query(ManifestCall).filter(ManifestCall.numero_commande == numero_commande).first()
