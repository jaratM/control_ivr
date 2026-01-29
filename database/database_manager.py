from typing import List, Optional, Union, Tuple
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc, Date, func
from sqlalchemy.dialects.postgresql import insert
from .models import Call, Manifest, ManifestStatus, IngestionStatus, ManifestCall, ManifestCallStatus
from loguru import logger
import math
import pandas as pd

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

def clean_value(v):
    if v is None:
        return None

    if pd.isna(v):
        return None

    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()

    return v

def bulk_insert_manifest_calls(db: Session, manifest_calls_data: List[dict]):
    if not manifest_calls_data:
        return

    columns = {c.name for c in ManifestCall.__table__.columns}
    cleaned_data = []

    for row in manifest_calls_data:
        # cleaned_row = {k: clean_value(v) for k, v in row.items()}
        cleaned_row = {k: clean_value(v) for k, v in row.items() if k in columns}
        for col in columns:
            cleaned_row.setdefault(col, None)
        cleaned_data.append(cleaned_row)

    stmt = insert(ManifestCall).values(cleaned_data)

    update_dict = {
        col.name: stmt.excluded[col.name]
        for col in ManifestCall.__table__.columns
        if col.name != 'numero_commande'
    }

    stmt = stmt.on_conflict_do_update(
        index_elements=['numero_commande'],
        set_=update_dict
    )

    db.execute(stmt)
    db.commit()


def get_manifest_call(db: Session, numero_commande: str) -> Optional[ManifestCall]:
    """Get a manifest call by numero_commande."""
    return db.query(ManifestCall).filter(ManifestCall.numero_commande == numero_commande).first()

def get_manifest_types(db: Session, target_date: Union[datetime, str]) -> List[Tuple[str, str]]:
    """Get a list of unique (category, manifest_filename) tuples for a given date."""
    if isinstance(target_date, str):
        target_date = datetime.strptime(target_date, '%Y-%m-%d')
        
    return (
        db.query(
            ManifestCall.categorie,
            Manifest.filename,
            Manifest.id
        )
        .join(Manifest, ManifestCall.manifest_id == Manifest.id)
        .filter(
            func.date(ManifestCall.date_suspension) == target_date.date(),
            Manifest.status == ManifestStatus.COMPLETED
        )
        .distinct()
        .all()
    )
def get_manifest_calls(db: Session, manifest_id: str, categorie: str) -> List[dict]:
    """Get a list of serialized manifest calls for a given manifest_id and category."""
    
    def serialize_manifest_call(mc: ManifestCall, processed_at) -> dict:
        motif = (mc.motif_suspension or '').lower()
        classification = (mc.classification_modele or '').lower()
        nb_appel = mc.nbr_tentatives_appel or 0
        nb_tonn = mc.nb_tonnalite or 0
        
        # Conformité nb appel logic
        if motif == 'client injoignable':
            conformite_nb_appel = 'Conforme' if nb_appel >= 3 else 'Non conforme'
        else:
            conformite_nb_appel = 'Conforme' if nb_appel >= 1 else 'Non conforme'
        # Conformité declaratif logic
        conformite_declaratif = 'Conforme' if (motif and classification and motif == classification) else 'Non conforme'

        # Convert numeric strings to ensure no float formatting
        def to_str(val):
            if val is None:
                return None
            if isinstance(val, float):
                return str(int(val)) if val == int(val) else str(val)
            # Handle strings that look like floats (e.g., "538988772.0")
            if isinstance(val, str) and '.' in val:
                try:
                    float_val = float(val)
                    if float_val == int(float_val):
                        return str(int(float_val))
                except (ValueError, OverflowError):
                    pass
            return str(val)

        return {
            "numero_commande": mc.numero_commande,
            "manifest_id": mc.manifest_id,
            "MDN": to_str(mc.MDN),
            "numero_ordre": mc.numero_ordre,
            "client_number": to_str(mc.client_number),
            "date_commande": mc.date_commande.date().isoformat() if mc.date_commande else None,
            "date_suspension": mc.date_suspension.date().isoformat() if mc.date_suspension else None,
            "categorie": mc.categorie,
            "motif_suspension": mc.motif_suspension,
            "date_appel_technicien": mc.date_appel_technicien.date().isoformat() if mc.date_appel_technicien else None,
            "nbr_tentatives_appel": mc.nbr_tentatives_appel,
            "conformite_nb_appel": conformite_nb_appel,
            "conformite_intervalle": mc.conformite_intervalle,
            "nb_tonnalite": mc.nb_tonnalite,
            "conformite_nb_tonnalite": mc.conformite_nb_beeps,
            "high_beeps": mc.high_beeps,
            "classification_modele": mc.classification_modele,
            "qualite_communication": mc.qualite_communication if mc.motif_suspension != 'client injoignable' else '',
            "conformite_IAM": mc.conformite_IAM,
            "commentaire": mc.commentaire,
            "processed": 'Traité',
            "ville": mc.ville,
            "nom_prenom": mc.nom_prenom,
            "sous_resultat": "",
            "line_id": to_str(mc.line_id),
            "conformite_nb_beeps": mc.conformite_nb_beeps,
            "processed_at": processed_at.date().isoformat() if processed_at else None,
            "conformite_declaratif": conformite_declaratif,
            "conformite_joignabilite_client": 'Oui' if conformite_nb_appel == 'Conforme' else 'Non',
        }
    
    manifest_calls = (
        db.query(ManifestCall, Manifest.processed_at)
        .join(Manifest, ManifestCall.manifest_id == Manifest.id)
        .filter(ManifestCall.manifest_id == manifest_id, ManifestCall.categorie == categorie)
        .all()
    )
    
    return [serialize_manifest_call(mc, processed_at) for mc, processed_at in manifest_calls]
