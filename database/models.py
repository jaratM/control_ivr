from sqlalchemy import Column, String, DateTime, Integer, Boolean, ForeignKey, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime
import enum
import uuid

Base = declarative_base()

class IngestionStatus(enum.Enum):
    PENDING = "PENDING"
    INDEXED = "INDEXED"
    ERROR = "ERROR"

class ManifestStatus(enum.Enum):
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class ManifestCallStatus(enum.Enum):
    CLIENT_INJOINABLE = "CLIENT_INJOINABLE"
    CLIENT_REPORTE_RDV = "CLIENT_REPORTE_RDV"
    CLIENT_REFUSE_INSTALLATION = "CLIENT_REFUSE_INSTALLATION"
    CLIENT_ABSCENT = "CLIENT_ABSCENT"
    ABSENCE_ROUTEUR_CLIENT = "ABSENCE_ROUTEUR_CLIENT"
    LOCAL_FERME = "LOCAL_FERME"
    
class Call(Base):
    __tablename__ = 'calls'

    call_id = Column(String, primary_key=True) # Using String as UUIDs in JSON might be strings
    client_number = Column(String, index=True, nullable=False)
    technician_number = Column(String, nullable=True)
    start_time = Column(DateTime, index=True, nullable=False)
    end_time = Column(DateTime, nullable=True)
    branch = Column(String, nullable=True)
    
    # Storage paths
    s3_path_audio = Column(String, nullable=True)
    s3_path_json = Column(String, nullable=True)
    
    ingestion_status = Column(SQLEnum(IngestionStatus), default=IngestionStatus.PENDING)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships

class Manifest(Base):
    __tablename__ = 'manifests'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    status = Column(SQLEnum(ManifestStatus), default=ManifestStatus.PROCESSING)
    received_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)

    calls = relationship("ManifestCall", back_populates="manifest", cascade="all, delete-orphan")

class ManifestCall(Base):
    __tablename__ = 'manifest_calls'
    numero_commande = Column(String, primary_key=True)
    manifest_id = Column(String, ForeignKey('manifests.id', ondelete='CASCADE'), nullable=False)
    # Relationship back to Manifest
    manifest = relationship("Manifest", back_populates="calls")
    MDN = Column(String, nullable=True)
    client_number = Column(String, nullable=True)
    date_commande = Column(DateTime, nullable=True)
    date_suspension = Column(DateTime, nullable=True)
    categorie = Column(String, nullable=True)
    nbr_tentatives_appel = Column(Integer, nullable=True)
    conformite_intervalle = Column(String, nullable=True)
    appels_branch = Column(String, nullable=True)
    nb_tonnalite = Column(Integer, nullable=True)
    beep_count = Column(Integer, nullable=True)
    high_beeps = Column(Integer, nullable=True)
    status = Column(String, nullable=True)
    classification_modele = Column(String, nullable=True)
    qualite_communication = Column(String, nullable=True)
    conformite_IAM = Column(String, nullable=True)
    commentaire = Column(String, nullable=True)
    motif_suspension = Column(String, nullable=True)
    processed = Column(Boolean, nullable=True)
    compliance = Column(String, nullable=True)




