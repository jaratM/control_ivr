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
    verification_results = relationship("VerificationResult", back_populates="call", cascade="all, delete-orphan")

class Manifest(Base):
    __tablename__ = 'manifests'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    status = Column(SQLEnum(ManifestStatus), default=ManifestStatus.PROCESSING)
    received_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)

class VerificationResult(Base):
    __tablename__ = 'verification_results'

    id = Column(Integer, primary_key=True, autoincrement=True)
    call_id = Column(String, ForeignKey('calls.call_id'), nullable=False)
    is_compliant = Column(Boolean, nullable=False)
    issues = Column(JSONB, default=list)
    details = Column(JSONB, default=dict)
    processed_at = Column(DateTime, default=datetime.utcnow)

    call = relationship("Call", back_populates="verification_results")

