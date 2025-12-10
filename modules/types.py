from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np

@dataclass
class AudioMetadata:
    file_path: str
    file_id: str
    duration: float
    start_time: str

@dataclass
class AudioSegment:
    file_id: str
    segment_index: int
    audio_data: np.ndarray  # Simulation of raw audio segment
    duration: float

@dataclass
class TranscriptionResult:
    file_id: str
    segment_index: int
    text: str

@dataclass
class ClassificationInput:
    numero_commande: str
    file_id: str
    full_transcript: str
    metadata: AudioMetadata
    beep_count: int
    high_beeps: int

@dataclass
class ClassificationResult:
    file_id: str
    status: str # e.g., "Silence", "Le client refuse l'installation"..
    behavior: str # e.g., "Bien" or "Mauvais"

@dataclass
class ComplianceInput:
    numero_commande: str
    metadata: AudioMetadata
    beep_count: int
    classification: ClassificationResult
    high_beeps: int

@dataclass
class ComplianceResult:
    file_id: str
    is_compliant: bool
    issues: List[str]
    details: Dict[str, Any]
