"""
Pytest configuration and shared fixtures.
"""
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock
import pytest
import numpy as np
import torch
import yaml
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.models import Base, Manifest, ManifestCall, ManifestStatus
from modules.types import AudioSegment, TranscriptionResult, ClassificationResult


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'pipeline': {
            'num_ingestion_workers': 2,
            'num_classification_workers': 1,
            'batch_size': 4,
            'batch_timeout_ms': 100,
            'max_queue_size': 100
        },
        'processing': {
            'target_sample_rate': 16000,
            'chunk_duration_sec': 25,
            'overlap_sec': 0.5,
            'transcription_model': 'test-model'
        },
        'compliance_rules': {
            'max_beeps': 5
        },
        'storage': {
            'endpoint': 'localhost:9000',
            'access_key': 'test_key',
            'secret_key': 'test_secret',
            'bucket_name': 'test-bucket',
            'secure': False
        },
        'database': {
            'database_type': 'postgresql',
            'db_host': 'localhost',
            'db_port': 5432,
            'db_name': 'test_db',
            'db_user': 'test_user',
            'db_password': 'test_pass'
        },
        'classification': {
            'acquisition': 'config/acquisition.txt',
            'sav': 'config/sav.txt'
        },
        'logging': {
            'level': 'ERROR'
        }
    }


@pytest.fixture
def config_file(temp_dir, sample_config):
    """Create a temporary config YAML file."""
    config_path = os.path.join(temp_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(sample_config, f)
    return config_path


@pytest.fixture
def sample_audio_data():
    """Generate sample audio data for testing."""
    sample_rate = 16000
    duration = 2.0  # seconds
    num_samples = int(sample_rate * duration)
    
    # Generate a simple sine wave
    frequency = 440  # Hz (A note)
    t = np.linspace(0, duration, num_samples)
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    return audio, sample_rate


@pytest.fixture
def sample_audio_segment(sample_audio_data):
    """Create a sample AudioSegment for testing."""
    audio, sample_rate = sample_audio_data
    return AudioSegment(
        file_id="test_file_001",
        segment_index=0,
        audio_data=audio,
        duration=2.0,
        # start_time=0.0,
        # end_time=2.0
    )


@pytest.fixture
def sample_audio_file(temp_dir, sample_audio_data):
    """Create a sample audio WAV file for testing."""
    import soundfile as sf
    
    audio, sample_rate = sample_audio_data
    file_path = os.path.join(temp_dir, 'test_audio.wav')
    sf.write(file_path, audio, sample_rate)
    return file_path


@pytest.fixture
def sample_transcription_result():
    """Create a sample TranscriptionResult."""
    return TranscriptionResult(
        file_id="test_file_001",
        segment_index=0,
        text="السلام عليكم، بغيت نلغي الطلب ديالي"
    )


@pytest.fixture
def sample_classification_result():
    """Create a sample ClassificationResult."""
    return ClassificationResult(
        file_id="test_file_001",
        call_type=1,  # Client refuse installation
        technician_behavior=1,  # Good
        category_label="Client refuse installation",
        behavior_label="Bien"
    )


@pytest.fixture
def mock_minio_client():
    """Mock MinIO client for testing."""
    mock = MagicMock()
    mock.bucket_exists.return_value = True
    mock.list_objects.return_value = []
    mock.download_file.return_value = None
    mock.upload_file.return_value = None
    return mock


@pytest.fixture
def mock_llm_api(requests_mock):
    """Mock LLM API responses."""
    def create_mock_response(call_type=1, tech_behavior=1):
        return {
            'completion': f'{call_type}{tech_behavior}',
            'stop_reason': 'end_turn'
        }
    
    return create_mock_response


@pytest.fixture
def in_memory_db():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    yield session
    
    session.close()
    engine.dispose()


@pytest.fixture
def sample_manifest(in_memory_db):
    """Create a sample manifest in the database."""
    manifest = Manifest(
        id="test_manifest_001",
        filename="test_manifest.csv",
        status=ManifestStatus.PROCESSING,
        received_at=datetime.now(),
        processed_at=None
    )
    in_memory_db.add(manifest)
    in_memory_db.commit()
    return manifest


@pytest.fixture
def sample_manifest_call(in_memory_db, sample_manifest):
    """Create a sample manifest call in the database."""
    call = ManifestCall(
        numero_commande="CMD001",
        manifest_id=sample_manifest.id,
        client_number="0612345678",
        date_commande=datetime.now() - timedelta(days=5),
        date_suspension=datetime.now() - timedelta(days=1),
        motif_suspension="client injoignable",
        nbr_tentatives_appel=2,
        conformite_IAM="Non conforme",
        conformite_intervalle="",
        appels_branch="",
        commentaire="Moins de 3 tentatives",
        processed=False
    )
    in_memory_db.add(call)
    in_memory_db.commit()
    return call


@pytest.fixture
def sample_calls_metadata():
    """Sample call metadata list."""
    return [
        {
            'file_id': 'call_001',
            'path': '/audio/call_001.wav',
            'start_time': datetime.now() - timedelta(hours=5),
            'numero_commande': 'CMD001'
        },
        {
            'file_id': 'call_002',
            'path': '/audio/call_002.wav',
            'start_time': datetime.now() - timedelta(hours=3),
            'numero_commande': 'CMD001'
        },
        {
            'file_id': 'call_003',
            'path': '/audio/call_003.wav',
            'start_time': datetime.now() - timedelta(hours=1),
            'numero_commande': 'CMD001'
        }
    ]


@pytest.fixture
def mock_transcriber():
    """Mock Transcriber for testing."""
    mock = MagicMock()
    mock.device = 'cpu'
    mock.transcribe_batch.return_value = [
        TranscriptionResult(
            file_id="test_001",
            segment_index=0,
            text="السلام عليكم"
        )
    ]
    return mock


@pytest.fixture
def mock_classifier():
    """Mock Classifier for testing."""
    mock = MagicMock()
    mock.classify_full_text.return_value = {
        'file_id': 'test_001',
        'call_type': 1,
        'technician_behavior': 1,
        'category_label': 'Client refuse installation',
        'behavior_label': 'Bien'
    }
    return mock


@pytest.fixture(autouse=True)
def disable_gpu():
    """Disable GPU for all tests."""
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    yield
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        del os.environ['CUDA_VISIBLE_DEVICES']
