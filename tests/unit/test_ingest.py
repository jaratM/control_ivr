"""
Unit tests for the Ingestion Service.
"""
import os
import json
import pytest
from unittest.mock import Mock, MagicMock, patch, mock_open
from datetime import datetime
from services.ingest import IngestionService
from database.models import Call, IngestionStatus


class TestIngestionService:
    """Test IngestionService class."""

    @pytest.fixture
    def mock_db_session(self):
        """Create a mock database session."""
        session = Mock()
        session.query.return_value.filter.return_value.first.return_value = None
        session.add = Mock()
        session.commit = Mock()
        session.rollback = Mock()
        return session

    @pytest.fixture
    def mock_minio_client(self):
        """Create a mock MinIO storage client."""
        client = Mock()
        return client

    @pytest.fixture
    def ingestion_service(self, mock_db_session, mock_minio_client):
        """Create an IngestionService instance."""
        return IngestionService(mock_db_session, mock_minio_client)

    @pytest.fixture
    def valid_json_data(self):
        """Create valid JSON data for testing."""
        return {
            "CALL_ID": "test_call_001",
            "NUMERO_CLIENT": "0612345678",
            "NUMERO_TECHNICIEN": "0698765432",
            "DATE_DEBUT": "2025-11-19 09:33:13",
            "DATE_FIN": "2025-11-19 09:45:22",
            "BRANCHE": "ADSL"
        }

    def test_init(self, mock_db_session, mock_minio_client):
        """Test IngestionService initialization."""
        service = IngestionService(mock_db_session, mock_minio_client)
        assert service.db == mock_db_session
        assert service.minio == mock_minio_client

    @patch('os.path.isfile')
    @patch('os.listdir')
    @patch('services.ingest.IngestionService.process_json_file')
    def test_ingest_folder_processes_json_files(self, mock_process, mock_listdir, mock_isfile, ingestion_service):
        """Test that ingest_folder processes JSON files."""
        mock_listdir.return_value = ['file1.json', 'file2.json', 'file3.txt', 'file4.json']
        mock_isfile.return_value = True
        
        ingestion_service.ingest_folder('/test/folder')
        
        assert mock_process.call_count == 3
        mock_process.assert_any_call('/test/folder/file1.json')
        mock_process.assert_any_call('/test/folder/file2.json')
        mock_process.assert_any_call('/test/folder/file4.json')

    @patch('os.listdir')
    def test_ingest_folder_empty_directory(self, mock_listdir, ingestion_service):
        """Test ingest_folder with empty directory."""
        mock_listdir.return_value = []
        
        # Should not raise any errors
        ingestion_service.ingest_folder('/empty/folder')

    @patch('builtins.open', new_callable=mock_open)
    def test_process_json_file_creates_new_call(self, mock_file, ingestion_service, valid_json_data, mock_db_session):
        """Test processing a JSON file creates a new call."""
        mock_file.return_value.read.return_value = json.dumps(valid_json_data)
        
        ingestion_service.process_json_file('/test/path/test.json')
        
        # Verify database interactions
        assert mock_db_session.add.called
        assert mock_db_session.commit.called
        
        # Verify the call was created with correct attributes
        added_call = mock_db_session.add.call_args[0][0]
        assert isinstance(added_call, Call)
        assert added_call.call_id == "test_call_001"
        assert added_call.client_number == "0612345678"
        assert added_call.technician_number == "0698765432"
        assert added_call.branch == "ADSL"
        assert added_call.ingestion_status == IngestionStatus.INDEXED

    @patch('builtins.open', new_callable=mock_open)
    def test_process_json_file_updates_existing_call(self, mock_file, ingestion_service, valid_json_data, mock_db_session):
        """Test processing a JSON file updates existing call."""
        mock_file.return_value.read.return_value = json.dumps(valid_json_data)
        
        # Mock existing call
        existing_call = Mock()
        mock_db_session.query.return_value.filter.return_value.first.return_value = existing_call
        
        ingestion_service.process_json_file('/test/path/test.json')
        
        # Verify update operations
        assert existing_call.ingestion_status == IngestionStatus.INDEXED
        assert hasattr(existing_call, 'updated_at')
        assert mock_db_session.commit.called
        assert not mock_db_session.add.called

    @patch('builtins.open', new_callable=mock_open)
    def test_process_json_file_empty_json(self, mock_file, ingestion_service, mock_db_session):
        """Test processing empty JSON file."""
        mock_file.return_value.read.return_value = json.dumps({})
        
        ingestion_service.process_json_file('/test/path/empty.json')
        
        # Should not add or commit
        assert not mock_db_session.add.called
        assert not mock_db_session.commit.called

    @patch('builtins.open', new_callable=mock_open)
    def test_process_json_file_missing_call_id(self, mock_file, ingestion_service, mock_db_session):
        """Test processing JSON without CALL_ID."""
        invalid_data = {
            "NUMERO_CLIENT": "0612345678",
            "DATE_DEBUT": "2025-11-19 09:33:13",
            "DATE_FIN": "2025-11-19 09:45:22"
        }
        mock_file.return_value.read.return_value = json.dumps(invalid_data)
        
        ingestion_service.process_json_file('/test/path/invalid.json')
        
        # Should not add or commit
        assert not mock_db_session.add.called
        assert not mock_db_session.commit.called

    @patch('builtins.open', new_callable=mock_open)
    def test_process_json_file_invalid_date_format(self, mock_file, ingestion_service, mock_db_session):
        """Test processing JSON with invalid date format."""
        invalid_data = {
            "CALL_ID": "test_call_002",
            "NUMERO_CLIENT": "0612345678",
            "DATE_DEBUT": "invalid-date",
            "DATE_FIN": "2025-11-19 09:45:22"
        }
        mock_file.return_value.read.return_value = json.dumps(invalid_data)
        
        ingestion_service.process_json_file('/test/path/invalid_date.json')
        
        # Should not add or commit due to date parsing error
        assert not mock_db_session.add.called
        assert not mock_db_session.commit.called

    @patch('builtins.open', new_callable=mock_open)
    def test_process_json_file_constructs_audio_path(self, mock_file, ingestion_service, valid_json_data, mock_db_session):
        """Test that audio path is correctly constructed from JSON path."""
        mock_file.return_value.read.return_value = json.dumps(valid_json_data)
        
        ingestion_service.process_json_file('/test/path/test.json')
        
        added_call = mock_db_session.add.call_args[0][0]
        assert added_call.s3_path_audio == '/test/path/test.ogg'
        assert added_call.s3_path_json == '/test/path/test.json'

    @patch('builtins.open', side_effect=FileNotFoundError)
    def test_process_json_file_not_found(self, mock_file, ingestion_service, mock_db_session):
        """Test processing non-existent JSON file."""
        ingestion_service.process_json_file('/test/path/nonexistent.json')
        
        # Should rollback on error
        assert mock_db_session.rollback.called
        assert not mock_db_session.commit.called

    @patch('builtins.open', new_callable=mock_open)
    def test_process_json_file_invalid_json(self, mock_file, ingestion_service, mock_db_session):
        """Test processing invalid JSON content."""
        mock_file.return_value.read.return_value = "not valid json {"
        
        ingestion_service.process_json_file('/test/path/invalid.json')
        
        # Should rollback on JSON decode error
        assert mock_db_session.rollback.called
        assert not mock_db_session.commit.called

    @patch('builtins.open', new_callable=mock_open)
    def test_process_json_file_database_error(self, mock_file, ingestion_service, valid_json_data, mock_db_session):
        """Test handling database errors during processing."""
        mock_file.return_value.read.return_value = json.dumps(valid_json_data)
        mock_db_session.commit.side_effect = Exception("Database error")
        
        ingestion_service.process_json_file('/test/path/test.json')
        
        # Should rollback on commit error
        assert mock_db_session.rollback.called

    @patch('builtins.open', new_callable=mock_open)
    def test_process_json_file_parses_dates_correctly(self, mock_file, ingestion_service, valid_json_data, mock_db_session):
        """Test that dates are parsed correctly."""
        mock_file.return_value.read.return_value = json.dumps(valid_json_data)
        
        ingestion_service.process_json_file('/test/path/test.json')
        
        added_call = mock_db_session.add.call_args[0][0]
        assert isinstance(added_call.start_time, datetime)
        assert isinstance(added_call.end_time, datetime)
        assert added_call.start_time.year == 2025
        assert added_call.start_time.month == 11
        assert added_call.start_time.day == 19

    @patch('builtins.open', new_callable=mock_open)
    def test_process_json_file_handles_optional_fields(self, mock_file, ingestion_service, mock_db_session):
        """Test processing JSON with missing optional fields."""
        minimal_data = {
            "CALL_ID": "test_call_003",
            "DATE_DEBUT": "2025-11-19 09:33:13",
            "DATE_FIN": "2025-11-19 09:45:22"
        }
        mock_file.return_value.read.return_value = json.dumps(minimal_data)
        
        ingestion_service.process_json_file('/test/path/minimal.json')
        
        added_call = mock_db_session.add.call_args[0][0]
        assert added_call.call_id == "test_call_003"
        assert added_call.client_number is None
        assert added_call.technician_number is None
        assert added_call.branch is None
