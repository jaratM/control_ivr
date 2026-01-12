"""
Unit tests for the Manifest Processing Service.
"""
import pytest
import pandas as pd
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
from services.manifest import (
    ManifestProcessor, 
    extract_contact_phone,
    normalize_phone_number,
    ACQUISITION_STATUSES,
    SAV_STATUSES
)
from database.models import Manifest, ManifestStatus


class TestExtractContactPhone:
    """Test extract_contact_phone function."""

    def test_extract_phone_with_contact_keyword_international(self):
        """Test extracting international format phone after 'contact' keyword."""
        comment = "Rappel client contact 212612345678 apres 15h"
        result = extract_contact_phone(comment)
        assert result == "212612345678"

    def test_extract_phone_with_contact_keyword_local(self):
        """Test extracting local format phone after 'contact' keyword."""
        comment = "Rappel client contact 0612345678 apres 15h"
        result = extract_contact_phone(comment)
        assert result == "212612345678"

    def test_extract_phone_without_contact_keyword(self):
        """Test extracting first mobile number without 'contact' keyword."""
        comment = "Client disponible 0712345678 toute la journee"
        result = extract_contact_phone(comment)
        assert result == "212712345678"

    def test_extract_phone_contact_case_insensitive(self):
        """Test case insensitive 'contact' matching."""
        comment = "CONTACT 0612345678"
        result = extract_contact_phone(comment)
        assert result == "212612345678"

    def test_extract_phone_multiple_numbers_uses_first_after_contact(self):
        """Test that first number after 'contact' is selected when multiple exist."""
        comment = "Tel 0612345678 mais contact 0798765432 de preference"
        result = extract_contact_phone(comment)
        assert result == "212798765432"

    def test_extract_phone_no_mobile_numbers(self):
        """Test with no valid mobile numbers."""
        comment = "Pas de numero mobile valide 05123456"
        result = extract_contact_phone(comment)
        assert result is None

    def test_extract_phone_empty_string(self):
        """Test with empty string."""
        result = extract_contact_phone("")
        assert result is None

    def test_extract_phone_none_input(self):
        """Test with None input."""
        result = extract_contact_phone(None)
        assert result is None

    def test_extract_phone_non_string_input(self):
        """Test with non-string input."""
        result = extract_contact_phone(12345)
        assert result is None


class TestNormalizePhoneNumber:
    """Test normalize_phone_number function."""

    def test_normalize_local_format(self):
        """Test normalizing local format (06/07 prefix)."""
        assert normalize_phone_number("0612345678") == "212612345678"
        assert normalize_phone_number("0712345678") == "212712345678"

    def test_normalize_international_format_unchanged(self):
        """Test that international format remains unchanged."""
        assert normalize_phone_number("212612345678") == "212612345678"

    def test_normalize_none_input(self):
        """Test with None input."""
        assert normalize_phone_number(None) is None

    def test_normalize_empty_string(self):
        """Test with empty string."""
        assert normalize_phone_number("") is None


class TestManifestProcessor:
    """Test ManifestProcessor class."""

    @pytest.fixture
    def mock_db_session(self):
        """Create a mock database session."""
        session = Mock()
        return session

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        return {
            'csv_mappings': {
                'acquisition': {
                    'ADSL': {
                        'NUMERO_COMMANDE': 'numero_commande',
                        'NUMERO_CLIENT': 'client_number',
                        'DATE_COMMANDE': 'date_commande',
                        'DATE_SUSPENSION': 'date_suspension',
                        'STATUS': 'status_commande',
                        'MOTIF': 'motif_suspension'
                    },
                    'VULA': {
                        'NUMERO_COMMANDE': 'numero_commande',
                        'NUMERO_CLIENT': 'client_number',
                        'DATE_COMMANDE': 'date_commande',
                        'DATE_SUSPENSION': 'date_suspension',
                        'STATUS': 'status_commande',
                        'MOTIF': 'motif_suspension'
                    }
                },
                'SAV': {
                    'NUMERO_COMMANDE': 'numero_commande',
                    'COMMENTAIRE': 'client_number',
                    'DATE_COMMANDE': 'date_commande',
                    'DATE_SUSPENSION': 'date_suspension',
                    'MOTIF': 'motif_suspension'
                }
            }
        }

    @pytest.fixture
    def manifest_processor(self, mock_db_session, mock_config):
        """Create a ManifestProcessor instance."""
        return ManifestProcessor(mock_db_session, mock_config)

    @pytest.fixture
    def mock_manifest(self):
        """Create a mock Manifest record."""
        manifest = Mock()
        manifest.id = "test-manifest-id"
        manifest.filename = "test_file.csv"
        return manifest

    def test_init(self, mock_db_session, mock_config):
        """Test ManifestProcessor initialization."""
        processor = ManifestProcessor(mock_db_session, mock_config)
        assert processor.db == mock_db_session
        assert processor.config == mock_config

    @patch('services.manifest.ManifestProcessor._read_df')
    @patch('services.manifest.get_calls')
    def test_process_manifest_identifies_adsl_type(self, mock_get_calls, mock_read_df, manifest_processor, mock_manifest):
        """Test that ADSL manifest type is correctly identified."""
        # Create sample data
        df_data = pd.DataFrame({
            'numero_commande': ['CMD001'],
            'client_number': ['0612345678'],
            'date_commande': [datetime(2025, 1, 1)],
            'date_suspension': [datetime(2025, 1, 10)],
            'status_commande': ['Suspendue'],
            'motif_suspension': ['client refuse installation']
        })
        mock_read_df.return_value = df_data
        mock_get_calls.return_value = []

        result = manifest_processor.process_manifest(
            '/path/to/crc_adsl_20250101.csv',
            datetime(2025, 1, 10),
            mock_manifest
        )
        
        df_dict, calls_metadata, manifest_type, category = result
        assert manifest_type == 'ACQUISITION'
        assert category == 'ADSL'

    @patch('services.manifest.ManifestProcessor._read_df')
    @patch('services.manifest.get_calls')
    def test_process_manifest_identifies_vula_type(self, mock_get_calls, mock_read_df, manifest_processor, mock_manifest):
        """Test that VULA manifest type is correctly identified."""
        df_data = pd.DataFrame({
            'numero_commande': ['CMD001'],
            'client_number': ['0612345678'],
            'date_commande': [datetime(2025, 1, 1)],
            'date_suspension': [datetime(2025, 1, 10)],
            'status_commande': ['Suspendue'],
            'motif_suspension': ['client refuse installation']
        })
        mock_read_df.return_value = df_data
        mock_get_calls.return_value = []

        result = manifest_processor.process_manifest(
            '/path/to/crc_vula_20250101.csv',
            datetime(2025, 1, 10),
            mock_manifest
        )
        
        df_dict, calls_metadata, manifest_type, category = result
        assert manifest_type == 'ACQUISITION'
        assert category == 'VULA'

    @patch('services.manifest.ManifestProcessor._read_df')
    @patch('services.manifest.get_calls')
    def test_process_manifest_identifies_sav_type(self, mock_get_calls, mock_read_df, manifest_processor, mock_manifest):
        """Test that SAV manifest type is correctly identified."""
        # Create dataframe with the comment column BEFORE renaming
        df_data = pd.DataFrame({
            'NUMERO_COMMANDE': ['CMD001'],
            'COMMENTAIRE': ['Contact 0612345678'],  # Original column name
            'DATE_COMMANDE': ['2025-01-01'],
            'DATE_SUSPENSION': ['2025-01-10'],
            'MOTIF': ['client absent'],
            'DATE_RECYCLAGE': [pd.NaT]
        })
        mock_read_df.return_value = df_data
        mock_get_calls.return_value = []

        result = manifest_processor.process_manifest(
            '/path/to/sav_20250101.csv',
            datetime(2025, 1, 10),
            mock_manifest
        )
        
        df_dict, calls_metadata, manifest_type, category = result
        assert manifest_type == 'SAV'

    @patch('services.manifest.ManifestProcessor._read_df')
    @patch('services.manifest.update_manifest_status')
    def test_process_manifest_handles_unsupported_file_type(self, mock_update_status, mock_read_df, manifest_processor, mock_manifest):
        """Test handling of unsupported file types."""
        result = manifest_processor.process_manifest(
            '/path/to/unknown_type.csv',
            datetime(2025, 1, 10),
            mock_manifest
        )
        
        df_dict, calls_metadata, manifest_type, category = result
        assert df_dict == []
        assert calls_metadata == []
        assert manifest_type is None
        mock_update_status.assert_called_once()

    def test_filter_acquisition_status(self, manifest_processor):
        """Test filtering acquisition DataFrame by status."""
        df = pd.DataFrame({
            'status_commande': ['Suspendue', 'Active', 'Suspendue', 'Suspendue'],
            'motif_suspension': ['client refuse installation', 'N/A', 'client injoignable', 'autre motif'],
            'date_suspension': [
                datetime(2025, 1, 10),
                datetime(2025, 1, 10),
                datetime(2025, 1, 10),
                datetime(2025, 1, 10)
            ]
        })
        
        result = manifest_processor._filter_acquisition_status(df, None)
        assert len(result) == 2
        assert all(result.status_commande == 'Suspendue')

    def test_filter_acquisition_status_with_date(self, manifest_processor):
        """Test filtering acquisition by status and date."""
        df = pd.DataFrame({
            'status_commande': ['Suspendue', 'Suspendue'],
            'motif_suspension': ['client refuse installation', 'client refuse installation'],
            'date_suspension': [
                datetime(2025, 1, 10),
                datetime(2025, 1, 15)
            ]
        })
        
        result = manifest_processor._filter_acquisition_status(df, datetime(2025, 1, 10))
        assert len(result) == 1
        assert result.iloc[0].date_suspension.date() == datetime(2025, 1, 10).date()

    def test_filter_sav_status(self, manifest_processor):
        """Test filtering SAV DataFrame by status."""
        df = pd.DataFrame({
            'motif_suspension': ['client absent', 'autre motif', 'local ferme', 'motif invalide'],
            'date_suspension': [
                datetime(2025, 1, 10),
                datetime(2025, 1, 10),
                datetime(2025, 1, 10),
                datetime(2025, 1, 10)
            ]
        })
        
        result = manifest_processor._filter_sav_status(df, None)
        assert len(result) == 2

    def test_convert_date_columns(self, manifest_processor):
        """Test converting date columns to datetime."""
        df = pd.DataFrame({
            'date_commande': ['2025-01-01', '2025-01-02'],
            'date_suspension': ['2025-01-10', '2025-01-15'],
            'other_column': ['value1', 'value2']
        })
        
        result = manifest_processor._convert_date_columns(df, ['date_commande', 'date_suspension'])
        
        assert pd.api.types.is_datetime64_any_dtype(result['date_commande'])
        assert pd.api.types.is_datetime64_any_dtype(result['date_suspension'])
        assert not pd.api.types.is_datetime64_any_dtype(result['other_column'])

    def test_convert_date_columns_handles_invalid_dates(self, manifest_processor):
        """Test that invalid dates are converted to NaT."""
        df = pd.DataFrame({
            'date_commande': ['2025-01-01', 'invalid-date', '2025-01-03'],
        })
        
        result = manifest_processor._convert_date_columns(df, ['date_commande'])
        
        assert pd.isna(result['date_commande'].iloc[1])
        assert not pd.isna(result['date_commande'].iloc[0])

    def test_adjust_date_commande_from_recyclage(self, manifest_processor):
        """Test adjusting date_commande based on recyclage dates.
        
        Logic: recyclage_date must be >= date_commande and < date_suspension.
        The function iterates through recyclage columns and returns the first matching date,
        or keeps the original date_commande if no recyclage date meets the criteria.
        """
        df = pd.DataFrame({
            'date_commande': [datetime(2025, 1, 1), datetime(2025, 1, 1)],
            'date_suspension': [datetime(2025, 1, 15), datetime(2025, 1, 15)],
            'premiere_date_recyclage': [pd.NaT, pd.NaT],
            'deuxieme_date_recyclage': [pd.NaT, pd.NaT]
        })
        
        result = manifest_processor._adjust_date_commande_from_recyclage(df, 'ADSL')
        
        # Both rows should keep original date_commande since no valid recyclage dates
        assert result['date_commande'].iloc[0] == datetime(2025, 1, 1)
        assert result['date_commande'].iloc[1] == datetime(2025, 1, 1)

    @patch('services.manifest.get_calls')
    def test_extract_calls_metadata(self, mock_get_calls, manifest_processor):
        """Test extracting calls metadata from DataFrame."""
        df = pd.DataFrame({
            'client_number': ['212612345678', '212712345678'],
            'date_commande': [datetime(2025, 1, 1), datetime(2025, 1, 1)],
            'date_suspension': [datetime(2025, 1, 10), datetime(2025, 1, 10)],
            'numero_commande': ['CMD001', 'CMD002'],
            'motif_suspension': ['client refuse installation', 'client injoignable']
        })
        
        mock_get_calls.side_effect = [
            [{'call_id': 'call1'}],
            [{'call_id': 'call2'}, {'call_id': 'call3'}]
        ]
        
        result = manifest_processor._extract_calls_metadata(df)
        
        assert len(result) == 2
        assert len(result[0]) == 1
        assert len(result[1]) == 2
        assert mock_get_calls.call_count == 2

    @patch('services.manifest.get_calls')
    def test_extract_calls_metadata_handles_none(self, mock_get_calls, manifest_processor):
        """Test that None from get_calls is converted to empty list."""
        df = pd.DataFrame({
            'client_number': ['212612345678'],
            'date_commande': [datetime(2025, 1, 1)],
            'date_suspension': [datetime(2025, 1, 10)],
            'numero_commande': ['CMD001'],
            'motif_suspension': ['client refuse installation']
        })
        
        mock_get_calls.return_value = None
        
        result = manifest_processor._extract_calls_metadata(df)
        
        assert len(result) == 1
        assert result[0] == []

    @patch('services.manifest.get_calls')
    def test_extract_calls_metadata_handles_dict(self, mock_get_calls, manifest_processor):
        """Test that single dict from get_calls is converted to list."""
        df = pd.DataFrame({
            'client_number': ['212612345678'],
            'date_commande': [datetime(2025, 1, 1)],
            'date_suspension': [datetime(2025, 1, 10)],
            'numero_commande': ['CMD001'],
            'motif_suspension': ['client refuse installation']
        })
        
        mock_get_calls.return_value = {'call_id': 'call1'}
        
        result = manifest_processor._extract_calls_metadata(df)
        
        assert len(result) == 1
        assert isinstance(result[0], list)
        assert len(result[0]) == 1

    def test_read_df_csv_utf8(self, manifest_processor):
        """Test reading CSV file with UTF-8 encoding."""
        with patch('pandas.read_csv') as mock_read:
            mock_read.return_value = pd.DataFrame({'col': [1, 2]})
            
            result = manifest_processor._read_df('/path/to/file.csv')
            
            assert mock_read.called
            assert not result.empty

    def test_read_df_xlsx(self, manifest_processor):
        """Test reading Excel file."""
        with patch('pandas.read_excel') as mock_read:
            mock_read.return_value = pd.DataFrame({'col': [1, 2]})
            
            result = manifest_processor._read_df('/path/to/file.xlsx')
            
            assert mock_read.called
            assert not result.empty

    def test_read_df_fallback_encodings(self, manifest_processor):
        """Test fallback to different encodings."""
        with patch('pandas.read_csv') as mock_read:
            # First call raises UnicodeDecodeError, second succeeds
            mock_read.side_effect = [
                UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid'),
                pd.DataFrame({'col': [1, 2]})
            ]
            
            result = manifest_processor._read_df('/path/to/file.csv')
            
            assert mock_read.call_count == 2
            assert not result.empty
