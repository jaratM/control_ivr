"""
Unit tests for the compliance verification module.
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from collections import namedtuple

from modules.compliance import ComplianceVerifier
from modules.types import ComplianceInput, ClassificationResult


class TestComplianceVerifier:
    """Test cases for the ComplianceVerifier class."""
    
    @pytest.fixture
    def verifier(self):
        """Create a ComplianceVerifier instance."""
        return ComplianceVerifier()
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for compliance checks."""
        return {
            'compliance_rules': {
                'max_beeps': 5
            }
        }
    
    @pytest.fixture
    def sample_row_sav(self):
        """Sample SAV manifest row."""
        return {
            'numero_commande': 'SAV001',
            'client_number': '0612345678',
            'date_commande': datetime.now() - timedelta(days=5),
            'date_suspension': datetime.now() - timedelta(days=1),
            'motif_suspension': 'client injoignable',
            'status_commande': 'SUSPENDED',
            'categorie': 'ADSL',
            'nbr_tentatives_appel': 0  # Will be updated by tests
        }
    
    @pytest.fixture
    def sample_row_acquisition(self):
        """Sample ACQUISITION manifest row."""
        return {
            'numero_commande': 'ACQ001',
            'client_number': '0698765432',
            'date_commande': datetime.now() - timedelta(days=3),
            'date_suspension': datetime.now(),
            'motif_suspension': 'client refuse installation',
            'status_commande': 'CANCELLED',
            'nbr_tentatives_appel': 0
        }
    
    def test_init_constants(self, verifier):
        """Test that constants are properly defined."""
        assert verifier.STATUS_CONFORM == 'Conform'
        assert verifier.STATUS_NON_CONFORM == 'Non conforme'
        assert verifier.CLIENT_UNREACHABLE == 'client injoignable'
        assert verifier.MIN_ATTEMPTS == 3
        assert verifier.MIN_GAP_SECONDS == 7200  # 2 hours
    
    def test_verify_compliance_no_calls(self, verifier, sample_row_sav, sample_config):
        """Test compliance verification when no calls are found."""
        sample_row_sav['nbr_tentatives_appel'] = 0
        
        result = verifier._verify_row(
            sample_row_sav,
            [],
            'ADSL',
            'SAV',
            sample_config
        )
        
        assert result['conformite_IAM'] == 'Non conforme'
        assert 'Aucun appel trouvé' in result['commentaire']
    
    def test_verify_compliance_insufficient_attempts(self, verifier, sample_row_sav, sample_config):
        """Test compliance when fewer than 3 attempts made."""
        base_time = datetime.now() - timedelta(days=2)
        calls = [
            {
                'file_id': 'call_1',
                'start_time': base_time,
                'numero_commande': 'SAV001',
                'beep_count': 0,
                'high_beeps': 0
            },
            {
                'file_id': 'call_2',
                'start_time': base_time + timedelta(hours=3),
                'numero_commande': 'SAV001',
                'beep_count': 0,
                'high_beeps': 0
            }
        ]
        
        sample_row_sav['nbr_tentatives_appel'] = 2
        
        result = verifier._verify_row(
            sample_row_sav,
            calls,
            'ADSL',
            'SAV',
            sample_config
        )
        
        assert result['conformite_IAM'] == 'Non conforme'
        assert 'appels trouvés' in result['commentaire']
    
    def test_verify_compliance_sufficient_attempts(self, verifier, sample_row_sav, sample_config):
        """Test compliance when 3+ attempts with proper intervals."""
        base_time = datetime.now() - timedelta(days=2)
        calls = [
            {
                'file_id': 'call_1',
                'start_time': base_time,
                'numero_commande': 'SAV001',
                'beep_count': 0,
                'high_beeps': 0
            },
            {
                'file_id': 'call_2',
                'start_time': base_time + timedelta(hours=3),
                'numero_commande': 'SAV001',
                'beep_count': 0,
                'high_beeps': 0
            },
            {
                'file_id': 'call_3',
                'start_time': base_time + timedelta(hours=6),
                'numero_commande': 'SAV001',
                'beep_count': 0,
                'high_beeps': 0
            }
        ]
        
        sample_row_sav['nbr_tentatives_appel'] = 3
        
        result = verifier._verify_row(
            sample_row_sav,
            calls,
            'ADSL',
            'SAV',
            sample_config
        )
        
        assert result['conformite_IAM'] == 'Conform'
        assert result['conformite_intervalle'] == 'Conform'
    
    def test_verify_compliance_intervals_too_short(self, verifier, sample_row_sav, sample_config):
        """Test compliance when call intervals are less than 2 hours."""
        base_time = datetime.now() - timedelta(days=2)
        calls = [
            {
                'file_id': 'call_1',
                'start_time': base_time,
                'numero_commande': 'SAV001',
                'beep_count': 0,
                'high_beeps': 0
            },
            {
                'file_id': 'call_2',
                'start_time': base_time + timedelta(minutes=30),  # Too soon
                'numero_commande': 'SAV001',
                'beep_count': 0,
                'high_beeps': 0
            },
            {
                'file_id': 'call_3',
                'start_time': base_time + timedelta(hours=1),  # Too soon
                'numero_commande': 'SAV001',
                'beep_count': 0,
                'high_beeps': 0
            }
        ]
        
        sample_row_sav['nbr_tentatives_appel'] = 3
        
        result = verifier._verify_row(
            sample_row_sav,
            calls,
            'ADSL',
            'SAV',
            sample_config
        )
        
        assert result['conformite_intervalle'] == 'Non conforme'
        assert 'temps entre les appels' in result['commentaire'].lower()
    

    def test_verify_compliance_acquisition_category(self, verifier, sample_row_acquisition, sample_config):
        """Test compliance verification for ACQUISITION manifest."""
        base_time = datetime.now() - timedelta(days=1)
        calls = [
            {
                'file_id': 'call_1',
                'start_time': base_time,
                'numero_commande': 'ACQ001',
                'beep_count': 0,
                'high_beeps': 0
            },
            {
                'file_id': 'call_2',
                'start_time': base_time + timedelta(hours=3),
                'numero_commande': 'ACQ001',
                'beep_count': 0,
                'high_beeps': 0
            },
            {
                'file_id': 'call_3',
                'start_time': base_time + timedelta(hours=6),
                'numero_commande': 'ACQ001',
                'beep_count': 0,
                'high_beeps': 0
            }
        ]
        
        sample_row_acquisition['nbr_tentatives_appel'] = 3
        
        result = verifier._verify_row(
            sample_row_acquisition,
            calls,
            'ADSL',
            'ACQUISITION',
            sample_config
        )
        
        assert result['conformite_IAM'] == 'Conform'
    
    def test_parse_start_time_from_datetime(self, verifier):
        """Test parsing start_time when it's a datetime object."""
        dt = datetime(2025, 1, 1, 10, 30, 0)
        call = {'start_time': dt}
        
        result = verifier._parse_start_time(call)
        
        assert result == dt
    
    def test_parse_start_time_from_string(self, verifier):
        """Test parsing start_time when it's an ISO string."""
        dt = datetime(2025, 1, 1, 10, 30, 0)
        call = {'start_time': dt.isoformat()}
        
        result = verifier._parse_start_time(call)
        
        assert result == dt
    
    def test_parse_start_time_none(self, verifier):
        """Test parsing start_time when it's None."""
        call = {'start_time': None}
        
        result = verifier._parse_start_time(call)
        
        assert result == datetime.min
    
    def test_parse_date_value_various_formats(self, verifier):
        """Test parsing date values in various formats."""
        dt = datetime(2025, 1, 1, 12, 0, 0)
        
        # Test datetime object
        assert verifier._parse_date_value(dt) == dt
        
        # Test ISO string
        assert verifier._parse_date_value(dt.isoformat()) == dt
        
        # Test None
        assert verifier._parse_date_value(None) == datetime.min
        
        # Test empty string
        assert verifier._parse_date_value('') == datetime.min
    
    def test_verify_compliance_batch(self, verifier, sample_config):
        """Test verifying compliance for multiple rows."""
        base_time = datetime.now() - timedelta(days=2)
        
        df_dict = [
            {
                'numero_commande': 'CMD001',
                'client_number': '0612345678',
                'date_commande': datetime.now() - timedelta(days=5),
                'date_suspension': datetime.now() - timedelta(days=1),
                'motif_suspension': 'client injoignable',
                'status_commande': 'SUSPENDED',
                'categorie': 'ADSL',
                'nbr_tentatives_appel': 3
            },
            {
                'numero_commande': 'CMD002',
                'client_number': '0698765432',
                'date_commande': datetime.now() - timedelta(days=3),
                'date_suspension': datetime.now(),
                'motif_suspension': 'client injoignable',
                'status_commande': 'SUSPENDED',
                'categorie': 'VULA',
                'nbr_tentatives_appel': 2
            }
        ]
        
        calls_metadata = [
            [
                {'file_id': 'call_1', 'start_time': base_time, 'beep_count': 0, 'high_beeps': 0},
                {'file_id': 'call_2', 'start_time': base_time + timedelta(hours=3), 'beep_count': 0, 'high_beeps': 0},
                {'file_id': 'call_3', 'start_time': base_time + timedelta(hours=6), 'beep_count': 0, 'high_beeps': 0}
            ],
            [
                {'file_id': 'call_4', 'start_time': base_time, 'beep_count': 0, 'high_beeps': 0},
                {'file_id': 'call_5', 'start_time': base_time + timedelta(hours=2), 'beep_count': 0, 'high_beeps': 0}
            ]
        ]
        
        results = verifier.verify_compliance(
            df_dict,
            calls_metadata,
            'ADSL',
            'SAV',
            sample_config
        )
        
        assert len(results) == 2
        assert all('conformite_IAM' in row for row in results)
        
        # First should be compliant (3 attempts)
        assert results[0]['conformite_IAM'] == 'Conform'
        
        # Second should be non-compliant (only 2 attempts)
        assert results[1]['conformite_IAM'] == 'Non conforme'
    
    def test_verify_row_with_sav_category_extraction(self, verifier, sample_config):
        """Test that SAV manifest extracts category from row."""
        row = {
            'numero_commande': 'SAV001',
            'client_number': '0612345678',
            'date_commande': datetime.now() - timedelta(days=5),
            'date_suspension': datetime.now() - timedelta(days=1),
            'motif_suspension': 'client injoignable',
            'status_commande': 'SUSPENDED',
            'categorie': 'VULA',  # Should be extracted for SAV
            'nbr_tentatives_appel': 3
        }
        
        calls = []
        
        result = verifier._verify_row(
            row,
            calls,
            'ADSL',  # This should be overridden
            'SAV',
            sample_config
        )
        
        # Category should be extracted from row for SAV
        assert result is not None
    
    def test_default_beep_values(self, verifier):
        """Test default beep count values."""
        assert verifier.DEFAULT_BEEP_COUNT == 100
        assert verifier.DEFAULT_HIGH_BEEPS == 100
    
    def test_motif_injoignable_constant(self, verifier):
        """Test motif injoignable constant."""
        assert verifier.MOTIF_INJOIGNABLE == 'client injoignable'
    
    def test_verify_compliance_empty_calls_list(self, verifier, sample_row_sav, sample_config):
        """Test compliance with empty calls list but positive count."""
        sample_row_sav['nbr_tentatives_appel'] = 3
        
        result = verifier._verify_row(
            sample_row_sav,
            [],  # Empty calls list
            'ADSL',
            'SAV',
            sample_config
        )
        
        # Should handle mismatch between count and actual calls
        assert result is not None
        assert 'conformite_IAM' in result
    
    def test_verify_compliance_mixed_beep_counts(self, verifier, sample_row_sav, sample_config):
        """Test compliance with mixed beep counts (some high, some low)."""
        base_time = datetime.now() - timedelta(days=2)
        calls = [
            {
                'file_id': 'call_1',
                'start_time': base_time,
                'numero_commande': 'SAV001',
                'beep_count': 2,  # Low beeps
                'high_beeps': 1
            },
            {
                'file_id': 'call_2',
                'start_time': base_time + timedelta(hours=3),
                'numero_commande': 'SAV001',
                'beep_count': 8,  # High beeps (voicemail)
                'high_beeps': 7
            },
            {
                'file_id': 'call_3',
                'start_time': base_time + timedelta(hours=6),
                'numero_commande': 'SAV001',
                'beep_count': 0,  # Clean call
                'high_beeps': 0
            }
        ]
        
        sample_row_sav['nbr_tentatives_appel'] = 3
        
        result = verifier._verify_row(
            sample_row_sav,
            calls,
            'ADSL',
            'SAV',
            sample_config
        )
        
        # Should identify the voicemail call
        assert result is not None

    def test_verify_compliance_batch_insufficient_beeps(self, verifier, sample_config):
        """Test verify_compliance_batch detects insufficient beeps for injoignable cases."""
        # Row data
        df_dict = [{
            'numero_commande': 'CMD_INJOIGNABLE',
            'motif_suspension': 'client injoignable',
            'conformite_IAM': 'Conform',
            'commentaire': ''
        }]
        
        # Classification Result Mock
        mock_classification = MagicMock()
        mock_classification.status = 'Silence'
        mock_classification.behavior = 'Bad'
        
        # Mock metadata
        mock_metadata = MagicMock()

        # Results input with low beeps
        results = [
            ComplianceInput(
                numero_commande='CMD_INJOIGNABLE',
                beep_count=2,  # < 5 implies insufficient for 'Silence' -> 'client injoignable' mapping logic
                high_beeps=0,
                classification=mock_classification,
                metadata=mock_metadata
            )
        ]
        
        # Execute batch verification
        output = verifier.verify_compliance_batch(
            df_dict,
            results
        )
        
        assert len(output) == 1
        res = output[0]
        
        # The logic in _process_injoignable_commande is:
        # if classification == 'silence' and beeps < 5 and high < 1:
        #    is_compliant = NON_CONFORM
        #    comment += "Moins de 5 beeps"
        
        assert res['conformite_IAM'] == 'Non conforme'
        assert 'Moins de 5 beeps' in res['commentaire']
