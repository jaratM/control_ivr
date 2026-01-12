"""
Unit tests for the classification module.
"""
import pytest
import json
from unittest.mock import MagicMock, patch, Mock, mock_open

from modules.classification import Classifier


class TestClassifier:
    """Test cases for the Classifier class."""
    
    @pytest.fixture
    def mock_prompt_file(self, temp_dir):
        """Create a mock prompt file."""
        prompt_path = f"{temp_dir}/sav.txt"
        prompt_content = """Tu es un classificateur d'appels téléphoniques.
        Classifie l'appel selon les catégories suivantes:
        0: Silence
        1: Client refuse installation
        2: Client reporte RDV
        3: Client injoignable
        4: Autre
        5: Attente retour client
        6: Client absent
        7: Absence routeur client
        8: Local fermé
        
        Retourne uniquement deux chiffres: catégorie + comportement technicien (0=mauvais, 1=bien)
        """
        
        with open(prompt_path, 'w', encoding='utf-8') as f:
            f.write(prompt_content)
        
        return prompt_path
    
    @pytest.fixture
    def classifier_config(self, temp_dir, mock_prompt_file):
        """Configuration for classifier with mocked prompt file."""
        return {
            'api_url': 'http://test-api.com/classify',
            'classification': {
                'sav': mock_prompt_file,
                'acquisition': mock_prompt_file
            }
        }
    
    def test_init_with_api_url_parameter(self, classifier_config):
        """Test classifier initialization does not accept api_url but uses env var."""
        with pytest.raises(TypeError):
            Classifier(
                category='sav',
                api_url='http://custom-api.com',
                log=False,
                config=classifier_config
            )
    
    def test_init_with_env_api_url(self, classifier_config, monkeypatch):
        """Test classifier initialization with API URL from environment."""
        monkeypatch.setenv('BEDROCK_API_URL', 'http://env-api.com')
        
        classifier = Classifier(
            category='sav',
            log=False,
            config=classifier_config
        )
        
        assert classifier.api_url == 'http://env-api.com'

    
    def test_init_without_api_url_does_not_raise_error(self, monkeypatch):
        """Test that initialization without API URL does not raise error immediately."""
        monkeypatch.delenv('BEDROCK_API_URL', raising=False)
        monkeypatch.delenv('BEDROCK_API_ID', raising=False)
        
        # Should not raise
        classifier = Classifier(category='sav', log=False, config={})
        assert classifier.api_url is None
    
    def test_init_loads_prompt_from_file(self, classifier_config):
        """Test that classifier loads system prompt from file."""
        classifier = Classifier(
            category='sav',
            log=False,
            config=classifier_config
        )
        
        assert classifier.system_prompt is not None
        assert len(classifier.system_prompt) > 0
        assert 'classificateur' in classifier.system_prompt.lower() or 'client' in classifier.system_prompt.lower()
    
    def test_category_map(self, classifier_config):
        """Test that category map is correctly defined."""
        classifier = Classifier(
            category='sav',
            log=False,
            config=classifier_config
        )
        
        assert len(classifier.CATEGORY_MAP) == 9
        assert classifier.CATEGORY_MAP[0] == "Silence"
        assert classifier.CATEGORY_MAP[1] == "Client refuse installation"
        assert classifier.CATEGORY_MAP[3] == "CLIENT INJOIGNABLE"
    
    def test_technician_behavior_map(self, classifier_config):
        """Test that technician behavior map is correctly defined."""
        classifier = Classifier(
            category='sav',
            log=False,
            config=classifier_config
        )
        
        assert classifier.TECHNICIAN_BEHAVIOR_MAP[0] == "Mauvais"
        assert classifier.TECHNICIAN_BEHAVIOR_MAP[1] == "Bien"
    
    @patch('modules.classification.requests.post')
    def test_classify_full_text_success(self, mock_post, classifier_config, monkeypatch):
        """Test successful classification of text."""
        monkeypatch.setenv('BEDROCK_API_URL', 'http://test-api.com')
        # Mock API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'status': 'success',
            'data': {'response': '11'}
        }
        mock_post.return_value = mock_response
        
        classifier = Classifier(
            category='sav',
            log=False,
            config=classifier_config
        )
        
        result = classifier.classify_full_text(
            "السلام عليكم، بغيت نلغي الطلب ديالي",
            "test_file_001"
        )
        
        assert result.file_id == 'test_file_001'
        assert result.status == 'Client refuse installation'
        assert result.behavior == 'Bien'
    
    @patch('modules.classification.requests.post')
    def test_classify_full_text_different_outcomes(self, mock_post, classifier_config, monkeypatch):
        """Test classification with different call types."""
        monkeypatch.setenv('BEDROCK_API_URL', 'http://test-api.com')
        classifier = Classifier(
            category='sav',
            log=False,
            config=classifier_config
        )
        
        test_cases = [
            ('30', 'CLIENT INJOIGNABLE', 'Mauvais'),  # Unreachable + Bad
            ('61', 'Client Absent', 'Bien'),            # Absent + Good
            ('21', 'Client reporte RDV', 'Bien'),       # Reschedule + Good
        ]
        
        for response_code, expected_status, expected_behavior in test_cases:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'status': 'success',
                'data': {'response': response_code}
            }
            mock_post.return_value = mock_response
            
            result = classifier.classify_full_text("test text", "test_001")
            
            assert result.status == expected_status
            assert result.behavior == expected_behavior
    
    def test_classify_empty_text(self, classifier_config):
        """Test classification of empty text returns default (Silence + Good)."""
        classifier = Classifier(
            category='sav',
            log=False,
            config=classifier_config
        )
        
        result = classifier.classify_full_text("", "test_001")
        
        assert result.status == 'Silence'
        assert result.behavior == 'Bien'
    
    def test_classify_very_short_text(self, classifier_config):
        """Test classification of very short text (< 5 chars)."""
        classifier = Classifier(
            category='sav',
            log=False,
            config=classifier_config
        )
        
        result = classifier.classify_full_text("abc", "test_001")
        
        assert result.status == 'Silence'
        assert result.behavior == 'Bien'  # Good
    
    @patch('modules.classification.requests.post')
    def test_classify_api_error_returns_default(self, mock_post, classifier_config):
        """Test that API errors return default classification (Other + Good)."""
        # Mock API error
        mock_post.side_effect = Exception("API connection failed")
        
        classifier = Classifier(
            category='sav',
            log=False,
            config=classifier_config
        )
        
        result = classifier.classify_full_text("test text", "test_001")
        
        assert result.status == 'autre'  # Other
        assert result.behavior == 'Bien'  # Good
    
    @patch('modules.classification.requests.post')
    def test_classify_invalid_response_format(self, mock_post, classifier_config):
        """Test handling of invalid API response format."""
        # Mock invalid response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'completion': 'invalid',  # Not a two-digit number
            'stop_reason': 'end_turn'
        }
        mock_post.return_value = mock_response
        
        classifier = Classifier(
            category='sav',
            log=False,
            config=classifier_config
        )
        
        result = classifier.classify_full_text("test text", "test_001")
        
        # Should default to Other + Good
        assert result.status == 'autre'
        assert result.behavior == 'Bien'
    
    @patch('modules.classification.requests.post')
    def test_make_result_helper(self, mock_post, classifier_config):
        """Test the _make_result helper method."""
        classifier = Classifier(
            category='sav',
            log=False,
            config=classifier_config
        )
        
        result = classifier._make_result(1, 1, "test_123")
        
        assert result.file_id == 'test_123'
        assert result.status == 'Client refuse installation'
        assert result.behavior == 'Bien'
    
    @patch('modules.classification.requests.post')
    def test_make_result_invalid_behavior_defaults_to_good(self, mock_post, classifier_config):
        """Test that invalid technician behavior defaults to Good (1)."""
        classifier = Classifier(
            category='sav',
            log=False,
            config=classifier_config
        )
        
        result = classifier._make_result(1, 99, "test_123")  # Invalid behavior
        
        assert result.behavior == 'Bien'  # Should default to Good
    
    @patch('modules.classification.requests.post')
    def test_build_prompts(self, mock_post, classifier_config):
        """Test that prompts are correctly built."""
        classifier = Classifier(
            category='sav',
            log=False,
            config=classifier_config
        )
        
        test_text = "Test transcription text"
        system_prompt, user_prompt = classifier._build_prompts(test_text)
        
        assert system_prompt is not None
        assert test_text in user_prompt  # The text should be in the prompt
        assert 'Transcription:' in user_prompt  # Should have few-shot format
        assert len(system_prompt) > 0
    
    @patch('modules.classification.requests.post')
    def test_call_llm_method(self, mock_post, classifier_config, monkeypatch):
        """Test the _call_llm method."""
        monkeypatch.setenv('BEDROCK_API_URL', 'http://test-api.com')
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'status': 'success',
            'data': {'response': '31'}
        }
        mock_post.return_value = mock_response
        
        classifier = Classifier(
            category='sav',
            log=False,
            config=classifier_config
        )
        
        response = classifier._call_llm("system prompt", "user prompt")
        
        assert '31' in response
        mock_post.assert_called_once()
    
    @patch('modules.classification.requests.post')
    def test_parse_response_method(self, mock_post, classifier_config):
        """Test the _parse_response method."""
        classifier = Classifier(
            category='sav',
            log=False,
            config=classifier_config
        )
        
        call_type, tech_behavior, reasoning = classifier._parse_response("21")
        
        assert call_type == 2
        assert tech_behavior == 1
    
    @patch('modules.classification.requests.post')
    def test_classification_with_retry_logic(self, mock_post, classifier_config):
        """Test classification handles retries on API failures."""
        # First call fails, second succeeds
        mock_response_fail = MagicMock()
        mock_response_fail.status_code = 500
        
        mock_response_success = MagicMock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {
            'status': 'success',
            'data': {'response': '11'}
        }
        
        mock_post.side_effect = [
            Exception("Connection error"),
            mock_response_success
        ]
        
        config_with_retry = {**classifier_config, 'max_attempts': 5}
        
        classifier = Classifier(
            category='sav',
            log=False,
            config=config_with_retry
        )
        
        # Should eventually succeed or default gracefully
        result = classifier.classify_full_text("test text", "test_001")
        
        assert result is not None
        assert result.file_id == 'test_001'
        assert hasattr(result, 'status')
        assert hasattr(result, 'behavior')
