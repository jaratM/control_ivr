"""
Unit tests for the transcription module.
"""
import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch, Mock

from modules.transcription import Transcriber
from modules.types import AudioSegment, TranscriptionResult


class TestTranscriber:
    """Test cases for the Transcriber class."""
    
    def test_init_with_cpu(self, sample_config):
        """Test transcriber initialization with CPU."""
        with patch('modules.transcription.Wav2Vec2BertForCTC') as mock_model, \
             patch('modules.transcription.Wav2Vec2BertProcessor') as mock_processor:
            
            mock_model_instance = MagicMock()
            mock_model.from_pretrained.return_value = mock_model_instance
            mock_processor_instance = MagicMock()
            mock_processor.from_pretrained.return_value = mock_processor_instance
            
            transcriber = Transcriber(device='cpu', config=sample_config)
            
            assert transcriber.device == 'cpu'
            assert transcriber.model is not None
            assert transcriber.processor is not None
            mock_model.from_pretrained.assert_called_once()
            mock_processor.from_pretrained.assert_called_once()
    
    def test_init_with_custom_model_path(self):
        """Test transcriber initialization with custom model path."""
        custom_config = {
            'processing': {
                'transcription_model': '/custom/path/to/model'
            }
        }
        
        with patch('modules.transcription.Wav2Vec2BertForCTC') as mock_model, \
             patch('modules.transcription.Wav2Vec2BertProcessor') as mock_processor:
            
            mock_model_instance = MagicMock()
            mock_model.from_pretrained.return_value = mock_model_instance
            mock_processor_instance = MagicMock()
            mock_processor.from_pretrained.return_value = mock_processor_instance
            
            transcriber = Transcriber(device='cpu', config=custom_config)
            
            # Check that the custom path was used
            call_args = mock_model.from_pretrained.call_args[0]
            assert '/custom/path/to/model' in call_args[0]
    
    def test_transcribe_batch_empty(self, mock_transcriber):
        """Test transcribing an empty batch."""
        result = mock_transcriber.transcribe_batch([])
        # Should return empty list or handle gracefully
        assert isinstance(result, list)
    
    def test_transcribe_batch_single_segment(self, sample_config, sample_audio_segment):
        """Test transcribing a single audio segment."""
        with patch('modules.transcription.Wav2Vec2BertForCTC') as mock_model, \
             patch('modules.transcription.Wav2Vec2BertProcessor') as mock_processor:
            
            # Setup mocks
            mock_model_instance = MagicMock()
            mock_model.from_pretrained.return_value = mock_model_instance
            mock_model_instance.eval.return_value = None
            
            # Mock the forward pass
            mock_logits = torch.randn(1, 100, 50)  # (batch, time, vocab)
            mock_model_instance.return_value.logits = mock_logits
            
            mock_processor_instance = MagicMock()
            mock_processor.from_pretrained.return_value = mock_processor_instance
            
            # Mock processor call
            mock_inputs = {
                'input_features': torch.randn(1, 100, 80)
            }
            mock_processor_instance.return_value = mock_inputs
            
            # Mock batch_decode
            mock_processor_instance.batch_decode.return_value = ["test transcription"]
            
            transcriber = Transcriber(device='cpu', config=sample_config)
            transcriber.model = mock_model_instance
            transcriber.processor = mock_processor_instance
            
            results = transcriber.transcribe_batch([sample_audio_segment])
            
            assert len(results) == 1
            assert isinstance(results[0], TranscriptionResult)
            assert results[0].file_id == "test_file_001"
            assert results[0].segment_index == 0
            assert isinstance(results[0].text, str)
    
    def test_transcribe_batch_multiple_segments(self, sample_config):
        """Test transcribing multiple audio segments."""
        # Create multiple segments
        segments = [
            AudioSegment(
                file_id=f"test_{i}",
                segment_index=i,
                audio_data=np.random.randn(16000).astype(np.float32),
                duration=2.0,
                # start_time=float(i * 2),
                # end_time=float(i * 2 + 2)
            )
            for i in range(3)
        ]
        
        with patch('modules.transcription.Wav2Vec2BertForCTC') as mock_model, \
             patch('modules.transcription.Wav2Vec2BertProcessor') as mock_processor:
            
            # Setup mocks
            mock_model_instance = MagicMock()
            mock_model.from_pretrained.return_value = mock_model_instance
            mock_model_instance.eval.return_value = None
            
            # Mock the forward pass for batch of 3
            mock_logits = torch.randn(3, 100, 50)
            mock_model_instance.return_value.logits = mock_logits
            
            mock_processor_instance = MagicMock()
            mock_processor.from_pretrained.return_value = mock_processor_instance
            
            mock_inputs = {
                'input_features': torch.randn(3, 100, 80)
            }
            mock_processor_instance.return_value = mock_inputs
            
            mock_processor_instance.batch_decode.return_value = [
                "transcription 1",
                "transcription 2",
                "transcription 3"
            ]
            
            transcriber = Transcriber(device='cpu', config=sample_config)
            transcriber.model = mock_model_instance
            transcriber.processor = mock_processor_instance
            
            results = transcriber.transcribe_batch(segments)
            
            assert len(results) == 3
            for i, result in enumerate(results):
                assert isinstance(result, TranscriptionResult)
                assert result.file_id == f"test_{i}"
                assert result.segment_index == i
    
    def test_transcribe_batch_handles_padding(self, sample_config):
        """Test that transcriber handles different length audio segments."""
        # Create segments of different lengths
        segments = [
            AudioSegment(
                file_id="short",
                segment_index=0,
                audio_data=np.random.randn(8000).astype(np.float32),
                duration=0.5,
                # start_time=0.0,
                # end_time=0.5
            ),
            AudioSegment(
                file_id="long",
                segment_index=1,
                audio_data=np.random.randn(32000).astype(np.float32),
                duration=2.0,
                # start_time=0.0,
                # end_time=2.0
            )
        ]
        
        with patch('modules.transcription.Wav2Vec2BertForCTC') as mock_model, \
             patch('modules.transcription.Wav2Vec2BertProcessor') as mock_processor:
            
            mock_model_instance = MagicMock()
            mock_model.from_pretrained.return_value = mock_model_instance
            mock_model_instance.eval.return_value = None
            
            mock_logits = torch.randn(2, 100, 50)
            mock_model_instance.return_value.logits = mock_logits
            
            mock_processor_instance = MagicMock()
            mock_processor.from_pretrained.return_value = mock_processor_instance
            
            # Mock processor with padding
            mock_inputs = {
                'input_features': torch.randn(2, 100, 80)
            }
            mock_processor_instance.return_value = mock_inputs
            mock_processor_instance.batch_decode.return_value = ["text1", "text2"]
            
            transcriber = Transcriber(device='cpu', config=sample_config)
            transcriber.model = mock_model_instance
            transcriber.processor = mock_processor_instance
            
            results = transcriber.transcribe_batch(segments)
            
            # Verify processor was called with padding=True
            call_kwargs = mock_processor_instance.call_args[1]
            assert call_kwargs.get('padding') is True
            assert len(results) == 2
    
    def test_model_evaluation_mode(self, sample_config):
        """Test that model is set to evaluation mode."""
        with patch('modules.transcription.Wav2Vec2BertForCTC') as mock_model, \
             patch('modules.transcription.Wav2Vec2BertProcessor') as mock_processor:
            
            mock_model_instance = MagicMock()
            # Make .to() return the same mock instance
            mock_model_instance.to.return_value = mock_model_instance
            mock_model.from_pretrained.return_value = mock_model_instance
            mock_processor_instance = MagicMock()
            mock_processor.from_pretrained.return_value = mock_processor_instance
            
            transcriber = Transcriber(device='cpu', config=sample_config)
            
            # Verify eval() was called
            mock_model_instance.eval.assert_called_once()
    
    
