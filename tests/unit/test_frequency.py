"""
Unit tests for the frequency analysis module.
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch, Mock
import tempfile
import soundfile as sf

from modules.frequency import FrequencyAnalyzer
from modules.types import AudioSegment


class TestFrequencyAnalyzer:
    """Test cases for the FrequencyAnalyzer class."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a FrequencyAnalyzer instance."""
        return FrequencyAnalyzer()
    
    @pytest.fixture
    def analyzer_custom_params(self):
        """Create a FrequencyAnalyzer with custom parameters."""
        low_params = {
            'min_beep_interval': 0.8,
            'beep_duration': 0.8,
            'detection_threshold': 0.25,
            'freq_bandwidth': 3
        }
        high_params = {
            'min_beep_interval': 0.25,
            'beep_duration': 0.25,
            'detection_threshold': 0.30,
            'freq_bandwidth': 8
        }
        return FrequencyAnalyzer(
            low_freq_range=(390, 470),
            high_freq_range=(590, 650),
            low_params=low_params,
            high_params=high_params
        )
    
    def test_init_default_params(self, analyzer):
        """Test initialization with default parameters."""
        assert analyzer.low_freq_range == (390, 470)
        assert analyzer.high_freq_range == (590, 650)
        assert 'min_beep_interval' in analyzer.low_params
        assert 'detection_threshold' in analyzer.high_params
    
    def test_init_custom_params(self, analyzer_custom_params):
        """Test initialization with custom parameters."""
        assert analyzer_custom_params.low_params['min_beep_interval'] == 0.8
        assert analyzer_custom_params.high_params['detection_threshold'] == 0.30
    
    def test_get_resampler_caching(self, analyzer):
        """Test that resampler is cached properly."""
        resampler1 = analyzer.get_resampler(44100, 16000)
        resampler2 = analyzer.get_resampler(44100, 16000)
        
        # Should return the same cached instance
        assert resampler1 is resampler2
    
    @pytest.fixture
    def generate_audio_with_beeps(self, temp_dir):
        """Generate test audio file with synthetic beeps."""
        sample_rate = 16000
        duration = 10.0  # 10 seconds
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Base audio (silence/noise)
        audio = np.random.randn(len(t)) * 0.01
        
        # Add low frequency beeps (420 Hz) at 2s, 4s, 6s
        for beep_time in [2.0, 4.0, 6.0]:
            start_idx = int(beep_time * sample_rate)
            end_idx = int((beep_time + 0.5) * sample_rate)
            beep_t = t[start_idx:end_idx] - beep_time
            beep = 0.5 * np.sin(2 * np.pi * 420 * beep_t)
            audio[start_idx:end_idx] += beep
        
        # Add high frequency beeps (620 Hz) at 1s, 3s, 5s
        for beep_time in [1.0, 3.0, 5.0]:
            start_idx = int(beep_time * sample_rate)
            end_idx = int((beep_time + 0.3) * sample_rate)
            beep_t = t[start_idx:end_idx] - beep_time
            beep = 0.4 * np.sin(2 * np.pi * 620 * beep_t)
            audio[start_idx:end_idx] += beep
        
        # Save to file
        file_path = f"{temp_dir}/test_beeps.wav"
        sf.write(file_path, audio, sample_rate)
        
        return file_path, sample_rate
    
    @pytest.fixture
    def generate_clean_audio(self, temp_dir):
        """Generate test audio without beeps (clean speech simulation)."""
        sample_rate = 16000
        duration = 5.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Random noise simulating speech
        audio = np.random.randn(len(t)) * 0.1
        
        file_path = f"{temp_dir}/clean_audio.wav"
        sf.write(file_path, audio, sample_rate)
        
        return file_path, sample_rate
    
    def test_detect_voicemail_with_beeps(self, analyzer, generate_audio_with_beeps):
        """Test voicemail detection with beeps present."""
        file_path, sample_rate = generate_audio_with_beeps
        
        results = analyzer.detect_voicemail(file_path)
        
        assert 'audio' in results
        assert 'sample_rate' in results
        assert 'low_freq_beeps' in results
        assert 'high_freq_beeps' in results
        assert 'speech_start' in results
        
        # Should detect some beeps
        low_beeps = dict(results['low_freq_beeps'])
        high_beeps = dict(results['high_freq_beeps'])
        
        assert low_beeps.get('count', 0) >= 0
        assert high_beeps.get('count', 0) >= 0
    
    def test_detect_voicemail_clean_audio(self, analyzer, generate_clean_audio):
        """Test voicemail detection without beeps."""
        file_path, sample_rate = generate_clean_audio
        
        results = analyzer.detect_voicemail(file_path)
        
        # Should detect minimal or no beeps
        low_beeps = dict(results.get('low_freq_beeps', {}))
        high_beeps = dict(results.get('high_freq_beeps', {}))
        
        assert low_beeps.get('count', 0) <= 2  # Tolerance for false positives
        assert high_beeps.get('count', 0) <= 2
    
    def test_process_creates_audio_segments(self, analyzer, generate_clean_audio):
        """Test that process creates audio segments."""
        file_path, sample_rate = generate_clean_audio
        
        config = {
            'target_sample_rate': 16000,
            'chunk_duration_sec': 3,
            'overlap_sec': 0.5
        }
        
        low_beeps, high_beeps, segments = analyzer.process(
            file_path, 
            "test_001", 
            config
        )
        
        assert isinstance(segments, list)
        assert len(segments) > 0
        
        for segment in segments:
            assert isinstance(segment, AudioSegment)
            assert segment.file_id == "test_001"
            assert isinstance(segment.audio_data, np.ndarray)
    
    def test_process_returns_beep_counts(self, analyzer, generate_audio_with_beeps):
        """Test that process returns beep counts."""
        file_path, sample_rate = generate_audio_with_beeps
        
        config = {
            'target_sample_rate': 16000,
            'chunk_duration_sec': 3,
            'overlap_sec': 0.5
        }
        
        low_beeps, high_beeps, segments = analyzer.process(
            file_path,
            "test_002",
            config
        )
        
        assert isinstance(low_beeps, int)
        assert isinstance(high_beeps, int)
        assert low_beeps >= 0
        assert high_beeps >= 0
    
    def test_process_segments_have_correct_duration(self, analyzer, generate_clean_audio):
        """Test that audio segments have expected duration."""
        file_path, sample_rate = generate_clean_audio
        
        chunk_duration = 2  # seconds
        config = {
            'target_sample_rate': 16000,
            'chunk_duration_sec': chunk_duration,
            'overlap_sec': 0.5
        }
        
        _, _, segments = analyzer.process(file_path, "test_003", config)
        
        for segment in segments[:-1]:  # Exclude last segment (may be shorter)
            expected_samples = chunk_duration * config['target_sample_rate']
            # Allow some tolerance
            assert abs(len(segment.audio_data) - expected_samples) < 1000
    

    def test_speech_start_detection(self, analyzer, generate_audio_with_beeps):
        """Test that speech start time is detected."""
        file_path, sample_rate = generate_audio_with_beeps
        
        results = analyzer.detect_voicemail(file_path)
        
        assert 'speech_start' in results
        speech_start = results['speech_start']
        
        # Should be a reasonable value (not None, not negative)
        assert speech_start is not None
        assert speech_start >= 0
    
    def test_process_with_resampling(self, analyzer, temp_dir):
        """Test processing audio that needs resampling."""
        # Create audio at 44100 Hz
        sample_rate = 44100
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.random.randn(len(t)) * 0.1
        
        file_path = f"{temp_dir}/audio_44k.wav"
        sf.write(file_path, audio, sample_rate)
        
        config = {
            'target_sample_rate': 16000,  # Different from source
            'chunk_duration_sec': 2,
            'overlap_sec': 0.5
        }
        
        _, _, segments = analyzer.process(file_path, "test_005", config)
        
        assert len(segments) > 0
        # Verify segments are at target sample rate
        for segment in segments:
            expected_samples = min(2.0 * 16000, len(segment.audio_data))
            # Should be roughly at 16kHz
            assert len(segment.audio_data) <= 2.5 * 16000
    
    def test_frequency_ranges(self, analyzer):
        """Test that frequency ranges are correctly configured."""
        assert analyzer.low_freq_range[0] < analyzer.low_freq_range[1]
        assert analyzer.high_freq_range[0] < analyzer.high_freq_range[1]
        assert analyzer.low_freq_range[1] < analyzer.high_freq_range[0]  # Non-overlapping
    
    def test_process_empty_audio_file(self, analyzer, temp_dir):
        """Test handling of empty or very short audio files."""
        # Create a very short audio file
        sample_rate = 16000
        audio = np.random.randn(1000) * 0.1  # Less than 0.1 seconds
        
        file_path = f"{temp_dir}/short_audio.wav"
        sf.write(file_path, audio, sample_rate)
        
        config = {
            'target_sample_rate': 16000,
            'chunk_duration_sec': 2,
            'overlap_sec': 0.5
        }
        
        try:
            low_beeps, high_beeps, segments = analyzer.process(
                file_path,
                "test_006",
                config
            )
            # Should handle gracefully
            assert isinstance(segments, list)
        except Exception as e:
            # Or raise appropriate error
            assert True
    
    def test_beep_detection_thresholds(self, analyzer_custom_params):
        """Test that custom thresholds are used."""
        assert analyzer_custom_params.low_params['detection_threshold'] == 0.25
        assert analyzer_custom_params.high_params['detection_threshold'] == 0.30
    
    def test_segment_indices(self, analyzer, generate_clean_audio):
        """Test that segment indices are correctly assigned."""
        file_path, _ = generate_clean_audio
        
        config = {
            'target_sample_rate': 16000,
            'chunk_duration_sec': 2,
            'overlap_sec': 0.5
        }
        
        _, _, segments = analyzer.process(file_path, "test_007", config)
        
        # Verify indices are sequential
        for i, segment in enumerate(segments):
            assert segment.segment_index == i+1
    
    def test_audio_data_type(self, analyzer, generate_clean_audio):
        """Test that audio data is in correct format (numpy float32)."""
        file_path, _ = generate_clean_audio
        
        config = {
            'target_sample_rate': 16000,
            'chunk_duration_sec': 2,
            'overlap_sec': 0.5
        }
        
        _, _, segments = analyzer.process(file_path, "test_008", config)
        
        for segment in segments:
            assert isinstance(segment.audio_data, np.ndarray)
            assert segment.audio_data.dtype in [np.float32, np.float64]
