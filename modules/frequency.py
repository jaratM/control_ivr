import numpy as np
from typing import Tuple, List,Dict,Optional        
from .types import AudioSegment
import librosa
from scipy.signal import find_peaks, butter, sosfilt
from scipy import signal
import torchaudio
import torch
from functools import lru_cache



class FrequencyAnalyzer:
    """
    Dual frequency beep detector using bandpass filtering and envelope detection.
    Detects beeps independently in low and high frequency ranges with optimized filtering.
    Includes intelligent transcription start detection based on beep patterns and silence.
    
    Key Features:
    - Low frequency detection (390-470 Hz): First 60 seconds only for efficiency
    - High frequency detection (590-650 Hz): Entire audio
    - Intelligent filtering: Last 5s window + after last low beep
    - Transcription start detection: Lookahead for next beep + silence detection
    - Adaptive thresholds: 60th percentile baseline with 1.5x multiplier
    - Optimized envelope detection with Hilbert transform
    """
    
    def __init__(
        self,
        low_freq_range: Tuple[float, float] = (390, 470),
        high_freq_range: Tuple[float, float] = (590, 650),
        low_params: Dict = None,
        high_params: Dict = None
        ):
        self.low_freq_range = low_freq_range
        self.high_freq_range = high_freq_range
        
        self.low_params = low_params or {
            'min_beep_interval': 1.0,
            'beep_duration': 1.0,
            'detection_threshold': 0.30,
            'freq_bandwidth': 5
        }
        
        self.high_params = high_params or {
            'min_beep_interval': 0.30,
            'beep_duration': 0.30,
            'detection_threshold': 0.35,
            'freq_bandwidth': 10
        }

    @lru_cache(maxsize=100)
    def get_resampler(self, orig_freq: int, new_freq: int):
        """Cached resampler for efficiency"""
        return torchaudio.transforms.Resample(orig_freq, new_freq)

    def process(self, file_path: str, file_id: str, config: Dict) -> Tuple[int, int, List[AudioSegment]]:
        
        results = self.detect_voicemail(file_path)

        transcription_start = results.get('speech_start')

            
        waveform = results.get('audio')
        sample_rate = results.get('sample_rate')
        
        # Fix: Correctly count items in the dictionary, not keys
        high_beeps_val = results.get('high_freq_beeps')
        high_beeps_data = dict(high_beeps_val) if high_beeps_val else {}
        number_high_beeps = high_beeps_data.get('count', 0)
        
        low_beeps_val = results.get('low_freq_beeps')
        low_beeps_data = dict(low_beeps_val) if low_beeps_val else {}
        number_low_bips = low_beeps_data.get('count', 0)
        
        target_sample_rate = config.get('target_sample_rate', 16000)
        chunk_duration_sec = config.get('chunk_duration_sec', 25)
        overlap_sec = config.get('overlap_sec', 0)
        
        # Convert to Tensor for resampling
        # Handle input type safely
        if isinstance(waveform, np.ndarray):
            waveform_tensor = torch.from_numpy(waveform).float()
        else:
            waveform_tensor = torch.tensor(waveform).float()
            
        if sample_rate != target_sample_rate:
            resampler = self.get_resampler(sample_rate, target_sample_rate)
            waveform_tensor = resampler(waveform_tensor)
            sample_rate = target_sample_rate

        # Fix: Handle 1D vs 2D dimensions
        # librosa mono=True returns (N,). 
        if waveform_tensor.dim() == 2:
             # If (Channels, Time), average to mono
            waveform_tensor = waveform_tensor.mean(dim=0)
        
        total_samples = waveform_tensor.shape[0]
        
        chunk_samples = int(chunk_duration_sec * sample_rate)
        overlap_samples = int(overlap_sec * sample_rate)
        step_samples = chunk_samples - overlap_samples
        
        start = int(transcription_start * sample_rate)
        chunk_idx = 0
        segments = []
        
        while start < total_samples:
            end = min(start + chunk_samples, total_samples)
            
            # Slice 1D tensor
            chunk_tensor = waveform_tensor[start:end]
            
            # Check duration
            duration = (end - start) / sample_rate
            if duration < 0.1:
                start += step_samples
                continue

            # Convert back to numpy for AudioSegment
            chunk_np = chunk_tensor.numpy()

            seg = AudioSegment(
                file_id=file_id,
                segment_index=chunk_idx + 1,
                audio_data=chunk_np,
                duration=duration
            )
            segments.append(seg)
            chunk_idx += 1
            start += step_samples
            
            if end >= total_samples:
                break
            
        return number_high_beeps, number_low_bips, segments 
        
    def detect_voicemail(self, audio_path: str) -> Dict:
        """
        Optimized beep detection workflow:
        1. Detect low frequency beeps (first 60 seconds only)
        2. Detect high frequency beeps (entire audio)
        3. Apply intelligent filtering (last 5s window + after last low beep)
        4. Calculate transcription start with lookahead and silence detection
        """
        # Load audio
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        duration = len(y) / sr
        
        # Step 1: Detect low frequency beeps (first 60 seconds only for efficiency)
        low_beeps = self._detect_low_freq_beeps(y, sr, duration)
        
        # Step 2: Detect high frequency beeps (entire audio)
        high_beeps = self._detect_high_freq_beeps(y, sr, duration)
        
        # Step 3: Apply filtering logic to remove spurious high frequency beeps
        self._apply_beep_filters(low_beeps, high_beeps, duration)
        
        # Initialize results
        results = {
            'audio': y,
            'sample_rate': sr,
            'duration': duration,
            'low_freq_beeps': low_beeps,
            'high_freq_beeps': high_beeps,
            'last_beep_time': None,
            'next_beep_start': None,
            'beep_end_silence': None,
            'speech_start': 0
        }
        
        if not low_beeps['times']:
            return results
        
        # Step 4: Calculate transcription start time
        transcription_info = self._calculate_transcription_start(y, sr, low_beeps, high_beeps, duration)
        
        if transcription_info:
            results['last_beep_time'] = transcription_info['last_beep_time']
            results['next_beep_start'] = transcription_info.get('next_beep_start')
            results['beep_end_silence'] = transcription_info['silence_time']
            results['speech_start'] = transcription_info['speech_start']
            
            # Step 5: Filter high beeps before speech start
            self._filter_high_beeps_before_speech(high_beeps, transcription_info)
        
        return results
    
    def _detect_low_freq_beeps(self, y: np.ndarray, sr: int, duration: float) -> Dict:
        """Detect low frequency beeps in first 60 seconds only (optimized)."""
        first_60s_window = 60.0
        
        if duration > first_60s_window:
            start_sample = int(first_60s_window * sr)
            y_segment = y[:start_sample]
            
            results = self._detect_beeps_in_range(
                y_segment, sr, self.low_freq_range, "Low Freq", self.low_params
            )
            results['analysis_window'] = (0, first_60s_window)
        else:
            results = self._detect_beeps_in_range(
                y, sr, self.low_freq_range, "Low Freq", self.low_params
            )
            results['analysis_window'] = (0, duration)
        
        return results
    
    def _detect_high_freq_beeps(self, y: np.ndarray, sr: int, duration: float) -> Dict:
        """Detect high frequency beeps in entire audio."""
        results = self._detect_beeps_in_range(
            y, sr, self.high_freq_range, "High Freq", self.high_params
        )
        results['analysis_window'] = (0, duration)
        return results
    
    def _apply_beep_filters(self, low_results: Dict, high_results: Dict, duration: float) -> None:
        """Apply intelligent filtering to remove spurious high frequency beeps."""
        last_5s_window = 5.0
        original_high_count = high_results['count']
        
        # Filter 1: Keep only high beeps in last 5 seconds if audio is long
        if duration > last_5s_window:
            filtered_times = [t for t in high_results['times'] 
                            if t > (duration - last_5s_window)]
            
            # If too many beeps removed (>3), clear all (likely false positives)
            if original_high_count - len(filtered_times) > 3:
                filtered_times = []
            
            high_results['times'] = filtered_times
            high_results['count'] = len(filtered_times)
        
        # Filter 2: Keep only high beeps after last low beep
        if low_results['times']:
            last_low_time = low_results['times'][-1]
            filtered_times = [t for t in high_results['times'] if t > last_low_time]
            
            high_results['times'] = filtered_times
            high_results['count'] = len(filtered_times)
    
    def _filter_high_beeps_before_speech(self, high_results: Dict, 
                                        transcription_info: Optional[Dict]) -> None:
        """Filter high beeps occurring before transcription start."""
        if not transcription_info or transcription_info.get('speech_start') is None:
            return
        
        speech_start = transcription_info['speech_start']
        filtered_times = [t for t in high_results['times'] if t >= speech_start]
        high_results['times'] = filtered_times
        high_results['count'] = len(filtered_times)
    
    def _calculate_transcription_start(self, y: np.ndarray, sr: int, 
                                      low_results: Dict, high_results: Dict, 
                                      duration: float) -> Optional[Dict]:
        """Calculate transcription start time based on beep patterns and silence."""
        if not low_results['times']:
            return None
        
        last_beep_time = low_results['times'][-1]
        
        # Look ahead for next beep cycle
        next_beep_start = self._lookahead_next_beep(y, sr, last_beep_time)
        
        if next_beep_start:
            search_start = next_beep_start
        else:
            search_start = last_beep_time
        
        # Find silence after beep
        silence_time = self._find_silence_after_beep(y, sr, search_start)
        
        return {
            'last_beep_time': last_beep_time,
            'next_beep_start': next_beep_start,
            'silence_time': silence_time,
            'speech_start': silence_time
        }

    
    def _detect_beeps_in_range(self, y, sr, freq_range, label, params):
        """Detect beeps within specific frequency range using optimized method."""
        # Pre-filter to frequency range
        filtered = self._bandpass_filter(y, sr, freq_range[0], freq_range[1])
        
        # Find dominant frequency
        dominant_freq = self._find_dominant_frequency(filtered, sr, freq_range)
        
        # Narrow bandpass filter around dominant frequency
        narrow_min = dominant_freq - params['freq_bandwidth']
        narrow_max = dominant_freq + params['freq_bandwidth']
        filtered_narrow = self._bandpass_filter(y, sr, narrow_min, narrow_max)
        
        # Detect peaks using optimized envelope method
        beep_times = self._detect_peaks(filtered_narrow, sr, params)
        
        return {
            'count': len(beep_times),
            'times': beep_times.tolist(),
            'dominant_freq': dominant_freq,
            'range': freq_range,
            'narrow_range': (narrow_min, narrow_max)
        }
    
    def _get_envelope(self, y, sr, freq_range, params):
        """
        Get globally normalized envelope for a frequency band.
        FIXED: Returns consistently normalized envelope.
        """
        filtered = self._bandpass_filter(y, sr, freq_range[0], freq_range[1])
        dominant_freq = self._find_dominant_frequency(filtered, sr, freq_range)
        narrow_min = dominant_freq - params['freq_bandwidth']
        narrow_max = dominant_freq + params['freq_bandwidth']
        
        filtered_narrow = self._bandpass_filter(y, sr, narrow_min, narrow_max)
        
        # Compute envelope
        analytic = signal.hilbert(filtered_narrow)
        envelope = np.abs(analytic)
        
        # Smooth with 50ms window
        win_size = max(1, int(0.05 * sr))
        kernel = np.ones(win_size) / win_size
        envelope_smooth = np.convolve(envelope, kernel, mode='same')
        
        # FIXED: Global normalization only
        max_val = np.max(envelope_smooth)
        if max_val > 0:
            envelope_norm = envelope_smooth / max_val
        else:
            envelope_norm = envelope_smooth
        
        return {
            'envelope': envelope_norm,
            'time_axis': np.arange(len(envelope_norm)) / sr,
            'max_value': max_val,
            'dominant_freq': dominant_freq,
            'narrow_range': (narrow_min, narrow_max)
        }

    
    def _lookahead_next_beep(self, y: np.ndarray, sr: int, last_beep_time: float,
                            expected_interval: float = 5.0) -> Optional[float]:
        """Look ahead for the start of the next beep cycle."""
        # Prepare envelope
        filtered = self._bandpass_filter(y, sr, *self.low_freq_range)
        dominant_freq = self._find_dominant_frequency(filtered, sr, self.low_freq_range)
        
        narrow_min = dominant_freq - self.low_params['freq_bandwidth']
        narrow_max = dominant_freq + self.low_params['freq_bandwidth']
        filtered_narrow = self._bandpass_filter(y, sr, narrow_min, narrow_max)
        
        envelope = self._compute_normalized_envelope(filtered_narrow, sr)
        if envelope is None:
            return None
        
        # Define search window
        audio_dur = len(envelope) / sr
        expected_next = last_beep_time + expected_interval
        search_window = 1.5
        refractory = 0.35
        
        start = max(last_beep_time + refractory, expected_next - search_window / 2)
        end = min(audio_dur, expected_next + search_window / 2)

        if start >= audio_dur:
            return None

        # Enforce refractory: don't consider anything too close to last beep
        start = max(start, last_beep_time + refractory)

        s_idx = int(start * sr)
        e_idx = int(end * sr)
        segment = envelope[s_idx:e_idx]
        if segment.size == 0:
            return None

        # Threshold consistent with your newer detector
        threshold = self.low_params['detection_threshold']

        # Require a TRUE rising edge (no "already above" at segment start)
        crossings = np.where(np.diff((segment > threshold).astype(np.int8)) > 0)[0]
        if crossings.size == 0:
            return None
        rise_idx = crossings[0]

        # Validate with slope and sustain checks
        if not self._validate_beep_onset(segment, rise_idx, sr, threshold):
            return None
        
        return start + rise_idx / sr
    
    def _validate_beep_onset(self, segment: np.ndarray, rise_idx: int, 
                            sr: int, threshold: float) -> bool:
        """Validate beep onset with slope and sustain checks."""
        # Slope check
        win = max(2, int(0.02 * sr))
        pre_i0 = max(0, rise_idx - win)
        post_i1 = min(segment.size, rise_idx + win)
        slope = (np.mean(segment[rise_idx:post_i1]) - 
                np.mean(segment[pre_i0:rise_idx])) / (post_i1 - pre_i0 + 1e-9)
        
        if slope < 0.002:
            return False
        
        # Sustain check
        sustain_n = int(0.10 * sr)
        if rise_idx + sustain_n >= segment.size:
            return False
        
        return np.mean(segment[rise_idx:rise_idx + sustain_n]) >= threshold

    
    def _find_silence_after_beep(self, y: np.ndarray, sr: int, search_start_time: float, 
                                silence_threshold_db: float = -40) -> float:
        """Find first sustained silence in audio after specified time."""
        frame_length, hop_length = 512, 128
        
        rms = librosa.feature.rms(y=y, frame_length=frame_length, 
                                 hop_length=hop_length)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, 
                                       hop_length=hop_length)
        
        silence_threshold = 10 ** (silence_threshold_db / 20)  # -40 dB
        search_frame = int(search_start_time / (hop_length / sr))
        search_frame = max(0, min(search_frame, len(rms) - 1))
        
        # Look for sustained silence (at least 100ms)
        min_silence_frames = max(1, int(0.1 / (hop_length / sr)))
        
        for i in range(search_frame, len(rms) - min_silence_frames):
            window_end = min(i + min_silence_frames, len(rms))
            if np.all(rms[i:window_end] < silence_threshold):
                return times[i]
        
        return times[-1] if len(times) > 0 else search_start_time
    

    
    def _compute_normalized_envelope(self, y: np.ndarray, sr: int) -> Optional[np.ndarray]:
        """Compute normalized envelope using Hilbert transform."""
        analytic = signal.hilbert(y)
        envelope = np.abs(analytic)
        
        # Smooth with 50ms window
        window_size = max(1, int(0.05 * sr))
        kernel = np.ones(window_size) / window_size
        envelope_smooth = np.convolve(envelope, kernel, mode='same')
        
        if np.max(envelope_smooth) > 0:
            return envelope_smooth / np.max(envelope_smooth)
        return None
    
    def _detect_peaks(self, y: np.ndarray, sr: int, params: Dict) -> np.ndarray:
        """Detect peaks using optimized envelope detection with adaptive thresholds."""
        envelope_smooth = self._compute_normalized_envelope(y, sr)
        
        if envelope_smooth is None:
            return np.array([])
        
        # Adaptive threshold using 60th percentile baseline
        baseline = np.percentile(envelope_smooth, 60)
        threshold = max(params['detection_threshold'], baseline * 1.5)
        
        # Find peaks with range-specific parameters
        min_distance = int(params['min_beep_interval'] * sr)
        min_width = int(params['beep_duration'] * 0.5 * sr)
        
        peaks, _ = find_peaks(
            envelope_smooth,
            height=threshold,
            distance=min_distance,
            width=min_width,
            prominence=0.05
        )
        
        return peaks / sr
    
    def _find_dominant_frequency(self, y, sr, freq_range):
        """Find dominant frequency using FFT."""
        fft_vals = np.fft.rfft(y)
        fft_freqs = np.fft.rfftfreq(len(y), 1 / sr)
        magnitude = np.abs(fft_vals)
        
        mask = (fft_freqs >= freq_range[0]) & (fft_freqs <= freq_range[1])
        
        if not np.any(mask):
            return (freq_range[0] + freq_range[1]) / 2
        
        dominant_idx = np.argmax(magnitude[mask])
        return fft_freqs[mask][dominant_idx]
    
    def _bandpass_filter(self, y, sr, freq_min, freq_max):
        """Apply bandpass filter."""
        nyquist = sr / 2
        low = max(0.01, min(freq_min / nyquist, 0.99))
        high = max(0.01, min(freq_max / nyquist, 0.99))
        
        if low >= high:
            return y
        
        sos = butter(6, [low, high], btype='band', output='sos')
        return sosfilt(sos, y)
    