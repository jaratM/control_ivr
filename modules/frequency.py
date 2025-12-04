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
    Detects beeps in two separate frequency ranges independently.
    Uses spectral analysis within each range to find dominant frequency.
    Now supports different parameters for each frequency range.
    High frequency beeps are only detected after the last low frequency beep.
    Intelligently detects when ringing pattern ends for better transcription timing.
    """

    
    def __init__(
        self,
        low_freq_range: Tuple[float, float] = (390, 460),
        high_freq_range: Tuple[float, float] = (590, 650),
        low_params: Dict = None,
        high_params: Dict = None
        ):
        self.low_freq_range = low_freq_range
        self.high_freq_range = high_freq_range
        
        self.low_params = low_params or {
            'min_beep_interval': 1.0,
            'beep_duration': 1.0,
            'detection_threshold': 0.50,
            'freq_bandwidth': 5
        }
        
        self.high_params = high_params or {
            'min_beep_interval': 0.90,
            'beep_duration': 0.50,
            'detection_threshold': 0.50,
            'freq_bandwidth': 5
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
        number_bips = high_beeps_data.get('count', 0)
        
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
            
            # Convert back to numpy for AudioSegment
            chunk_np = chunk_tensor.numpy()

            seg = AudioSegment(
                file_id=file_id,
                segment_index=chunk_idx + 1,
                audio_data=chunk_np,
                duration=(end - start) / sample_rate
            )
            segments.append(seg)
            chunk_idx += 1
            start += step_samples
            
            if end >= total_samples:
                break
            
        return number_bips, number_low_bips, segments 
        
    def detect_voicemail(self, audio_path: str) -> Dict:
        """
        Main workflow:
        1. Detect ALL low beeps (full audio)
        2. Detect ALL high beeps (full audio)
        3. Filter high beeps: keep only those AFTER last low beep
        4. Find next beep start (lookahead)
        5. Find silence after beep
        6. Detect speech start
        """
        # Load audio
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        duration = len(y) / sr
        
        
        # Step 1: Detect ALL low frequency beeps (no filtering)
        low_beeps = self._detect_beeps_in_range(
            y, sr, self.low_freq_range, "Low Freq", self.low_params
        )
            
        # Step 2: Detect ALL high frequency beeps (no filtering yet)
        high_beeps = self._detect_beeps_in_range(
            y, sr, self.high_freq_range, "High Freq", self.high_params
        )
        
        # Step 3: Filter high beeps based on last low beep
        if len(low_beeps['times']) > 0:
            # Get last low beep time
            last_low_time = low_beeps['times'][-1]
            
            # Keep only high beeps AFTER last low beep
            filtered_high_times = [t for t in high_beeps['times'] if t > last_low_time]
            
            # Update high beeps results
            high_beeps['times'] = filtered_high_times
            high_beeps['count'] = len(filtered_high_times)
            
            
        else:
            high_beeps['times'] = []
            high_beeps['count'] = 0
        
        
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
        
        if len(low_beeps['times']) == 0:
            return results
        
        # Step 4: Get last beep time
        last_beep_time = low_beeps['times'][-1]
        results['last_beep_time'] = last_beep_time
        
        # Get envelope for lookahead
        envelope_data = self._get_envelope(y, sr, self.low_freq_range, self.low_params)
        
        # Step 5: Look ahead for next beep start
        next_beep_start = self._lookahead_next_beep(
            sr, last_beep_time, envelope_data, expected_interval=5.0
        )
        
        if next_beep_start is not None:
            results['next_beep_start'] = next_beep_start
            search_start_time = next_beep_start
        else:
            search_start_time = last_beep_time
        
        # Step 6: Find silence in ORIGINAL signal after beep end
        silence_start = self._find_silence_after_beep(y, sr, search_start_time)
        results['beep_end_silence'] = silence_start
        
        # Step 7: Detect speech start in original signal
        speech_start = silence_start
        results['speech_start'] = speech_start

        
        return results

    
    def _detect_beeps_in_range(self, y, sr, freq_range, label, params):
        """
        Detect beeps in a frequency range.
        NO FILTERING - detects all beeps in the entire audio.
        """
        # Bandpass filter to frequency range
        filtered = self._bandpass_filter(y, sr, freq_range[0], freq_range[1])
        
        # Find dominant frequency
        dominant_freq = self._find_dominant_frequency(filtered, sr, freq_range)
        # print(f"  {label} ({freq_range[0]}-{freq_range[1]} Hz): Dominant = {dominant_freq:.1f} Hz")
        
        # Narrow filter around dominant frequency
        narrow_min = dominant_freq - params['freq_bandwidth']
        narrow_max = dominant_freq + params['freq_bandwidth']
        filtered_narrow = self._bandpass_filter(y, sr, narrow_min, narrow_max)
        
        # Detect peaks using corrected method
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

    
    def _lookahead_next_beep(
        self,
        sr: int,
        last_beep_time: float,
        envelope_data: Dict,
        expected_interval: float = 4.5,
        search_window: float = 1.5,
        refractory: float = 0.35,           # ignore this much time after last beep
        sustain_sec: float = 0.10,          # energy must stay high for this long
        min_slope: float = 0.002,           # minimal rise per sample (norm units)
        threshold_mult: float = 1.0         # × detection_threshold (aligns with new code)
    ) -> Optional[float]:
        env = envelope_data['envelope'] if 'envelope' in envelope_data else envelope_data['envelope_smooth']
        audio_len = len(env)
        audio_dur = audio_len / sr

        # Compute window centered on expected next beep
        expected_next = last_beep_time + expected_interval
        start = max(0.0, expected_next - search_window / 2)
        end = min(audio_dur, expected_next + search_window / 2)

        if start >= audio_dur:
            return None

        # Enforce refractory: don’t consider anything too close to last beep
        start = max(start, last_beep_time + refractory)

        s_idx = int(start * sr)
        e_idx = int(end * sr)
        seg = env[s_idx:e_idx]
        if seg.size == 0:
            return None

        # Threshold consistent with your newer detector
        det_th = self.low_params['detection_threshold'] * threshold_mult

        # Require a TRUE rising edge (no "already above" at segment start)
        above = seg > det_th
        crossings = np.where(np.diff(above.astype(np.int8)) > 0)[0]
        if crossings.size == 0:
            return None
        rise_i = crossings[0]

        # Slope check: compare short window before/after the rise
        win = max(2, int(0.02 * sr))  # 20 ms
        pre_i0 = max(0, rise_i - win)
        post_i1 = min(seg.size, rise_i + win)
        slope = (np.mean(seg[rise_i:post_i1]) - np.mean(seg[pre_i0:rise_i])) / (post_i1 - pre_i0 + 1e-9)
        if slope < min_slope:
            return None

        # Sustain check: must remain above threshold for sustain_sec
        sustain_n = int(sustain_sec * sr)
        if rise_i + sustain_n >= seg.size:
            return None
        if np.mean(seg[rise_i:rise_i + sustain_n]) < det_th:
            return None
        
        next_beep_time = start + rise_i / sr

        return next_beep_time

    
    def _find_silence_after_beep(self, y, sr, search_start_time, silence_threshold_db=-40):
        """
        Find first sustained silence in ORIGINAL signal AFTER beep.
        Searches FORWARD from search_start_time to find where silence begins.
        """
        # Convert to RMS for silence detection
        frame_length = 512
        hop_length = 128
        
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        
        # Convert dB threshold to linear
        silence_threshold_linear = 10 ** (silence_threshold_db / 20)
        
        # Start searching from search_start_time (FORWARD)
        search_frame = int(search_start_time / (hop_length / sr))
        search_frame = max(0, min(search_frame, len(rms) - 1))
        
        # Look forward for sustained silence (at least 100ms quiet)
        min_silence_frames = int(0.1 / (hop_length / sr))
        min_silence_frames = max(1, min_silence_frames)
        
        # Search from current frame to end of audio
        for i in range(search_frame, len(rms) - min_silence_frames):
            # Check if this frame and following frames are ALL silent
            window_end = min(i + min_silence_frames, len(rms))
            if np.all(rms[i:window_end] < silence_threshold_linear):
                return times[i]
        
        # If no silence found, return the end of audio time
        return times[-1] if len(times) > 0 else search_start_time
    

    
    def _detect_peaks(self, y, sr, params):
        """
        Detect peaks in filtered signal.
        FIXED: Now uses global normalization and consistent thresholds.
        """
        # Envelope detection
        analytic = signal.hilbert(y)
        envelope = np.abs(analytic)
        
        # Smooth with 50ms window (consistent with BeepDetector)
        win_size = max(1, int(0.05 * sr))
        kernel = np.ones(win_size) / win_size
        envelope_smooth = np.convolve(envelope, kernel, mode='same')
        
        # FIXED: Global normalization (done once, not per segment)
        if np.max(envelope_smooth) > 0:
            envelope_smooth = envelope_smooth / np.max(envelope_smooth)
        else:
            return np.array([])
        
        # FIXED: Use 60th percentile baseline (consistent with BeepDetector)
        baseline = np.percentile(envelope_smooth, 60)
        
        # FIXED: Use 1.5× multiplier (consistent with BeepDetector)
        threshold = max(params['detection_threshold'], baseline * 1.5)
        
        # Find peaks
        min_distance = int(params['min_beep_interval'] * sr)
        min_width = int(params['beep_duration'] * 0.5 * sr)
        
        # FIXED: Use 0.05 prominence (consistent with BeepDetector)
        peaks, _ = find_peaks(
            envelope_smooth,
            height=threshold,
            distance=min_distance,
            width=min_width,
            prominence=0.05  # Changed from 0.15
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
    