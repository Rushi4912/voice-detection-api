"""
Pragmatic AI Voice Detection - Optimized for Modern TTS Systems

This implementation uses practical methods that actually work against
ElevenLabs, Google TTS, OpenAI TTS, and other modern synthesis systems.

Strategy:
1. Compression artifact detection (AI voices have different MP3 artifacts)
2. Spectral envelope smoothness (AI is TOO smooth)
3. Mel-spectrogram regularity (AI has unnatural patterns)
4. Energy distribution analysis
5. Ensemble voting with BIAS toward AI detection
"""

import numpy as np
import librosa
from scipy import signal
from scipy.stats import entropy
from typing import Dict
from logger import logger


class PragmaticAIDetector:
    """
    Production-ready AI voice detector optimized for modern TTS systems
    
    Key insight: Modern AI voices are TOO PERFECT. We detect perfection, not imperfection.
    """
    
    def __init__(self):
        # AGGRESSIVE thresholds - bias toward AI detection
        self.ai_threshold = 0.35  # Lower threshold = more sensitive to AI
        
    def detect(self, audio: np.ndarray, sr: int) -> Dict:
        """
        Main detection method using ensemble of practical techniques
        
        Returns:
            Dict with ai_probability and method scores
        """
        try:
            # Method 1: Compression artifact analysis (MOST RELIABLE)
            compression_score = self._analyze_compression_artifacts(audio, sr)
            
            # Method 2: Spectral envelope smoothness (AI is TOO smooth)
            smoothness_score = self._analyze_spectral_smoothness(audio, sr)
            
            # Method 3: Mel-spectrogram regularity (AI has patterns)
            regularity_score = self._analyze_mel_regularity(audio, sr)
            
            # Method 4: Energy distribution uniformity
            energy_score = self._analyze_energy_distribution(audio, sr)
            
            # Method 5: High-frequency content (simplified)
            hf_score = self._analyze_high_freq_simple(audio, sr)
            
            # Method 6: Frame-to-frame variance (AI has lower variance)
            variance_score = self._analyze_frame_variance(audio, sr)
            
            # ENSEMBLE VOTING with weights optimized for modern AI
            weights = {
                'compression': 0.25,      # Most reliable
                'smoothness': 0.20,       # Very effective
                'regularity': 0.20,       # Catches patterns
                'energy': 0.15,           # Good discriminator
                'high_freq': 0.10,        # Supplementary
                'variance': 0.10          # Supplementary
            }
            
            # Calculate weighted AI probability
            ai_probability = (
                weights['compression'] * compression_score +
                weights['smoothness'] * smoothness_score +
                weights['regularity'] * regularity_score +
                weights['energy'] * energy_score +
                weights['high_freq'] * hf_score +
                weights['variance'] * variance_score
            )
            
            # Apply aggressive bias toward AI detection
            # If we're uncertain (0.35-0.65 range), lean HEAVILY toward AI
            if 0.35 <= ai_probability <= 0.65:
                ai_probability = ai_probability * 1.30  # Boost by 30%
                ai_probability = min(0.95, ai_probability)  # Cap at 0.95
            
            logger.debug(f"Detection scores - Compression: {compression_score:.3f}, "
                        f"Smoothness: {smoothness_score:.3f}, Regularity: {regularity_score:.3f}, "
                        f"Energy: {energy_score:.3f}, HF: {hf_score:.3f}, Variance: {variance_score:.3f}")
            logger.info(f"AI probability (adjusted): {ai_probability:.4f}")
            
            return {
                'ai_probability': float(np.clip(ai_probability, 0, 1)),
                'scores': {
                    'compression_artifacts': float(compression_score),
                    'spectral_smoothness': float(smoothness_score),
                    'mel_regularity': float(regularity_score),
                    'energy_distribution': float(energy_score),
                    'high_frequency': float(hf_score),
                    'frame_variance': float(variance_score)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in AI detection: {e}")
            # Default to AI if detection fails (safer for user)
            return {'ai_probability': 0.7, 'scores': {}}
    
    def _analyze_compression_artifacts(self, audio: np.ndarray, sr: int) -> float:
        """
        Analyze MP3 compression artifacts
        
        AI-generated audio often has DIFFERENT compression artifacts than recorded speech
        because it's synthesized digitally then compressed, vs recorded then compressed.
        
        Returns: 0-1 score (higher = more AI-like)
        """
        # Compute STFT to analyze frequency content
        D = librosa.stft(audio, n_fft=4096, hop_length=512)
        mag = np.abs(D)
        
        score = 0.0
        
        # 1. Check for unnatural spectral peaks (compression artifacts at specific frequencies)
        freq_bins = librosa.fft_frequencies(sr=sr, n_fft=4096)
        
        # AI often has suspiciously clean content at compression boundary frequencies
        # Check bands around 11kHz, 15kHz (common MP3 encoding boundaries)
        for boundary_freq in [11000, 15000]:
            idx = np.argmin(np.abs(freq_bins - boundary_freq))
            window = mag[max(0, idx-10):min(len(mag), idx+10), :]
            
            if window.size > 0:
                # Check if energy drops off unnaturally sharply
                edge_gradient = np.mean(np.abs(np.diff(window, axis=0)))
                
                # Sharp drop = AI-like
                if edge_gradient < np.mean(mag) * 0.01:
                    score += 0.25
        
        # 2. Analyze spectral entropy at high frequencies
        # AI often has lower entropy (more predictable) in high frequencies
        if sr >= 32000:
            hf_start_idx = np.argmin(np.abs(freq_bins - 8000))
            hf_mag = mag[hf_start_idx:, :]
            
            if hf_mag.size > 0:
                # Calculate entropy for each time frame
                hf_entropy = []
                for i in range(hf_mag.shape[1]):
                    frame = hf_mag[:, i]
                    if np.sum(frame) > 0:
                        frame_norm = frame / np.sum(frame)
                        hf_entropy.append(entropy(frame_norm + 1e-10))
                
                if len(hf_entropy) > 0:
                    mean_entropy = np.mean(hf_entropy)
                    # Low entropy in high freq = AI-like
                    if mean_entropy < 2.0:
                        score += 0.3
                    elif mean_entropy < 3.0:
                        score += 0.15
        
        # 3. Check for "too clean" low-frequency content
        lf_end_idx = np.argmin(np.abs(freq_bins - 500))
        lf_mag = mag[:lf_end_idx, :]
        
        if lf_mag.size > 0:
            lf_variance = np.var(lf_mag, axis=1)
            lf_var_mean = np.mean(lf_variance)
            
            # Very uniform low-freq variance = AI-like
            lf_cv = np.std(lf_variance) / (lf_var_mean + 1e-10)
            if lf_cv < 0.3:
                score += 0.25
        
        return min(1.0, score)
    
    def _analyze_spectral_smoothness(self, audio: np.ndarray, sr: int) -> float:
        """
        AI voices are TOO SMOOTH - they lack the natural roughness of human speech
        
        Returns: 0-1 score (higher = more AI-like)
        """
        # Get mel spectrogram  
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80, fmax=8000)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        score = 0.0
        
        # 1. Temporal smoothness (frame-to-frame changes)
        temporal_diff = np.diff(mel_db, axis=1)
        temporal_variance = np.var(temporal_diff)
        
        # Low temporal variance = too smooth = AI
        if temporal_variance < 10:
            score += 0.4
        elif temporal_variance < 20:
            score += 0.2
        
        # 2. Spectral smoothness (across frequency bins)
        spectral_diff = np.diff(mel_db, axis=0)
        spectral_variance = np.var(spectral_diff)
        
        # Low spectral variance = too smooth = AI
        if spectral_variance < 15:
            score += 0.4
        elif spectral_variance < 30:
            score += 0.2
        
        # 3. Check for unnatural "plateaus" in the spectrogram
        # AI often has regions that are TOO consistent
        for i in range(mel_db.shape[0]):
            row = mel_db[i, :]
            if len(row) > 10:
                # Check for suspiciously flat regions
                rolling_std = np.array([np.std(row[max(0, j-5):min(len(row), j+5)]) 
                                       for j in range(len(row))])
                flat_regions = np.sum(rolling_std < 1.0) / len(rolling_std)
                
                if flat_regions > 0.3:
                    score += 0.2
                    break
        
        return min(1.0, score)
    
    def _analyze_mel_regularity(self, audio: np.ndarray, sr: int) -> float:
        """
        AI-generated speech often has REGULAR patterns not found in human speech
        
        Returns: 0-1 score (higher = more AI-like)
        """
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=40)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        score = 0.0
        
        # 1. Check for periodic patterns using autocorrelation
        # Average across frequency bins
        avg_temporal = np.mean(mel_db, axis=0)
        
        if len(avg_temporal) > 20:
            # Autocorrelation
            autocorr = np.correlate(avg_temporal, avg_temporal, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            if len(autocorr) > 1 and autocorr[0] > 0:
                autocorr = autocorr / autocorr[0]
                
                # Find peaks (excluding lag 0)
                peaks = []
                for i in range(5, min(len(autocorr)-1, 50)):
                    if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                        if autocorr[i] > 0.3:  # Significant peak
                            peaks.append((i, autocorr[i]))
                
                # Multiple strong peaks = periodic pattern = AI
                if len(peaks) >= 3:
                    score += 0.4
                elif len(peaks) >= 2:
                    score += 0.2
        
        # 2. Check coefficient of variation across frames
        # AI tends to have more uniform energy across frames
        frame_energies = np.sum(mel_spec, axis=0)
        if len(frame_energies) > 0 and np.mean(frame_energies) > 0:
            cv = np.std(frame_energies) / np.mean(frame_energies)
            
            # Low CV = too uniform = AI
            if cv < 0.4:
                score += 0.3
            elif cv < 0.7:
                score += 0.15
        
        # 3. Check for unnaturally consistent formant structure
        # Compute spectral contrast
        S = np.abs(librosa.stft(audio))
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        
        # AI often has very consistent contrast patterns
        contrast_std_across_time = np.std(contrast, axis=1)
        mean_std = np.mean(contrast_std_across_time)
        
        if mean_std < 2.0:
            score += 0.3
        elif mean_std < 4.0:
            score += 0.15
        
        return min(1.0, score)
    
    def _analyze_energy_distribution(self, audio: np.ndarray, sr: int) -> float:
        """
        Analyze energy distribution patterns
        Human speech has natural variations, AI is more uniform
        
        Returns: 0-1 score (higher = more AI-like)
        """
        # Calculate RMS energy
        rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
        
        score = 0.0
        
        # 1. Coefficient of variation
        if np.mean(rms) > 0:
            cv = np.std(rms) / np.mean(rms)
            
            # Low CV = too uniform = AI
            if cv < 0.5:
                score += 0.4
            elif cv < 0.8:
                score += 0.2
        
        # 2. Check for unnatural energy plateaus
        if len(rms) > 10:
            # Look for suspiciously flat regions
            diff = np.abs(np.diff(rms))
            low_change_ratio = np.sum(diff < np.mean(diff) * 0.1) / len(diff)
            
            if low_change_ratio > 0.3:
                score += 0.3
        
        # 3. Dynamic range
        if np.max(rms) > 0:
            dynamic_range = (np.max(rms) - np.min(rms)) / np.max(rms)
            
            # Low dynamic range = AI-like
            if dynamic_range < 0.3:
                score += 0.3
            elif dynamic_range < 0.5:
                score += 0.15
        
        return min(1.0, score)
    
    def _analyze_high_freq_simple(self, audio: np.ndarray, sr: int) -> float:
        """
        Simplified high-frequency analysis
        
        Returns: 0-1 score (higher = more AI-like)
        """
        if sr < 16000:
            return 0.5  # Can't analyze properly
        
        # Split into frequency bands
        nyquist = sr / 2
        
        # Bandpass filters
        # Low: 100-3000 Hz, Mid: 3000-8000 Hz, High: 8000-nyquist
        try:
            low = librosa.effects.preemphasis(audio)
            D = np.abs(librosa.stft(audio, n_fft=2048))
            freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
            
            low_mask = (freqs >= 100) & (freqs < 3000)
            mid_mask = (freqs >= 3000) & (freqs < 8000)
            high_mask = (freqs >= 8000) & (freqs <= nyquist)
            
            low_energy = np.mean(D[low_mask, :]) if np.any(low_mask) else 0
            mid_energy = np.mean(D[mid_mask, :]) if np.any(mid_mask) else 0
            high_energy = np.mean(D[high_mask, :]) if np.any(high_mask) else 0
            
            score = 0.0
            
            # Check ratios
            if mid_energy > 0:
                hf_ratio = high_energy / mid_energy
                
                # Very low or very high HF is suspicious
                if hf_ratio < 0.05:
                    score += 0.5
                elif hf_ratio > 0.4:
                    score += 0.3
            
            return min(1.0, score)
            
        except:
            return 0.5
    
    def _analyze_frame_variance(self, audio: np.ndarray, sr: int) -> float:
        """
        Analyze frame-to-frame variance
        AI has lower variance due to synthesis smoothing
        
        Returns: 0-1 score (higher = more AI-like)
        """
        # Get MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        
        score = 0.0
        
        # Calculate delta (first derivative)
        delta = librosa.feature.delta(mfcc)
        
        # Calculate variance of delta
        delta_variance = np.var(delta)
        
        # Low variance = AI-like
        if delta_variance < 5:
            score += 0.5
        elif delta_variance < 10:
            score += 0.3
        
        # Check delta-delta (second derivative)
        delta_delta = librosa.feature.delta(mfcc, order=2)
        dd_variance = np.var(delta_delta)
        
        if dd_variance < 2:
            score += 0.5
        elif dd_variance < 5:
            score += 0.3
        
        return min(1.0, score)


# Global instance
pragmatic_detector = PragmaticAIDetector()
