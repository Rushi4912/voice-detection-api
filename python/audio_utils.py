import base64
import io
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from typing import Tuple, Optional
from config import config
from logger import logger

class AudioProcessor:
    """Audio processing utilities for voice detection"""
    
    def __init__(self):
        self.sample_rate = config.SAMPLE_RATE
        self.max_duration = config.MAX_AUDIO_DURATION
        self.min_duration = config.MIN_AUDIO_DURATION
    
    def decode_base64_audio(self, base64_string: str) -> Tuple[np.ndarray, int]:
        """
        Decode base64 audio to numpy array with robust error handling
        Uses temporary files to resolve FFmpeg pipe seeking errors
        Supports multiple formats: MP3, WAV, OGG, M4A, FLAC
        
        Args:
            base64_string: Base64 encoded audio
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        temp_file_path = None
        try:
            # Fix base64 padding if necessary
            base64_string_clean = base64_string.strip()
            padding_needed = len(base64_string_clean) % 4
            if padding_needed:
                base64_string_clean += '=' * (4 - padding_needed)
            
            # Decode base64
            try:
                audio_bytes = base64.b64decode(base64_string_clean, validate=True)
            except Exception as b64_error:
                logger.warning(f"Base64 validation failed, trying without validation: {b64_error}")
                audio_bytes = base64.b64decode(base64_string_clean)
            
            logger.info(f"Decoded audio bytes: {len(audio_bytes)} bytes ({len(audio_bytes)/1024:.2f} KB)")
            
            if len(audio_bytes) < 100:
                raise ValueError(f"Decoded audio too small: {len(audio_bytes)} bytes")

            # Write to a temporary file to solve ffmpeg pipe seeking issues
            import tempfile
            import os
            
            # Create a temp file with a specific suffix if possible, or generic
            # We don't know the format yet, so we write binary
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp_audio') as tmp:
                tmp.write(audio_bytes)
                temp_file_path = tmp.name
            
            logger.debug(f"Wrote audio query to temp file: {temp_file_path}")
            
            # Try multiple audio formats with pydub from the file
            formats_to_try = ['mp3', 'wav', 'ogg', 'm4a', 'flac']
            audio = None
            last_error = None
            
            for fmt in formats_to_try:
                try:
                    audio = AudioSegment.from_file(
                        temp_file_path, 
                        format=fmt,
                        parameters=["-ignore_length", "1"] # Help with truncated files
                    )
                    logger.info(f"Successfully decoded as {fmt.upper()} from temp file")
                    break
                except Exception as fmt_error:
                    last_error = fmt_error
                    continue
            
            # If pydub fails, try librosa directly on the file path
            if audio is None:
                logger.warning("Pydub format detection failed, trying librosa directly on file")
                try:
                    audio_array, sr = librosa.load(
                        temp_file_path, 
                        sr=self.sample_rate, 
                        mono=True
                    )
                    logger.info(f"Librosa file load succeeded: shape={audio_array.shape}, sr={sr}")
                    return audio_array, sr
                except Exception as librosa_error:
                    error_msg = f"Failed to decode audio. Last pydub error: {str(last_error)[:200]}. Librosa error: {str(librosa_error)[:200]}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            
            # Verify we have valid audio object from pydub
            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            # Export to WAV in memory for final librosa loading
            # (librosa is best for converting to numpy array in standard format)
            wav_io = io.BytesIO()
            audio.export(wav_io, format='wav')
            wav_io.seek(0)
            
            audio_array, sr = librosa.load(
                wav_io, 
                sr=self.sample_rate, 
                mono=True
            )
            
            logger.info(f"Audio loaded: shape={audio_array.shape}, sr={sr}, duration={len(audio_array)/sr:.2f}s")
            
            if len(audio_array) == 0:
                raise ValueError("Audio array is empty after decoding")
            
            return audio_array, sr
            
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error decoding audio: {str(e)}")
            raise ValueError(f"Failed to decode audio: {str(e)}")
        finally:
            # Clean up temp file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                    logger.debug(f"Deleted temp file: {temp_file_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to delete temp file {temp_file_path}: {cleanup_error}")
    
    def validate_audio(self, audio: np.ndarray, sr: int) -> bool:
        """
        Validate audio duration and quality
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            True if valid
        """
        duration = len(audio) / sr
        
        if duration < self.min_duration:
            raise ValueError(f"Audio too short: {duration:.2f}s (minimum {self.min_duration}s)")
        
        if duration > self.max_duration:
            raise ValueError(f"Audio too long: {duration:.2f}s (maximum {self.max_duration}s)")
        
        if np.all(audio == 0):
            raise ValueError("Audio contains only silence")
        
        return True
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range"""
        if np.max(np.abs(audio)) > 0:
            return audio / np.max(np.abs(audio))
        return audio
    
    def extract_features(self, audio: np.ndarray, sr: int) -> dict:
        """
        Extract audio features for analysis
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary of features
        """
        try:
            # Mel-frequency cepstral coefficients
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            
            # RMS energy
            rms = librosa.feature.rms(y=audio)[0]
            
            # Pitch
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
            
            features = {
                'mfcc_mean': mfcc_mean.tolist(),
                'mfcc_std': mfcc_std.tolist(),
                'spectral_centroid_mean': float(np.mean(spectral_centroids)),
                'spectral_centroid_std': float(np.std(spectral_centroids)),
                'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
                'spectral_contrast_mean': float(np.mean(spectral_contrast)),
                'zcr_mean': float(np.mean(zcr)),
                'zcr_std': float(np.std(zcr)),
                'chroma_mean': float(np.mean(chroma)),
                'rms_mean': float(np.mean(rms)),
                'rms_std': float(np.std(rms)),
                'pitch_mean': float(np.mean(pitches[pitches > 0])) if np.any(pitches > 0) else 0.0,
                'duration': len(audio) / sr
            }
            
            logger.debug(f"Extracted {len(features)} audio features")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            raise

    def extract_f0(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract Fundamental Frequency (F0) using pYIN
        Returns: f0 array (with NaNs for unvoiced regions)
        """
        try:
            # pYIN is robust but slow. for real-time, we limit duration or use fast settings
            # We'll valid frames only
            fmin = librosa.note_to_hz('C2')
            fmax = librosa.note_to_hz('C7')
            
            f0, _, _ = librosa.pyin(audio, sr=sr, fmin=fmin, fmax=fmax, frame_length=2048)
            return f0
        except Exception as e:
            logger.error(f"Error extracting F0: {e}")
            return np.array([])

    def calculate_pitch_stats(self, audio: np.ndarray, sr: int) -> dict:
        """Calculate pitch statistics including variance and range"""
        f0 = self.extract_f0(audio, sr)
        
        # Remove unvoiced segments (NaNs)
        f0_clean = f0[~np.isnan(f0)]
        
        if len(f0_clean) == 0:
            return {'variance': 0.0, 'mean': 0.0, 'range': 0.0}
            
        return {
            'variance': float(np.std(f0_clean)),
            'mean': float(np.mean(f0_clean)),
            'range': float(np.max(f0_clean) - np.min(f0_clean))
        }

    def calculate_jitter(self, audio: np.ndarray, sr: int) -> float:
        """
        Calculate Jitter (frequency perturbation) - measure of voice roughness
        Higher jitter = more natural/rough (Human)
        Very low jitter = robotic/perfect (AI)
        """
        try:
            f0 = self.extract_f0(audio, sr)
            f0_clean = f0[~np.isnan(f0)]
            
            if len(f0_clean) < 2:
                return 0.0
                
            # Calculate absolute difference between consecutive F0 periods
            # Convert Hz to period length (1/Hz)
            periods = 1.0 / f0_clean
            diffs = np.abs(np.diff(periods))
            avg_diff = np.mean(diffs)
            avg_period = np.mean(periods)
            
            # Jitter as relative percentage
            if avg_period > 0:
                jitter = avg_diff / avg_period
                return float(jitter)
            return 0.0
        except Exception:
            return 0.0

    def calculate_shimmer(self, audio: np.ndarray, sr: int) -> float:
        """
        Calculate Shimmer (amplitude perturbation)
        """
        try:
            # Get frame-to-frame amplitude variations
            hop_length = 512
            rmse = librosa.feature.rms(y=audio, frame_length=2048, hop_length=hop_length)[0]
            
            # Filter non-silent frames
            rmse_valid = rmse[rmse > 0.001]
            
            if len(rmse_valid) < 2:
                return 0.0
                
            diffs = np.abs(np.diff(rmse_valid))
            avg_diff = np.mean(diffs)
            avg_amp = np.mean(rmse_valid)
            
            if avg_amp > 0:
                shimmer = avg_diff / avg_amp
                return float(shimmer)
            return 0.0
        except Exception:
            return 0.0
    
    def calculate_spectral_consistency(self, audio: np.ndarray, sr: int) -> float:
        """Calculate spectral consistency (lower = more AI-like)"""
        spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        return float(np.mean(np.std(spec_contrast, axis=1)))

# Global processor instance
audio_processor = AudioProcessor()