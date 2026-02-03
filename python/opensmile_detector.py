"""
OpenSMILE Acoustic Feature-Based AI Voice Detector

This module uses OpenSMILE to extract professional-grade acoustic features
and an SVM classifier trained on ASVspoof/WaveFake datasets.

Key features extracted (eGeMAPSv02):
- Prosody: F0, jitter, shimmer, loudness
- Voice Quality: HNR, formants F1-F3
- Spectral: MFCCs, spectral slope, alpha ratio

References:
- eGeMAPSv02: Extended Geneva Minimalistic Acoustic Parameter Set
- ASVspoof 2019/2021: Audio deepfake detection benchmark
"""

import numpy as np
import os
import pickle
from typing import Dict, Tuple, Optional
from pathlib import Path

try:
    import opensmile
    OPENSMILE_AVAILABLE = True
except ImportError:
    OPENSMILE_AVAILABLE = False
    print("Warning: opensmile not installed. Using fallback librosa features.")

import librosa
from scipy.stats import skew, kurtosis
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

from logger import logger


class OpenSMILEDetector:
    """
    AI Voice Detector using OpenSMILE acoustic features + ML classifier
    
    This is one of the three detectors in our ensemble:
    1. OpenSMILE Acoustic (this) - prosody, voice quality, spectral
    2. CNN Spectrogram - visual patterns in mel spectrograms
    3. Pragmatic Detector - heuristic-based methods
    """
    
    # Model paths
    MODEL_DIR = Path(__file__).parent / "models"
    CLASSIFIER_PATH = MODEL_DIR / "acoustic_classifier.pkl"
    SCALER_PATH = MODEL_DIR / "acoustic_scaler.pkl"
    
    def __init__(self):
        self.use_opensmile = OPENSMILE_AVAILABLE
        self.smile = None
        self.classifier = None
        self.scaler = None
        self.is_trained = False
        
        if self.use_opensmile:
            self._init_opensmile()
        
        self._load_or_create_classifier()
        logger.info(f"OpenSMILEDetector initialized. OpenSMILE available: {self.use_opensmile}")
    
    def _init_opensmile(self):
        """Initialize OpenSMILE with eGeMAPSv02 feature set"""
        try:
            # eGeMAPSv02 extracts 88 features optimized for voice analysis
            self.smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.Functionals
            )
            logger.info("OpenSMILE initialized with eGeMAPSv02 feature set (88 features)")
        except Exception as e:
            logger.error(f"Failed to initialize OpenSMILE: {e}")
            self.use_opensmile = False
    
    def _load_or_create_classifier(self):
        """Load pretrained classifier or create a new one"""
        try:
            if self.CLASSIFIER_PATH.exists() and self.SCALER_PATH.exists():
                self.classifier = joblib.load(self.CLASSIFIER_PATH)
                self.scaler = joblib.load(self.SCALER_PATH)
                self.is_trained = True
                logger.info("Loaded pretrained acoustic classifier")
            else:
                # Create new classifier (will need training)
                self.classifier = SVC(
                    kernel='rbf',
                    C=10.0,
                    gamma='scale',
                    probability=True,
                    class_weight='balanced'
                )
                self.scaler = StandardScaler()
                self.is_trained = False
                logger.info("Created new acoustic classifier (needs training)")
        except Exception as e:
            logger.error(f"Error loading classifier: {e}")
            self.classifier = SVC(kernel='rbf', probability=True, class_weight='balanced')
            self.scaler = StandardScaler()
            self.is_trained = False
    
    def extract_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract acoustic features from audio
        
        Uses OpenSMILE if available, otherwise falls back to librosa-based features.
        
        Args:
            audio: Audio waveform (numpy array)
            sr: Sample rate
            
        Returns:
            Feature vector (numpy array)
        """
        if self.use_opensmile and self.smile is not None:
            return self._extract_opensmile_features(audio, sr)
        else:
            return self._extract_librosa_features(audio, sr)
    
    def _extract_opensmile_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract features using OpenSMILE eGeMAPSv02"""
        try:
            # OpenSMILE expects float32 audio
            audio_float = audio.astype(np.float32)
            
            # Process audio - returns DataFrame with 88 features
            features_df = self.smile.process_signal(audio_float, sr)
            
            # Convert to numpy array
            features = features_df.values.flatten()
            
            logger.debug(f"Extracted {len(features)} OpenSMILE features")
            return features
            
        except Exception as e:
            logger.error(f"OpenSMILE extraction failed: {e}")
            # Fallback to librosa
            return self._extract_librosa_features(audio, sr)
    
    def _extract_librosa_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Fallback feature extraction using librosa
        
        Extracts features similar to eGeMAPSv02:
        - F0 statistics (mean, std, min, max, range)
        - Jitter and shimmer
        - HNR (Harmonics-to-Noise Ratio)
        - Formants F1, F2, F3 statistics
        - MFCCs (13 coefficients + deltas)
        - Spectral features (centroid, rolloff, flux)
        - Energy features
        """
        features = []
        
        try:
            # 1. F0 (Pitch) features
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, fmin=50, fmax=500, sr=sr
            )
            f0_valid = f0[~np.isnan(f0)]
            
            if len(f0_valid) > 0:
                features.extend([
                    np.mean(f0_valid),
                    np.std(f0_valid),
                    np.min(f0_valid),
                    np.max(f0_valid),
                    np.max(f0_valid) - np.min(f0_valid),  # F0 range
                    np.percentile(f0_valid, 25),
                    np.percentile(f0_valid, 75),
                    skew(f0_valid),
                    kurtosis(f0_valid)
                ])
            else:
                features.extend([0] * 9)
            
            # 2. Jitter (F0 perturbation)
            if len(f0_valid) > 1:
                f0_diff = np.abs(np.diff(f0_valid))
                jitter = np.mean(f0_diff) / (np.mean(f0_valid) + 1e-10)
                jitter_ppq = np.mean(np.abs(np.diff(f0_diff))) / (np.mean(f0_valid) + 1e-10)
                features.extend([jitter, jitter_ppq])
            else:
                features.extend([0, 0])
            
            # 3. Shimmer (amplitude perturbation)
            rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
            if len(rms) > 1:
                rms_diff = np.abs(np.diff(rms))
                shimmer = np.mean(rms_diff) / (np.mean(rms) + 1e-10)
                shimmer_dda = np.mean(np.abs(np.diff(rms_diff))) / (np.mean(rms) + 1e-10)
                features.extend([shimmer, shimmer_dda])
            else:
                features.extend([0, 0])
            
            # 4. HNR approximation using spectral features
            S = np.abs(librosa.stft(audio))
            S_harmonic, S_percussive = librosa.decompose.hpss(S)
            hnr = np.mean(S_harmonic) / (np.mean(S_percussive) + 1e-10)
            features.append(np.clip(hnr, 0, 100))
            
            # 5. MFCC features (13 coefficients)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            features.extend(mfcc_mean)
            features.extend(mfcc_std)
            
            # 6. Delta MFCC
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
            features.extend(mfcc_delta_mean)
            
            # 7. Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
            spectral_flatness = librosa.feature.spectral_flatness(y=audio)[0]
            
            features.extend([
                np.mean(spectral_centroid),
                np.std(spectral_centroid),
                np.mean(spectral_rolloff),
                np.std(spectral_rolloff),
                np.mean(spectral_bandwidth),
                np.mean(spectral_flatness)
            ])
            
            # 8. Energy features
            features.extend([
                np.mean(rms),
                np.std(rms),
                np.max(rms) - np.min(rms),  # Dynamic range
                skew(rms),
                kurtosis(rms)
            ])
            
            # 9. Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            features.extend([np.mean(zcr), np.std(zcr)])
            
            # 10. Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            features.extend(np.mean(contrast, axis=1))
            
            logger.debug(f"Extracted {len(features)} librosa features (fallback)")
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            # Return zeros if extraction fails
            features = [0] * 88  # Match OpenSMILE feature count
        
        return np.array(features)
    
    def detect(self, audio: np.ndarray, sr: int) -> Dict:
        """
        Detect AI-generated voice using acoustic features
        
        Args:
            audio: Audio waveform
            sr: Sample rate
            
        Returns:
            Dict with:
            - ai_probability: float (0-1)
            - confidence: float (0-1)
            - method: str
            - features_extracted: int
        """
        try:
            # Extract features
            features = self.extract_features(audio, sr)
            
            if not self.is_trained:
                # If not trained, use heuristic-based scoring
                return self._heuristic_detection(features, audio, sr)
            
            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Predict
            ai_probability = self.classifier.predict_proba(features_scaled)[0][1]
            
            # Confidence based on distance from decision boundary
            confidence = abs(ai_probability - 0.5) * 2  # 0-1 scale
            confidence = min(0.95, max(0.5, confidence))
            
            return {
                'ai_probability': float(ai_probability),
                'confidence': float(confidence),
                'method': 'opensmile_svm' if self.use_opensmile else 'librosa_svm',
                'features_extracted': len(features)
            }
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return {
                'ai_probability': 0.5,
                'confidence': 0.3,
                'method': 'fallback',
                'error': str(e)
            }
    
    def _heuristic_detection(self, features: np.ndarray, audio: np.ndarray, sr: int) -> Dict:
        """
        Heuristic-based detection when classifier is not trained
        Uses acoustic analysis rules from voice pathology research
        """
        scores = []
        
        # Rule 1: Jitter analysis (AI typically has lower jitter)
        # Extract jitter from features (index depends on feature order)
        f0, _, _ = librosa.pyin(audio, fmin=50, fmax=500, sr=sr)
        f0_valid = f0[~np.isnan(f0)]
        
        if len(f0_valid) > 1:
            f0_diff = np.abs(np.diff(f0_valid))
            jitter = np.mean(f0_diff) / (np.mean(f0_valid) + 1e-10)
            
            # Low jitter (< 0.5%) suggests AI
            if jitter < 0.005:
                scores.append(0.7)
            elif jitter < 0.01:
                scores.append(0.5)
            else:
                scores.append(0.3)
        else:
            scores.append(0.5)
        
        # Rule 2: Shimmer analysis (AI typically has lower shimmer)
        rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
        if len(rms) > 1:
            shimmer = np.mean(np.abs(np.diff(rms))) / (np.mean(rms) + 1e-10)
            
            # Low shimmer suggests AI
            if shimmer < 0.02:
                scores.append(0.7)
            elif shimmer < 0.05:
                scores.append(0.5)
            else:
                scores.append(0.3)
        else:
            scores.append(0.5)
        
        # Rule 3: F0 variability (AI often has less natural F0 variation)
        if len(f0_valid) > 0:
            f0_cv = np.std(f0_valid) / (np.mean(f0_valid) + 1e-10)
            
            if f0_cv < 0.1:
                scores.append(0.7)  # Too consistent = AI
            elif f0_cv > 0.4:
                scores.append(0.3)  # High variation = Human
            else:
                scores.append(0.5)
        else:
            scores.append(0.5)
        
        # Rule 4: HNR (Harmonics-to-Noise Ratio)
        # AI tends to have unnaturally high HNR
        S = np.abs(librosa.stft(audio))
        S_harmonic, S_percussive = librosa.decompose.hpss(S)
        hnr = np.mean(S_harmonic) / (np.mean(S_percussive) + 1e-10)
        
        if hnr > 10:
            scores.append(0.7)  # Too clean = AI
        elif hnr < 3:
            scores.append(0.3)  # Natural noise = Human
        else:
            scores.append(0.5)
        
        # Combine scores
        ai_probability = np.mean(scores)
        confidence = 0.5 + abs(ai_probability - 0.5) * 0.4
        
        return {
            'ai_probability': float(ai_probability),
            'confidence': float(confidence),
            'method': 'acoustic_heuristic',
            'features_extracted': len(features),
            'note': 'Using heuristic mode - classifier not trained'
        }
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Train the acoustic classifier
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (0=Human, 1=AI)
            
        Returns:
            Training metrics dict
        """
        try:
            logger.info(f"Training acoustic classifier on {len(y)} samples...")
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train SVM
            self.classifier.fit(X_scaled, y)
            
            # Calculate training accuracy
            train_accuracy = self.classifier.score(X_scaled, y)
            
            # Save models
            self.MODEL_DIR.mkdir(exist_ok=True)
            joblib.dump(self.classifier, self.CLASSIFIER_PATH)
            joblib.dump(self.scaler, self.SCALER_PATH)
            
            self.is_trained = True
            
            logger.info(f"Training complete. Accuracy: {train_accuracy:.4f}")
            
            return {
                'success': True,
                'train_accuracy': float(train_accuracy),
                'n_samples': len(y),
                'n_features': X.shape[1]
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def save_model(self, path: Optional[Path] = None):
        """Save classifier and scaler to disk"""
        if path is None:
            path = self.MODEL_DIR
        
        path.mkdir(exist_ok=True)
        joblib.dump(self.classifier, path / "acoustic_classifier.pkl")
        joblib.dump(self.scaler, path / "acoustic_scaler.pkl")
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Optional[Path] = None):
        """Load classifier and scaler from disk"""
        if path is None:
            path = self.MODEL_DIR
        
        self.classifier = joblib.load(path / "acoustic_classifier.pkl")
        self.scaler = joblib.load(path / "acoustic_scaler.pkl")
        self.is_trained = True
        logger.info(f"Model loaded from {path}")


# Global instance
opensmile_detector = OpenSMILEDetector()
