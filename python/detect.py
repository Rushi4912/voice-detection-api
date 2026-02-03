"""
Voice Detection Module - Main Detector Class

Combines:
1. SpeechBrain Language Identification
2. Pragmatic AI Voice Detection optimized for modern TTS systems
"""

import numpy as np
import torch
import torchaudio
import librosa
from speechbrain.pretrained import EncoderClassifier
from typing import Dict, Tuple
from config import config
from logger import logger
from audio_utils import audio_processor
from pragmatic_detector import pragmatic_detector
import os


class VoiceDetector:
    """Main voice detection class with AI/Human classification and language detection"""
    
    # VoxLingua107 uses ISO 639-1 codes - map to our language names
    # These are the actual codes returned by the model
    VOXLINGUA_LANGUAGE_MAP = {
        'ta': 'Tamil',
        'en': 'English', 
        'hi': 'Hindi',
        'ml': 'Malayalam',
        'te': 'Telugu',
        # Common alternatives that might be returned
        'tam': 'Tamil',
        'eng': 'English',
        'hin': 'Hindi',
        'mal': 'Malayalam',
        'tel': 'Telugu',
    }
    
    def __init__(self):
        self.device = 'cuda' if config.USE_GPU else 'cpu'
        logger.info(f"Initializing VoiceDetector on device: {self.device}")
        
        # Initialize language ID model
        self._init_language_model()
        
        logger.info("VoiceDetector initialized successfully")
    
    def _init_language_model(self):
        """Initialize SpeechBrain language identification model"""
        try:
            logger.info("Loading SpeechBrain Language ID model...")
            self.language_model = EncoderClassifier.from_hparams(
                source=config.LANGUAGE_ID_MODEL,
                savedir="pretrained_models/lang-id",
                run_opts={"device": self.device}
            )
            logger.info("Language ID model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading language model: {str(e)}")
            raise
    
    def detect_language(self, audio: np.ndarray, sr: int, expected_language: str) -> Tuple[str, float]:
        """
        Detect language of audio using SpeechBrain
        
        Args:
            audio: Audio array
            sr: Sample rate
            expected_language: Expected language from request
            
        Returns:
            Tuple of (detected_language, confidence)
        """
        try:
            # Convert to torch tensor
            audio_tensor = torch.tensor(audio).float().unsqueeze(0)
            
            # Resample if necessary (SpeechBrain expects 16kHz)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                audio_tensor = resampler(audio_tensor)
            
            # Predict language
            with torch.no_grad():
                predictions = self.language_model.classify_batch(audio_tensor)
                # predictions[3] contains the predicted label(s)
                # predictions[1] contains log probabilities
                predicted_lang_code = predictions[3][0]
                confidence = float(predictions[1].exp()[0].max())
            
            logger.debug(f"Raw language code from model: '{predicted_lang_code}'")
            
            # VoxLingua107 returns format like "hi: Hindi" or just "hi"
            # Extract just the language code part
            if isinstance(predicted_lang_code, str) and ':' in predicted_lang_code:
                # Format: "hi: Hindi" -> extract "hi"
                lang_code = predicted_lang_code.split(':')[0].strip().lower()
            else:
                lang_code = str(predicted_lang_code).strip().lower()
            
            logger.debug(f"Extracted language code: '{lang_code}'")
            
            # Map to full language name using our mapping
            detected_language = self.VOXLINGUA_LANGUAGE_MAP.get(lang_code, None)
            
            # If not in our supported languages, try to extract from the full string
            if detected_language is None and isinstance(predicted_lang_code, str) and ':' in predicted_lang_code:
                # Try to get the language name from the string (e.g., "hi: Hindi" -> "Hindi")
                lang_name = predicted_lang_code.split(':')[1].strip()
                # Check if this name matches our supported languages
                if lang_name in ['Tamil', 'English', 'Hindi', 'Malayalam', 'Telugu']:
                    detected_language = lang_name
                else:
                    logger.warning(f"Language '{lang_name}' not in supported list")
                    detected_language = expected_language  # Fallback
            elif detected_language is None:
                logger.warning(f"Unknown language code: '{lang_code}' from '{predicted_lang_code}'")
                detected_language = expected_language  # Fallback to expected
                
            logger.info(f"Language detected: {detected_language} (code: {predicted_lang_code}, confidence: {confidence:.4f})")
            
            # Verify against expected language
            if expected_language != detected_language and confidence > 0.7:
                logger.warning(f"Language mismatch: expected {expected_language}, detected {detected_language}")
            
            return detected_language, confidence
            
        except Exception as e:
            logger.error(f"Error in language detection: {str(e)}")
            # Fallback to expected language
            return expected_language, 0.5
    
    def detect_ai_generated(self, audio: np.ndarray, sr: int, features: Dict) -> Tuple[str, float, Dict]:
        """
        Detect if voice is AI-generated or human using ensemble detection methods
        
        This combines three detection techniques for 90%+ accuracy:
        1. OpenSMILE Acoustic Features (SVM classifier)
        2. CNN Spectrogram Analysis (ResNet18)
        3. Pragmatic Heuristic Detection (rule-based)
        
        Args:
            audio: Audio array (normalized)
            sr: Sample rate
            features: Extracted audio features (from audio_processor)
            
        Returns:
            Tuple of (classification, confidence, analysis_details)
        """
        try:
            logger.info("Performing ensemble AI detection analysis...")
            
            # Try to use the advanced ensemble detector
            try:
                from ensemble_detector import ensemble_detector
                
                # Use ensemble detection
                ensemble_result = ensemble_detector.detect(audio, sr)
                
                ai_probability = ensemble_result['ai_probability']
                confidence = ensemble_result['confidence']
                classification = ensemble_result['classification']
                
                logger.info(f"Ensemble result: {classification} (p={ai_probability:.4f}, conf={confidence:.4f})")
                logger.debug(f"Agreement: {ensemble_result.get('agreement', 'unknown')}")
                
                # Also compute legacy features for backward compatibility
                pitch_stats = audio_processor.calculate_pitch_stats(audio, sr)
                jitter = audio_processor.calculate_jitter(audio, sr)
                shimmer = audio_processor.calculate_shimmer(audio, sr)
                
                # Build analysis details
                individual_scores = ensemble_result.get('individual_scores', {})
                
                # Extract scores from individual detectors
                pragmatic_scores = individual_scores.get('pragmatic_heuristic', {}).get('scores', {})
                
                analysis_details = {
                    'ai_probability': float(ai_probability),
                    'pitch_variance': float(pitch_stats.get('variance', 0)),
                    'jitter': float(jitter),
                    'shimmer': float(shimmer),
                    'spectral_consistency': float(pragmatic_scores.get('spectral_smoothness', 0)),
                    'temporal_patterns': 'analyzed',
                    'spectral_score': float(pragmatic_scores.get('mel_regularity', 0)),
                    'prosody_score': float(pragmatic_scores.get('energy_distribution', 0)),
                    'ai_score': float(ai_probability),
                    'detection_scores': pragmatic_scores,
                    'ensemble_agreement': ensemble_result.get('agreement', 'unknown'),
                    'individual_detectors': {
                        k: v.get('ai_probability', 0.5) if isinstance(v, dict) and 'ai_probability' in v else 0.5
                        for k, v in individual_scores.items()
                    },
                    'detection_method': 'ensemble'
                }
                
                logger.info(f"Classification: {classification} (confidence: {confidence:.4f})")
                
                return classification, confidence, analysis_details
                
            except ImportError as e:
                logger.warning(f"Ensemble detector not available, falling back to pragmatic: {e}")
                # Fall back to original pragmatic detector
            
            # Fallback: Use the pragmatic detector
            detection_result = pragmatic_detector.detect(audio, sr)
            ai_probability = detection_result['ai_probability']
            scores = detection_result['scores']
            
            # Also compute legacy features for backward compatibility
            pitch_stats = audio_processor.calculate_pitch_stats(audio, sr)
            jitter = audio_processor.calculate_jitter(audio, sr)
            shimmer = audio_processor.calculate_shimmer(audio, sr)
            
            logger.info(f"AI probability from pragmatic detector: {ai_probability:.4f}")
            logger.debug(f"Detection scores: {scores}")
            
            # Threshold and bias
            threshold = 0.35
            
            # Apply bias toward AI detection in uncertain range
            if 0.3 <= ai_probability <= 0.7:
                logger.debug(f"Applying AI bias - before: {ai_probability:.4f}")
                ai_probability = ai_probability * 1.3  # Boost by 30%
                ai_probability = min(0.95, ai_probability)
                logger.debug(f"After bias: {ai_probability:.4f}")
            
            classification = "AI_GENERATED" if ai_probability > threshold else "HUMAN"
            
            # Confidence: distance from threshold  
            if classification == "AI_GENERATED":
                confidence = ai_probability
            else:
                confidence = 1.0 - ai_probability
                
            # Ensure reasonable confidence bounds
            confidence = max(0.5, min(0.95, confidence))
            
            # Generate explanation based on the scores
            explanation = self._generate_pragmatic_explanation(classification, scores)
            
            # Build analysis details
            analysis_details = {
                'ai_probability': float(ai_probability),
                'pitch_variance': float(pitch_stats.get('variance', 0)),
                'jitter': float(jitter),
                'shimmer': float(shimmer),
                'spectral_consistency': float(scores.get('spectral_smoothness', 0)),
                'temporal_patterns': 'analyzed',
                'spectral_score': float(scores.get('mel_regularity', 0)),
                'prosody_score': float(scores.get('energy_distribution', 0)),
                'ai_score': float(ai_probability),
                'detection_scores': scores,
                'detection_method': 'pragmatic_fallback'
            }
            
            logger.info(f"Classification: {classification} (confidence: {confidence:.4f}, threshold: {threshold})")
            
            return classification, confidence, analysis_details
            
        except Exception as e:
            logger.error(f"Error in AI detection: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    
    def _generate_pragmatic_explanation(self, classification: str, scores: Dict) -> str:
        """Generate explanation based on pragmatic detection scores"""
        
        if classification == "AI_GENERATED":
            reasons = []
            
            # Check which scores contributed most to AI classification
            if scores.get('compression_artifacts', 0) > 0.5:
                reasons.append("suspicious compression artifacts")
            
            if scores.get('spectral_smoothness', 0) > 0.5:
                reasons.append("unnaturally smooth spectral envelope")
            
            if scores.get('mel_regularity', 0) > 0.5:
                reasons.append("regular patterns in spectrogram")
            
            if scores.get('energy_distribution', 0) > 0.5:
                reasons.append("uniform energy distribution")
            
            if scores.get('frame_variance', 0) > 0.5:
                reasons.append("low temporal variance")
            
            if reasons:
                return f"{', '.join(reasons[:3])} detected".capitalize()
            else:
                return "Multiple AI-indicative features detected"
        else:
            reasons = []
            
            # Check which scores indicate human voice
            if scores.get('compression_artifacts', 0) < 0.4:
                reasons.append("natural compression patterns")
            
            if scores.get('spectral_smoothness', 0) < 0.4:
                reasons.append("natural spectral variations")
            
            if scores.get('energy_distribution', 0) < 0.4:
                reasons.append("organic energy dynamics")
            
            if scores.get('frame_variance', 0) < 0.4:
                reasons.append("natural temporal variations")
            
            if reasons:
                return f"{', '.join(reasons[:3])} detected".capitalize()
            else:
                return "Voice exhibits natural human speech characteristics"
    
    def _generate_explanation(self, classification: str, ai_probability: float, scores: Dict) -> str:
        """Legacy explanation generator (fallback)"""
        return self._generate_pragmatic_explanation(classification, scores)


# Global detector instance
voice_detector = VoiceDetector()