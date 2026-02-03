"""
Advanced Ensemble AI Voice Detector

This module combines three detection methods for robust AI voice detection:
1. OpenSMILE Acoustic Features (SVM classifier)
2. CNN Spectrogram Analysis (ResNet18)
3. Pragmatic Heuristic Detection (rule-based)

Ensemble Strategy:
- Weighted voting with learned weights
- Agreement analysis for confidence boosting
- Calibrated probability outputs
- Uncertainty quantification

Target: 90%+ accuracy on AI voice detection
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json

from logger import logger


class EnsembleDetector:
    """
    Advanced ensemble detector combining multiple AI voice detection methods
    
    Key Features:
    - Weighted voting with optimized weights
    - Agreement-based confidence boosting
    - Calibrated probability outputs
    - Fallback handling when individual detectors fail
    """
    
    # Default ensemble weights (can be tuned via calibration)
    DEFAULT_WEIGHTS = {
        'cnn_spectrogram': 0.45,      # Most accurate when trained
        'opensmile_acoustic': 0.35,   # Robust to compression artifacts
        'pragmatic_heuristic': 0.20   # Fast fallback
    }
    
    # Weights file path
    WEIGHTS_PATH = Path(__file__).parent / "models" / "ensemble_weights.json"
    
    def __init__(self):
        """Initialize ensemble detector with all component detectors"""
        self.weights = self.DEFAULT_WEIGHTS.copy()
        self.detectors_initialized = False
        
        # Lazy loading of detectors
        self.opensmile_detector = None
        self.cnn_detector = None
        self.pragmatic_detector = None
        
        # Load calibrated weights if available
        self._load_weights()
        
        logger.info("EnsembleDetector initialized")
    
    def _init_detectors(self):
        """Initialize all component detectors (lazy loading)"""
        if self.detectors_initialized:
            return
        
        try:
            # Import and initialize detectors
            from opensmile_detector import opensmile_detector
            self.opensmile_detector = opensmile_detector
            logger.info("OpenSMILE detector loaded")
        except Exception as e:
            logger.warning(f"Failed to load OpenSMILE detector: {e}")
            self.opensmile_detector = None
        
        try:
            from cnn_detector import cnn_detector
            self.cnn_detector = cnn_detector
            logger.info("CNN detector loaded")
        except Exception as e:
            logger.warning(f"Failed to load CNN detector: {e}")
            self.cnn_detector = None
        
        try:
            from pragmatic_detector import pragmatic_detector
            self.pragmatic_detector = pragmatic_detector
            logger.info("Pragmatic detector loaded")
        except Exception as e:
            logger.warning(f"Failed to load Pragmatic detector: {e}")
            self.pragmatic_detector = None
        
        self.detectors_initialized = True
    
    def _load_weights(self):
        """Load calibrated weights from file if available"""
        try:
            if self.WEIGHTS_PATH.exists():
                with open(self.WEIGHTS_PATH, 'r') as f:
                    saved_weights = json.load(f)
                self.weights.update(saved_weights)
                logger.info(f"Loaded calibrated weights: {self.weights}")
        except Exception as e:
            logger.warning(f"Could not load weights: {e}")
    
    def _save_weights(self):
        """Save calibrated weights to file"""
        try:
            self.WEIGHTS_PATH.parent.mkdir(exist_ok=True)
            with open(self.WEIGHTS_PATH, 'w') as f:
                json.dump(self.weights, f, indent=2)
            logger.info(f"Saved weights to {self.WEIGHTS_PATH}")
        except Exception as e:
            logger.error(f"Could not save weights: {e}")
    
    def detect(self, audio: np.ndarray, sr: int) -> Dict:
        """
        Perform ensemble AI voice detection
        
        Args:
            audio: Audio waveform (numpy array)
            sr: Sample rate
            
        Returns:
            Dict with:
            - classification: 'AI_GENERATED' or 'HUMAN'
            - ai_probability: float (0-1)
            - confidence: float (0-1)
            - individual_scores: Dict of each detector's results
            - agreement: str ('unanimous', 'majority', 'split')
            - explanation: str
        """
        # Initialize detectors if not already done
        self._init_detectors()
        
        individual_results = {}
        predictions = []
        
        # 1. OpenSMILE Acoustic Detection
        if self.opensmile_detector is not None:
            try:
                result = self.opensmile_detector.detect(audio, sr)
                individual_results['opensmile_acoustic'] = result
                predictions.append(('opensmile_acoustic', result['ai_probability']))
                logger.debug(f"OpenSMILE: {result['ai_probability']:.3f}")
            except Exception as e:
                logger.error(f"OpenSMILE detection failed: {e}")
                individual_results['opensmile_acoustic'] = {'error': str(e)}
        
        # 2. CNN Spectrogram Detection
        if self.cnn_detector is not None:
            try:
                result = self.cnn_detector.detect(audio, sr)
                individual_results['cnn_spectrogram'] = result
                predictions.append(('cnn_spectrogram', result['ai_probability']))
                logger.debug(f"CNN: {result['ai_probability']:.3f}")
            except Exception as e:
                logger.error(f"CNN detection failed: {e}")
                individual_results['cnn_spectrogram'] = {'error': str(e)}
        
        # 3. Pragmatic Heuristic Detection
        if self.pragmatic_detector is not None:
            try:
                result = self.pragmatic_detector.detect(audio, sr)
                individual_results['pragmatic_heuristic'] = result
                ai_prob = result.get('ai_probability', 0.5)
                predictions.append(('pragmatic_heuristic', ai_prob))
                logger.debug(f"Pragmatic: {ai_prob:.3f}")
            except Exception as e:
                logger.error(f"Pragmatic detection failed: {e}")
                individual_results['pragmatic_heuristic'] = {'error': str(e)}
        
        # Calculate ensemble result
        if len(predictions) == 0:
            # All detectors failed - return uncertain result
            logger.error("All detectors failed!")
            return {
                'classification': 'UNCERTAIN',
                'ai_probability': 0.5,
                'confidence': 0.1,
                'individual_scores': individual_results,
                'agreement': 'none',
                'explanation': 'All detection methods failed'
            }
        
        # Weighted ensemble
        weighted_sum = 0.0
        weight_total = 0.0
        
        for detector_name, ai_prob in predictions:
            weight = self.weights.get(detector_name, 0.33)
            weighted_sum += weight * ai_prob
            weight_total += weight
        
        # Normalize
        ai_probability = weighted_sum / weight_total if weight_total > 0 else 0.5
        
        # Analyze agreement
        ai_votes = sum(1 for _, prob in predictions if prob > 0.5)
        human_votes = len(predictions) - ai_votes
        
        if ai_votes == len(predictions):
            agreement = 'unanimous_ai'
            agreement_boost = 1.15  # Boost confidence when all agree
        elif human_votes == len(predictions):
            agreement = 'unanimous_human'
            agreement_boost = 1.15
        elif abs(ai_votes - human_votes) >= 1:
            agreement = 'majority'
            agreement_boost = 1.0
        else:
            agreement = 'split'
            agreement_boost = 0.85  # Reduce confidence when split
        
        # Apply agreement boost to probability (push toward extremes)
        if agreement_boost > 1.0:
            if ai_probability > 0.5:
                ai_probability = 0.5 + (ai_probability - 0.5) * agreement_boost
            else:
                ai_probability = 0.5 - (0.5 - ai_probability) * agreement_boost
        
        # Clip probability
        ai_probability = np.clip(ai_probability, 0.02, 0.98)
        
        # Classification with threshold
        threshold = 0.35  # Bias toward AI detection (lower threshold)
        classification = 'AI_GENERATED' if ai_probability > threshold else 'HUMAN'
        
        # Calculate confidence
        # Higher confidence when far from threshold and detectors agree
        distance_from_threshold = abs(ai_probability - threshold)
        base_confidence = 0.5 + distance_from_threshold * 0.8
        confidence = base_confidence * agreement_boost
        confidence = np.clip(confidence, 0.4, 0.95)
        
        # Generate explanation
        explanation = self._generate_explanation(
            classification, ai_probability, individual_results, agreement
        )
        
        logger.info(f"Ensemble result: {classification} (p={ai_probability:.3f}, conf={confidence:.3f}, {agreement})")
        
        return {
            'classification': classification,
            'ai_probability': float(ai_probability),
            'confidence': float(confidence),
            'individual_scores': individual_results,
            'agreement': agreement,
            'explanation': explanation,
            'threshold': threshold,
            'weights_used': {k: v for k, v in self.weights.items() 
                           if any(k == pred[0] for pred in predictions)}
        }
    
    def _generate_explanation(self, classification: str, ai_probability: float,
                             individual_results: Dict, agreement: str) -> str:
        """Generate human-readable explanation of the detection result"""
        
        reasons = []
        
        # Analyze each detector's contribution
        for detector_name, result in individual_results.items():
            if 'error' in result:
                continue
            
            prob = result.get('ai_probability', 0.5)
            
            if detector_name == 'opensmile_acoustic':
                if prob > 0.6:
                    reasons.append("acoustic features show AI-like patterns")
                elif prob < 0.4:
                    reasons.append("natural acoustic characteristics detected")
            
            elif detector_name == 'cnn_spectrogram':
                if prob > 0.6:
                    reasons.append("spectrogram shows synthetic patterns")
                elif prob < 0.4:
                    reasons.append("natural spectrogram texture")
            
            elif detector_name == 'pragmatic_heuristic':
                if prob > 0.6:
                    reasons.append("heuristic analysis indicates AI generation")
                elif prob < 0.4:
                    reasons.append("heuristic markers suggest human voice")
        
        # Format explanation
        if classification == 'AI_GENERATED':
            if len(reasons) > 0:
                return f"AI detected: {', '.join(reasons[:2])}"
            elif agreement == 'unanimous_ai':
                return "All detection methods indicate AI-generated audio"
            else:
                return "Multiple AI-indicative features detected"
        else:
            if len(reasons) > 0:
                return f"Human voice: {', '.join(reasons[:2])}"
            elif agreement == 'unanimous_human':
                return "All detection methods indicate human voice"
            else:
                return "Voice exhibits natural human speech characteristics"
    
    def calibrate_weights(self, validation_data: List[Tuple[np.ndarray, int]], sr: int) -> Dict:
        """
        Calibrate ensemble weights using validation data
        
        Args:
            validation_data: List of (audio, label) tuples where label is 0=Human, 1=AI
            sr: Sample rate
            
        Returns:
            Calibration results including optimal weights
        """
        self._init_detectors()
        
        logger.info(f"Calibrating weights on {len(validation_data)} samples...")
        
        # Collect predictions from each detector
        all_predictions = {
            'opensmile_acoustic': [],
            'cnn_spectrogram': [],
            'pragmatic_heuristic': []
        }
        labels = []
        
        for audio, label in validation_data:
            labels.append(label)
            
            # Get each detector's prediction
            if self.opensmile_detector:
                try:
                    result = self.opensmile_detector.detect(audio, sr)
                    all_predictions['opensmile_acoustic'].append(result['ai_probability'])
                except:
                    all_predictions['opensmile_acoustic'].append(0.5)
            else:
                all_predictions['opensmile_acoustic'].append(0.5)
            
            if self.cnn_detector:
                try:
                    result = self.cnn_detector.detect(audio, sr)
                    all_predictions['cnn_spectrogram'].append(result['ai_probability'])
                except:
                    all_predictions['cnn_spectrogram'].append(0.5)
            else:
                all_predictions['cnn_spectrogram'].append(0.5)
            
            if self.pragmatic_detector:
                try:
                    result = self.pragmatic_detector.detect(audio, sr)
                    all_predictions['pragmatic_heuristic'].append(result.get('ai_probability', 0.5))
                except:
                    all_predictions['pragmatic_heuristic'].append(0.5)
            else:
                all_predictions['pragmatic_heuristic'].append(0.5)
        
        labels = np.array(labels)
        
        # Calculate individual detector accuracies
        detector_accuracies = {}
        for name, preds in all_predictions.items():
            preds = np.array(preds)
            predictions = (preds > 0.5).astype(int)
            accuracy = (predictions == labels).mean()
            detector_accuracies[name] = accuracy
            logger.info(f"{name} accuracy: {accuracy:.4f}")
        
        # Optimize weights using grid search
        best_accuracy = 0
        best_weights = self.DEFAULT_WEIGHTS.copy()
        
        # Grid search over weight combinations
        for w1 in np.arange(0.2, 0.7, 0.1):
            for w2 in np.arange(0.2, 0.7, 0.1):
                w3 = 1.0 - w1 - w2
                if w3 < 0.1 or w3 > 0.5:
                    continue
                
                weights = {
                    'cnn_spectrogram': w1,
                    'opensmile_acoustic': w2,
                    'pragmatic_heuristic': w3
                }
                
                # Calculate weighted ensemble predictions
                ensemble_preds = (
                    w1 * np.array(all_predictions['cnn_spectrogram']) +
                    w2 * np.array(all_predictions['opensmile_acoustic']) +
                    w3 * np.array(all_predictions['pragmatic_heuristic'])
                )
                
                predictions = (ensemble_preds > 0.35).astype(int)  # Using our threshold
                accuracy = (predictions == labels).mean()
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_weights = weights.copy()
        
        # Update and save weights
        self.weights = best_weights
        self._save_weights()
        
        logger.info(f"Calibration complete. Best accuracy: {best_accuracy:.4f}")
        logger.info(f"Optimal weights: {best_weights}")
        
        return {
            'success': True,
            'best_accuracy': float(best_accuracy),
            'optimal_weights': best_weights,
            'individual_accuracies': detector_accuracies,
            'n_samples': len(validation_data)
        }
    
    def get_detector_status(self) -> Dict:
        """Get status of all detectors"""
        self._init_detectors()
        
        status = {
            'opensmile_acoustic': {
                'available': self.opensmile_detector is not None,
                'trained': self.opensmile_detector.is_trained if self.opensmile_detector else False
            },
            'cnn_spectrogram': {
                'available': self.cnn_detector is not None,
                'trained': self.cnn_detector.is_trained if self.cnn_detector else False
            },
            'pragmatic_heuristic': {
                'available': self.pragmatic_detector is not None,
                'trained': True  # Heuristic is always "trained"
            },
            'weights': self.weights
        }
        
        return status


# Global instance
ensemble_detector = EnsembleDetector()
