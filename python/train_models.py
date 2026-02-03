"""
Model Training Script for AI Voice Detection

This script trains the AI voice detection models:
1. Downloads training data (ASVspoof subset + WaveFake subset)
2. Trains OpenSMILE acoustic classifier (SVM)
3. Fine-tunes CNN spectrogram model (ResNet18)
4. Calibrates ensemble weights

Estimated time: 3-4 hours on CPU with 16GB RAM

Usage:
    python train_models.py                    # Full training pipeline
    python train_models.py --download-only    # Just download data
    python train_models.py --acoustic-only    # Train only acoustic classifier
    python train_models.py --cnn-only         # Train only CNN model
    python train_models.py --calibrate-only   # Just calibrate ensemble weights
"""

import os
import sys
import argparse
import numpy as np
import librosa
import requests
import zipfile
import tarfile
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm
import json
import random

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from logger import logger


# Configuration
DATA_DIR = Path(__file__).parent / "training_data"
MODEL_DIR = Path(__file__).parent / "models"
SAMPLE_RATE = 16000

# Dataset URLs (using publicly available subsets)
DATASET_URLS = {
    # WaveFake subset from GitHub releases
    'wavefake_subset': 'https://github.com/RUB-SysSec/WaveFake/releases/download/v1.0/wavefake_ljspeech_small.zip',
    # Alternative: LJSpeech bonafide samples
    'ljspeech_samples': 'https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2',
}


def download_file(url: str, dest_path: Path, desc: str = "Downloading") -> bool:
    """Download file with progress bar"""
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as f:
            with tqdm(total=total_size, unit='iB', unit_scale=True, desc=desc) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)
        
        return True
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def create_synthetic_training_data(num_samples: int = 500) -> List[Tuple[np.ndarray, int]]:
    """
    Create synthetic training data for initial testing
    
    This generates audio samples with characteristics of AI vs Human:
    - AI: Smooth spectral envelope, consistent pitch, low jitter
    - Human: Variable pitch, natural jitter/shimmer, rich harmonics
    
    Note: This is for testing the pipeline. For production, use real datasets.
    """
    logger.info(f"Creating {num_samples} synthetic training samples...")
    
    training_data = []
    duration = 3.0  # seconds
    num_samples_audio = int(duration * SAMPLE_RATE)
    
    for i in tqdm(range(num_samples), desc="Generating synthetic data"):
        label = i % 2  # Alternating Human (0) and AI (1)
        
        if label == 0:  # Human-like
            # Variable F0 with natural jitter
            t = np.linspace(0, duration, num_samples_audio)
            f0_base = np.random.uniform(100, 250)  # Variable base frequency
            f0_variation = 20 + np.random.rand() * 30  # Significant variation
            jitter_amount = 0.02 + np.random.rand() * 0.03
            
            # F0 with natural variation
            f0 = f0_base + f0_variation * np.sin(2 * np.pi * 0.5 * t)
            f0 += np.random.randn(len(t)) * f0_base * jitter_amount
            
            # Generate audio with harmonics
            audio = np.zeros(num_samples_audio)
            for harmonic in range(1, 6):
                amp = 1.0 / harmonic
                phase = 2 * np.pi * np.cumsum(f0 * harmonic / SAMPLE_RATE)
                audio += amp * np.sin(phase)
            
            # Add natural noise
            audio += np.random.randn(num_samples_audio) * 0.05
            
            # Variable amplitude envelope
            envelope = np.ones(num_samples_audio)
            envelope *= 1 + 0.3 * np.sin(2 * np.pi * 1.5 * t)
            envelope += np.random.randn(num_samples_audio) * 0.1
            audio *= np.clip(envelope, 0.3, 1.5)
            
        else:  # AI-like
            # Consistent F0 with minimal jitter
            t = np.linspace(0, duration, num_samples_audio)
            f0_base = np.random.uniform(120, 180)  # Consistent frequency
            f0_variation = 5 + np.random.rand() * 5  # Minimal variation
            jitter_amount = 0.001 + np.random.rand() * 0.002
            
            # Smooth F0
            f0 = f0_base + f0_variation * np.sin(2 * np.pi * 0.3 * t)
            
            # Generate very clean audio
            audio = np.zeros(num_samples_audio)
            for harmonic in range(1, 8):
                amp = 1.0 / harmonic
                phase = 2 * np.pi * np.cumsum(f0 * harmonic / SAMPLE_RATE)
                audio += amp * np.sin(phase)
            
            # Minimal noise
            audio += np.random.randn(num_samples_audio) * 0.005
            
            # Smooth amplitude envelope
            envelope = np.ones(num_samples_audio)
            envelope *= 1 + 0.05 * np.sin(2 * np.pi * 0.5 * t)
            audio *= envelope
        
        # Normalize
        audio = audio / np.max(np.abs(audio))
        
        training_data.append((audio, label))
    
    # Shuffle
    random.shuffle(training_data)
    
    logger.info(f"Generated {len(training_data)} samples (50% Human, 50% AI)")
    return training_data


def load_audio_file(file_path: Path) -> Optional[Tuple[np.ndarray, int]]:
    """Load audio file and resample to target rate"""
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True, duration=5.0)
        return audio, sr
    except Exception as e:
        logger.warning(f"Could not load {file_path}: {e}")
        return None


def prepare_dataset_from_folder(folder_path: Path, label: int) -> List[Tuple[np.ndarray, int]]:
    """Load all audio files from a folder with given label"""
    data = []
    
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    
    for ext in audio_extensions:
        for file_path in folder_path.glob(f'**/*{ext}'):
            result = load_audio_file(file_path)
            if result:
                audio, sr = result
                data.append((audio, label))
    
    logger.info(f"Loaded {len(data)} samples from {folder_path}")
    return data


def train_acoustic_classifier(training_data: List[Tuple[np.ndarray, int]]) -> dict:
    """Train the OpenSMILE/librosa acoustic classifier"""
    logger.info("Training acoustic classifier...")
    
    from opensmile_detector import opensmile_detector
    
    # Extract features
    X = []
    y = []
    expected_feature_len = None
    
    for audio, label in tqdm(training_data, desc="Extracting acoustic features"):
        try:
            features = opensmile_detector.extract_features(audio, SAMPLE_RATE)
            features = np.array(features).flatten()
            
            # Determine expected feature length from first valid sample
            if expected_feature_len is None:
                expected_feature_len = len(features)
                logger.info(f"Expected feature length: {expected_feature_len}")
            
            # Skip samples with inconsistent feature length
            if len(features) != expected_feature_len:
                logger.warning(f"Skipping sample with inconsistent feature length: {len(features)} vs {expected_feature_len}")
                continue
            
            # Handle NaN values - replace with 0, skip if all NaN
            if np.all(np.isnan(features)):
                logger.warning("Skipping sample with all NaN features")
                continue
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                
            X.append(features)
            y.append(label)
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            continue
    
    logger.info(f"Successfully extracted features from {len(X)} samples")
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    
    logger.info(f"Feature matrix shape: {X.shape}")
    
    # Train
    result = opensmile_detector.train(X, y)
    
    return result


def train_cnn_model(training_data: List[Tuple[np.ndarray, int]], 
                   epochs: int = 10, 
                   batch_size: int = 16) -> dict:
    """Train the CNN spectrogram model"""
    logger.info("Training CNN model...")
    
    from cnn_detector import cnn_detector
    
    # Train
    result = cnn_detector.train(
        training_data,
        sr=SAMPLE_RATE,
        epochs=epochs,
        batch_size=batch_size
    )
    
    return result


def calibrate_ensemble(validation_data: List[Tuple[np.ndarray, int]]) -> dict:
    """Calibrate ensemble weights using validation data"""
    logger.info("Calibrating ensemble weights...")
    
    from ensemble_detector import ensemble_detector
    
    result = ensemble_detector.calibrate_weights(validation_data, SAMPLE_RATE)
    
    return result


def split_data(data: List[Tuple[np.ndarray, int]], 
              train_ratio: float = 0.8) -> Tuple[List, List]:
    """Split data into training and validation sets"""
    random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]


def main():
    parser = argparse.ArgumentParser(description='Train AI Voice Detection Models')
    parser.add_argument('--download-only', action='store_true', help='Only download datasets')
    parser.add_argument('--acoustic-only', action='store_true', help='Train only acoustic classifier')
    parser.add_argument('--cnn-only', action='store_true', help='Train only CNN model')
    parser.add_argument('--calibrate-only', action='store_true', help='Only calibrate ensemble')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data for testing')
    parser.add_argument('--samples', type=int, default=500, help='Number of training samples')
    parser.add_argument('--epochs', type=int, default=10, help='CNN training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for CNN')
    parser.add_argument('--data-dir', type=str, help='Custom training data directory')
    
    args = parser.parse_args()
    
    # Create directories
    DATA_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("AI Voice Detection Model Training")
    logger.info("=" * 60)
    
    # Prepare training data
    if args.synthetic:
        logger.info("Using synthetic training data...")
        all_data = create_synthetic_training_data(args.samples)
    elif args.data_dir:
        logger.info(f"Loading data from {args.data_dir}...")
        data_path = Path(args.data_dir)
        
        # Assume subfolders: human/ and ai/
        human_data = prepare_dataset_from_folder(data_path / 'human', label=0)
        ai_data = prepare_dataset_from_folder(data_path / 'ai', label=1)
        all_data = human_data + ai_data
        random.shuffle(all_data)
    else:
        logger.info("No data source specified. Using synthetic data for demo...")
        all_data = create_synthetic_training_data(args.samples)
    
    if len(all_data) < 10:
        logger.error("Not enough training data. Need at least 10 samples.")
        return
    
    # Split data
    train_data, val_data = split_data(all_data, train_ratio=0.8)
    logger.info(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")
    
    results = {}
    
    # Train acoustic classifier
    if not args.cnn_only and not args.calibrate_only:
        logger.info("\n" + "=" * 40)
        logger.info("PHASE 1: Training Acoustic Classifier")
        logger.info("=" * 40)
        
        results['acoustic'] = train_acoustic_classifier(train_data)
        logger.info(f"Acoustic training result: {results['acoustic']}")
    
    # Train CNN model
    if not args.acoustic_only and not args.calibrate_only:
        logger.info("\n" + "=" * 40)
        logger.info("PHASE 2: Training CNN Model")
        logger.info("=" * 40)
        
        results['cnn'] = train_cnn_model(
            train_data, 
            epochs=args.epochs, 
            batch_size=args.batch_size
        )
        logger.info(f"CNN training result: {results['cnn']}")
    
    # Calibrate ensemble
    if not args.acoustic_only and not args.cnn_only:
        logger.info("\n" + "=" * 40)
        logger.info("PHASE 3: Calibrating Ensemble")
        logger.info("=" * 40)
        
        results['ensemble'] = calibrate_ensemble(val_data)
        logger.info(f"Ensemble calibration result: {results['ensemble']}")
    
    # Save results summary
    summary_path = MODEL_DIR / "training_summary.json"
    with open(summary_path, 'w') as f:
        # Convert numpy types to Python types
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        json.dump(convert(results), f, indent=2)
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {summary_path}")
    logger.info("\nModel files created:")
    for f in MODEL_DIR.glob("*"):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            logger.info(f"  - {f.name} ({size_kb:.1f} KB)")
    
    return results


if __name__ == '__main__':
    main()
