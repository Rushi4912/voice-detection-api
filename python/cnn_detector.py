"""
CNN Spectrogram-Based AI Voice Detector

This module uses a ResNet18 CNN trained on mel spectrograms to detect
AI-generated audio. This is the state-of-the-art approach for audio
deepfake detection.

Architecture:
- Input: Mel Spectrogram (128 bands x variable time)
- Backbone: ResNet18 (pretrained on ImageNet, fine-tuned on ASVspoof)
- Output: Binary classification (AI=1, Human=0)

Training Data:
- ASVspoof 2019/2021 (Logical Access)
- WaveFake dataset
- Custom ElevenLabs/OpenAI TTS samples

References:
- ASVspoof Challenge: https://www.asvspoof.org/
- WaveFake: https://github.com/RUB-SysSec/WaveFake
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import librosa
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import os

from logger import logger

# Check for timm (PyTorch Image Models)
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    logger.warning("timm not installed. Using custom ResNet implementation.")


class MelSpectrogramTransform:
    """
    Transform audio waveform to mel spectrogram suitable for CNN input
    
    Output: 128x128 normalized mel spectrogram (grayscale -> 3-channel for ResNet)
    """
    
    def __init__(self, 
                 sr: int = 16000,
                 n_mels: int = 128,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 target_size: int = 128):
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_size = target_size
    
    def __call__(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        """
        Convert audio to mel spectrogram tensor
        
        Args:
            audio: Audio waveform
            sr: Sample rate
            
        Returns:
            Tensor of shape (3, 128, 128) suitable for ResNet
        """
        # Resample if needed
        if sr != self.sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1]
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        
        # Resize to target size (128x128)
        # Use interpolation for time axis
        if mel_spec_norm.shape[1] != self.target_size:
            # Pad or crop time dimension
            if mel_spec_norm.shape[1] < self.target_size:
                # Pad with zeros
                pad_width = self.target_size - mel_spec_norm.shape[1]
                mel_spec_norm = np.pad(mel_spec_norm, ((0, 0), (0, pad_width)), mode='constant')
            else:
                # Crop or use center
                start = (mel_spec_norm.shape[1] - self.target_size) // 2
                mel_spec_norm = mel_spec_norm[:, start:start + self.target_size]
        
        # Convert to tensor (1, H, W)
        mel_tensor = torch.from_numpy(mel_spec_norm).float().unsqueeze(0)
        
        # Repeat to 3 channels for ResNet (expects RGB input)
        mel_tensor = mel_tensor.repeat(3, 1, 1)
        
        return mel_tensor


class SimpleResNet(nn.Module):
    """
    Simplified ResNet for audio classification (fallback when timm not available)
    """
    
    def __init__(self, num_classes: int = 2):
        super().__init__()
        
        # Initial conv
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet blocks (simplified)
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int = 1):
        layers = []
        
        # First block with potential downsampling
        layers.append(self._basic_block(in_channels, out_channels, stride))
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(self._basic_block(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _basic_block(self, in_channels: int, out_channels: int, stride: int = 1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        return BasicBlock(in_channels, out_channels, stride, downsample)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


class BasicBlock(nn.Module):
    """Basic ResNet block"""
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class CNNDetector:
    """
    CNN-based AI Voice Detector using ResNet18 on mel spectrograms
    
    This is one of the three detectors in our ensemble:
    1. OpenSMILE Acoustic - prosody, voice quality, spectral
    2. CNN Spectrogram (this) - visual patterns in mel spectrograms
    3. Pragmatic Detector - heuristic-based methods
    """
    
    # Model paths
    MODEL_DIR = Path(__file__).parent / "models"
    MODEL_PATH = MODEL_DIR / "resnet_asvspoof.pt"
    
    def __init__(self, device: str = None):
        """
        Initialize CNN detector
        
        Args:
            device: 'cuda' or 'cpu'. Auto-detected if None.
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.model = None
        self.transform = MelSpectrogramTransform()
        self.is_trained = False
        
        self._load_or_create_model()
        logger.info(f"CNNDetector initialized on device: {self.device}")
    
    def _load_or_create_model(self):
        """Load pretrained model or create new one"""
        try:
            if self.MODEL_PATH.exists():
                # Load pretrained model
                self._create_model()
                state_dict = torch.load(self.MODEL_PATH, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.is_trained = True
                logger.info("Loaded pretrained CNN model")
            else:
                # Create new model (needs training)
                self._create_model()
                self.is_trained = False
                logger.info("Created new CNN model (needs training)")
            
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Error loading CNN model: {e}")
            self._create_model()
            self.model.to(self.device)
            self.model.eval()
    
    def _create_model(self):
        """Create ResNet18 model for binary classification"""
        if TIMM_AVAILABLE:
            # Use timm's pretrained ResNet18
            self.model = timm.create_model(
                'resnet18',
                pretrained=True,
                num_classes=2
            )
            logger.info("Created ResNet18 using timm (pretrained on ImageNet)")
        else:
            # Use our simple implementation
            self.model = SimpleResNet(num_classes=2)
            logger.info("Created SimpleResNet (custom implementation)")
    
    def preprocess(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        """
        Preprocess audio to mel spectrogram tensor
        
        Args:
            audio: Audio waveform
            sr: Sample rate
            
        Returns:
            Tensor ready for model input (1, 3, 128, 128)
        """
        mel_tensor = self.transform(audio, sr)
        return mel_tensor.unsqueeze(0)  # Add batch dimension
    
    @torch.no_grad()
    def detect(self, audio: np.ndarray, sr: int) -> Dict:
        """
        Detect AI-generated voice using CNN on mel spectrogram
        
        Args:
            audio: Audio waveform
            sr: Sample rate
            
        Returns:
            Dict with:
            - ai_probability: float (0-1)
            - confidence: float (0-1)
            - method: str
        """
        try:
            if not self.is_trained:
                # Use heuristic-based spectrogram analysis
                return self._heuristic_detection(audio, sr)
            
            # Preprocess
            input_tensor = self.preprocess(audio, sr).to(self.device)
            
            # Forward pass
            self.model.eval()
            logits = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=1)
            
            # Get AI probability (class 1)
            ai_probability = probabilities[0, 1].item()
            
            # Confidence based on probability distance from 0.5
            confidence = abs(ai_probability - 0.5) * 2
            confidence = min(0.95, max(0.5, confidence))
            
            return {
                'ai_probability': float(ai_probability),
                'confidence': float(confidence),
                'method': 'resnet18_spectrogram',
                'device': self.device
            }
            
        except Exception as e:
            logger.error(f"CNN detection error: {e}")
            return {
                'ai_probability': 0.5,
                'confidence': 0.3,
                'method': 'fallback',
                'error': str(e)
            }
    
    def _heuristic_detection(self, audio: np.ndarray, sr: int) -> Dict:
        """
        Heuristic detection based on spectrogram patterns when model is not trained
        """
        try:
            # Get mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_mels=128
            )
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            scores = []
            
            # 1. Check temporal smoothness (AI tends to be smoother)
            temporal_diff = np.diff(mel_db, axis=1)
            temporal_variance = np.var(temporal_diff)
            
            if temporal_variance < 15:
                scores.append(0.7)  # Too smooth = AI
            elif temporal_variance > 30:
                scores.append(0.3)  # Natural variation = Human
            else:
                scores.append(0.5)
            
            # 2. Check spectral consistency (AI has unnatural patterns)
            spectral_std = np.std(mel_db, axis=0)
            spectral_cv = np.std(spectral_std) / (np.mean(spectral_std) + 1e-10)
            
            if spectral_cv < 0.3:
                scores.append(0.7)  # Too consistent = AI
            elif spectral_cv > 0.6:
                scores.append(0.3)  # Natural = Human
            else:
                scores.append(0.5)
            
            # 3. Check for unnatural high-frequency content
            high_freq_energy = np.mean(mel_db[100:, :])
            mid_freq_energy = np.mean(mel_db[40:100, :])
            
            hf_ratio = high_freq_energy / (mid_freq_energy + 1e-10)
            
            if hf_ratio > 0.9:  # Unnaturally high HF
                scores.append(0.6)
            elif hf_ratio < 0.5:  # Natural rolloff
                scores.append(0.4)
            else:
                scores.append(0.5)
            
            # 4. Check for periodic patterns
            avg_temporal = np.mean(mel_db, axis=0)
            if len(avg_temporal) > 20:
                autocorr = np.correlate(avg_temporal, avg_temporal, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                if len(autocorr) > 1:
                    autocorr = autocorr / (autocorr[0] + 1e-10)
                    
                    # Count significant peaks
                    peaks = 0
                    for i in range(5, min(len(autocorr)-1, 50)):
                        if autocorr[i] > 0.3 and autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                            peaks += 1
                    
                    if peaks >= 3:
                        scores.append(0.7)  # Periodic = AI
                    else:
                        scores.append(0.4)
            
            ai_probability = np.mean(scores)
            confidence = 0.5 + abs(ai_probability - 0.5) * 0.3
            
            return {
                'ai_probability': float(ai_probability),
                'confidence': float(confidence),
                'method': 'spectrogram_heuristic',
                'note': 'Using heuristic mode - CNN not trained'
            }
            
        except Exception as e:
            logger.error(f"Heuristic detection error: {e}")
            return {
                'ai_probability': 0.5,
                'confidence': 0.3,
                'method': 'fallback',
                'error': str(e)
            }
    
    def train_step(self, audio_batch: List[np.ndarray], sr: int, labels: List[int],
                   optimizer: torch.optim.Optimizer) -> float:
        """
        Single training step
        
        Args:
            audio_batch: List of audio waveforms
            sr: Sample rate
            labels: List of labels (0=Human, 1=AI)
            optimizer: PyTorch optimizer
            
        Returns:
            Loss value
        """
        self.model.train()
        
        # Prepare batch
        tensors = [self.preprocess(audio, sr).squeeze(0) for audio in audio_batch]
        input_batch = torch.stack(tensors).to(self.device)
        label_tensor = torch.tensor(labels, dtype=torch.long).to(self.device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = self.model(input_batch)
        loss = F.cross_entropy(logits, label_tensor)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def train(self, train_data: List[Tuple[np.ndarray, int]], sr: int,
              epochs: int = 10, batch_size: int = 16, lr: float = 1e-4) -> Dict:
        """
        Train the CNN model
        
        Args:
            train_data: List of (audio, label) tuples
            sr: Sample rate
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            
        Returns:
            Training metrics
        """
        logger.info(f"Training CNN on {len(train_data)} samples for {epochs} epochs...")
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        
        history = {'loss': [], 'accuracy': []}
        
        for epoch in range(epochs):
            epoch_losses = []
            correct = 0
            total = 0
            
            # Shuffle data
            indices = np.random.permutation(len(train_data))
            
            # Process in batches
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_audio = [train_data[j][0] for j in batch_indices]
                batch_labels = [train_data[j][1] for j in batch_indices]
                
                loss = self.train_step(batch_audio, sr, batch_labels, optimizer)
                epoch_losses.append(loss)
                
                # Calculate accuracy
                with torch.no_grad():
                    tensors = [self.preprocess(audio, sr).squeeze(0) for audio in batch_audio]
                    input_batch = torch.stack(tensors).to(self.device)
                    logits = self.model(input_batch)
                    predictions = logits.argmax(dim=1).cpu().numpy()
                    correct += (predictions == np.array(batch_labels)).sum()
                    total += len(batch_labels)
            
            scheduler.step()
            
            avg_loss = np.mean(epoch_losses)
            accuracy = correct / total
            history['loss'].append(avg_loss)
            history['accuracy'].append(accuracy)
            
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # Save model
        self.MODEL_DIR.mkdir(exist_ok=True)
        torch.save(self.model.state_dict(), self.MODEL_PATH)
        self.is_trained = True
        
        logger.info(f"Training complete. Model saved to {self.MODEL_PATH}")
        
        return {
            'success': True,
            'epochs': epochs,
            'final_loss': history['loss'][-1],
            'final_accuracy': history['accuracy'][-1],
            'history': history
        }
    
    def save_model(self, path: Optional[Path] = None):
        """Save model to disk"""
        if path is None:
            path = self.MODEL_PATH
        
        path.parent.mkdir(exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Optional[Path] = None):
        """Load model from disk"""
        if path is None:
            path = self.MODEL_PATH
        
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.is_trained = True
        logger.info(f"Model loaded from {path}")


# Global instance
cnn_detector = CNNDetector()
