# AI Voice Detection - Training Guide

## Quick Start: Train on Real Data

### Step 1: Collect Audio Samples

**Folder Structure:**
```
training_data/
├── human/           # Real human recordings
│   ├── tamil/
│   ├── english/
│   ├── hindi/
│   ├── malayalam/
│   └── telugu/
│
└── ai/              # AI-generated audio
    ├── tamil/
    ├── english/
    ├── hindi/
    ├── malayalam/
    └── telugu/
```

### Step 2: Add Your Audio Files

**For Human Samples:**
- Record real people speaking
- Use phone recordings
- Download from speech datasets

**For AI Samples:**
- Use ElevenLabs (https://elevenlabs.io)
- Use Google TTS
- Use Amazon Polly
- Use any AI voice generator

**Requirements:**
- 3-10 seconds per audio
- WAV, MP3, FLAC, OGG, M4A formats
- 50-100 samples per language per category

### Step 3: Train the Models

```bash
cd /home/rushikesh/Desktop/voice-detection-api/python

# Train with default settings (10 epochs)
./venv/bin/python3 train_models.py --data-dir training_data --epochs 10

# Train with more epochs for better accuracy
./venv/bin/python3 train_models.py --data-dir training_data --epochs 20
```

**Training Time (CPU with 16GB RAM):**
- 100 samples: ~10-15 minutes
- 500 samples: ~1-2 hours
- 1000 samples: ~3-4 hours

### Step 4: Test the API

```bash
# Start the API
./venv/bin/python3 app.py
```

Then test with Postman:
```
POST http://localhost:5000/analyze
Content-Type: application/json

{
    "language": "English",
    "audio_base64": "<your-base64-encoded-audio>"
}
```

---

## How the Detection Works

```
┌──────────────────────────────────────────────────────────────┐
│                        AUDIO INPUT                           │
└─────────────────────────┬────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
   │   ACOUSTIC   │ │     CNN      │ │  PRAGMATIC   │
   │   DETECTOR   │ │   DETECTOR   │ │   DETECTOR   │
   │              │ │              │ │              │
   │ F0, Jitter,  │ │ ResNet18 on  │ │ Heuristic    │
   │ Shimmer,     │ │ Mel Spectro- │ │ rules for    │
   │ MFCCs → SVM  │ │ gram (128    │ │ smoothness,  │
   │              │ │ bands)       │ │ patterns     │
   └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
          │                │                │
          │   Score: 0-1   │   Score: 0-1   │   Score: 0-1
          │                │                │
          └───────────────┬┴────────────────┘
                          │
                          ▼
               ┌─────────────────────┐
               │  ENSEMBLE VOTING    │
               │                     │
               │  Weights:           │
               │  - Acoustic: 30%    │
               │  - CNN: 20%         │
               │  - Pragmatic: 50%   │
               └──────────┬──────────┘
                          │
                          ▼
               ┌─────────────────────┐
               │   FINAL RESULT      │
               │                     │
               │  AI_GENERATED or    │
               │  HUMAN              │
               │  + Confidence Score │
               └─────────────────────┘
```

---

## Tips for 90% Accuracy

1. **Diverse AI Sources**: Use samples from different TTS services
2. **Diverse Human Speakers**: Different ages, genders, accents
3. **Balanced Dataset**: Similar number of human and AI samples
4. **Clean Audio**: Avoid very noisy recordings
5. **Train Longer**: More epochs = better accuracy (up to a point)

---

## Files in This Project

```
python/
├── app.py                 # Flask API server
├── detect.py              # Main detector (uses ensemble)
├── ensemble_detector.py   # Combines all 3 methods
├── opensmile_detector.py  # Acoustic features + SVM
├── cnn_detector.py        # ResNet18 on spectrograms
├── pragmatic_detector.py  # Heuristic-based detection
├── train_models.py        # Training script
├── audio_utils.py         # Audio processing utilities
├── config.py              # Configuration
├── logger.py              # Logging setup
├── requirements.txt       # Python dependencies
├── models/                # Trained model files
│   ├── acoustic_classifier.pkl
│   ├── acoustic_scaler.pkl
│   ├── resnet_asvspoof.pt
│   └── ensemble_weights.json
└── training_data/         # Your training samples go here
```
