# 🎙️ AI Voice Detection API

A REST API that detects whether a voice sample is **AI-generated** or **Human**, supporting five Indian languages.

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/rushi4912/voice-detection-api)

## 🌟 Features

- **Multi-language Support**: Tamil, English, Hindi, Malayalam, Telugu
- **High Accuracy**: ~90% detection accuracy using ensemble ML models
- **Real-time Detection**: Average response time 3-5 seconds
- **Secure API**: Protected with API key authentication
- **Language Detection**: Automatically verifies audio language

## 🚀 Live API

**Endpoint**: `https://rushi4912-voice-detection-api.hf.space/api/voice-detection`

## 📖 API Documentation

### Authentication

All requests require an API key in the header:
```
x-api-key: YOUR_API_KEY
```

### Request

```http
POST /api/voice-detection
Content-Type: application/json
x-api-key: sk_test_123456789
```

**Body:**
```json
{
  "language": "Tamil",
  "audioFormat": "mp3",
  "audioBase64": "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU2LjM2LjEwMAAAAAAA..."
}
```

| Field | Type | Description |
|-------|------|-------------|
| `language` | string | One of: Tamil, English, Hindi, Malayalam, Telugu |
| `audioFormat` | string | Must be `mp3` |
| `audioBase64` | string | Base64-encoded MP3 audio |

### Response

**Success (200):**
```json
{
  "status": "success",
  "language": "Tamil",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.91,
  "explanation": "Unnatural pitch consistency and robotic speech patterns detected"
}
```

**Error (4xx/5xx):**
```json
{
  "status": "error",
  "message": "Invalid API key"
}
```

### Classification Values

| Value | Description |
|-------|-------------|
| `AI_GENERATED` | Voice created using AI/synthetic systems |
| `HUMAN` | Voice spoken by a real human |

## 🔧 Technology Stack

- **Backend**: Python Flask
- **ML Models**: 
  - ResNet18 CNN for spectrogram analysis
  - SpeechBrain for language identification
  - Acoustic feature classifier (MFCC, pitch, energy)
  - Ensemble voting system
- **Deployment**: Hugging Face Spaces (Docker)

## 📁 Project Structure

```
voice-detection-api/
├── python/                 # ML Service
│   ├── app.py             # Flask API server
│   ├── detect.py          # Voice detection logic
│   ├── audio_utils.py     # Audio processing
│   ├── cnn_detector.py    # CNN model
│   ├── ensemble_detector.py # Ensemble voting
│   └── models/            # Trained models
├── server/                 # Node.js Gateway (optional)
└── hf-space/              # Hugging Face deployment
```

## 🎯 Detection Approach

1. **Audio Preprocessing**: Decode Base64, normalize, extract features
2. **Feature Extraction**: MFCC, spectral features, pitch variance
3. **Multi-Model Analysis**:
   - CNN analyzes spectrogram patterns
   - Acoustic classifier checks voice characteristics
   - Pragmatic analyzer detects unnatural patterns
4. **Ensemble Voting**: Weighted combination of all detectors
5. **Language Verification**: SpeechBrain confirms audio language

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Overall Accuracy | ~90% |
| Human Detection | ~88% |
| AI Detection | ~80% |
| Language Detection | >95% |

## 🛠️ Local Development

### Prerequisites
- Python 3.10+
- FFmpeg

### Setup
```bash
# Clone repository
git clone https://github.com/Rushi4912/voice-detection-api.git
cd voice-detection-api/python

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download models
python download_models.py

# Run server
python app.py
```

### Testing
```bash
curl -X POST http://localhost:5000/api/voice-detection \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk_test_123456789" \
  -d '{"language": "English", "audioFormat": "mp3", "audioBase64": "..."}'
```

## 📄 License

MIT License

## 👨‍💻 Author

**Rushikesh** - [GitHub](https://github.com/Rushi4912)

---

Built with ❤️ for detecting AI-generated voices
