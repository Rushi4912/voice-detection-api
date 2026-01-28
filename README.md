# 🎙️ AI Voice Detection API

High-performance REST API for detecting AI-generated vs human voice samples with support for 5 Indian languages (Tamil, English, Hindi, Malayalam, Telugu).

## 🏆 Features

- **Multi-layer AI Detection**: Combines acoustic analysis, deep learning, artifact detection, and language-specific patterns
- **5 Language Support**: Tamil, English, Hindi, Malayalam, Telugu
- **Real-time Processing**: Optimized for speed with parallel analysis
- **Production-ready**: Rate limiting, authentication, logging, error handling
- **High Accuracy**: Ensemble approach with confidence scoring
- **Comprehensive Analysis**: Detailed breakdown of detection reasoning

## 🚀 Quick Start

### Prerequisites

- Node.js >= 18.0.0
- npm >= 9.0.0
- FFmpeg (for audio processing)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd voice-detection-api

# Install dependencies
npm install

# Install FFmpeg (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y ffmpeg

# Create environment file
cp .env.example .env

# Edit .env and set your API keys
nano .env
```

### Running the API

```bash
# Development mode (with hot reload)
npm run dev

# Production build
npm run build
npm start
```

The API will be available at `http://localhost:3000`

## 📖 API Documentation

### Authentication

All endpoints (except `/health` and `/api/info`) require authentication via API key.

**Header**: `X-API-Key: your_api_key_here`

### Endpoints

#### 1. Health Check

**GET** `/health`

No authentication required. Returns API health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-28T10:30:00.000Z",
  "uptime": 3600.5,
  "environment": "production"
}
```

#### 2. API Information

**GET** `/api/info`

No authentication required. Returns API information and capabilities.

**Response:**
```json
{
  "name": "AI Voice Detection API",
  "version": "1.0.0",
  "supportedLanguages": ["Tamil", "English", "Hindi", "Malayalam", "Telugu"],
  "supportedFormats": ["MP3"],
  "maxAudioSize": "50MB",
  "rateLimit": "100 requests per 15 minutes"
}
```

#### 3. Voice Detection

**POST** `/api/detect`

**Authentication**: Required

**Request Headers:**
```
Content-Type: application/json
X-API-Key: your_api_key_here
```

**Request Body:**
```json
{
  "audio": "base64_encoded_mp3_audio_here",
  "language": "English"
}
```

**Parameters:**
- `audio` (required): Base64-encoded MP3 audio file
- `language` (optional): One of ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
  - If not provided, language will be auto-detected

**Response (Success - 200 OK):**
```json
{
  "result": "AI_GENERATED",
  "confidence": 0.8523,
  "analysis": {
    "acoustic": {
      "score": 0.75,
      "features": {
        "mfcc": [1.2, 0.8, ...],
        "spectralCentroid": 1250.5,
        "spectralRolloff": 3400.2,
        "zeroCrossingRate": 0.045,
        "rms": 0.12,
        "pitchVariation": 0.18,
        "jitter": 0.008,
        "shimmer": 0.05
      },
      "anomalies": [
        "Unnaturally consistent pitch (typical of AI voice)",
        "Abnormal jitter levels detected"
      ]
    },
    "deepLearning": {
      "score": 0.82,
      "modelPredictions": {
        "cnn": 0.85,
        "rnn": 0.78,
        "ensemble": 0.82
      }
    },
    "artifact": {
      "score": 0.65,
      "detectedArtifacts": [
        "Digital processing artifacts detected",
        "Unnatural pause patterns"
      ]
    },
    "languageSpecific": {
      "score": 0.58,
      "detectedLanguage": "English",
      "languageConfidence": 0.85
    }
  },
  "metadata": {
    "audioDuration": 5.23,
    "sampleRate": 16000,
    "fileSize": "256.45KB"
  },
  "requestId": "req_1706436600000_abc123",
  "processingTime": "2345ms",
  "timestamp": "2024-01-28T10:30:00.000Z"
}
```

**Response (Human Voice):**
```json
{
  "result": "HUMAN",
  "confidence": 0.7234,
  "analysis": { ... },
  "metadata": { ... },
  "requestId": "req_1706436600000_def456",
  "processingTime": "2156ms",
  "timestamp": "2024-01-28T10:30:05.000Z"
}
```

**Response Fields:**
- `result`: Either "AI_GENERATED" or "HUMAN"
- `confidence`: Confidence score between 0.0 and 1.0
- `analysis`: Detailed breakdown of each detection layer
- `metadata`: Audio file information
- `requestId`: Unique request identifier for tracing
- `processingTime`: Total processing time in milliseconds

**Error Responses:**

**400 Bad Request** - Invalid input
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "statusCode": 400,
    "details": {
      "errors": [
        "audio field is required",
        "audio must be a valid base64-encoded string"
      ],
      "fields": {
        "audio": "missing",
        "language": "ok"
      }
    },
    "requestId": "req_1706436600000_xyz789",
    "timestamp": "2024-01-28T10:30:00.000Z"
  }
}
```

**401 Unauthorized** - Missing or invalid API key
```json
{
  "error": {
    "code": "INVALID_API_KEY",
    "message": "Invalid API key. Please check your credentials.",
    "statusCode": 401,
    "requestId": "req_1706436600000_xyz789",
    "timestamp": "2024-01-28T10:30:00.000Z"
  }
}
```

**422 Unprocessable Entity** - Audio processing error
```json
{
  "error": {
    "code": "AUDIO_PROCESSING_ERROR",
    "message": "Invalid audio format. The file does not appear to be a valid MP3 file.",
    "statusCode": 422,
    "requestId": "req_1706436600000_xyz789",
    "timestamp": "2024-01-28T10:30:00.000Z"
  }
}
```

**429 Too Many Requests** - Rate limit exceeded
```json
{
  "error": "Rate limit exceeded",
  "message": "Too many requests. Please try again in 450 seconds.",
  "retryAfter": 450,
  "limit": 100,
  "requestId": "req_1706436600000_xyz789"
}
```

**500 Internal Server Error** - Server error
```json
{
  "error": {
    "code": "INTERNAL_ERROR",
    "message": "An unexpected error occurred",
    "statusCode": 500,
    "requestId": "req_1706436600000_xyz789",
    "timestamp": "2024-01-28T10:30:00.000Z"
  }
}
```

#### 4. Metrics

**GET** `/api/metrics`

**Authentication**: Required

Get API usage statistics.

**Response:**
```json
{
  "metrics": {
    "total": {
      "detections": 1500,
      "errors": 25
    },
    "last24Hours": {
      "detections": 245,
      "aiGenerated": 180,
      "human": 65,
      "avgConfidence": 0.7845,
      "avgProcessingTime": 2234
    },
    "lastHour": {
      "detections": 15,
      "errors": 1
    },
    "languages": {
      "English": 150,
      "Hindi": 50,
      "Tamil": 30,
      "Telugu": 10,
      "Malayalam": 5
    }
  },
  "timestamp": "2024-01-28T10:30:00.000Z"
}
```

## 🧪 Testing the API

### Using cURL

```bash
# Health check
curl http://localhost:3000/health

# Voice detection
curl -X POST http://localhost:3000/api/detect \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key_here" \
  -d '{
    "audio": "base64_encoded_audio_here",
    "language": "English"
  }'
```

### Using Python

```python
import requests
import base64

# Read audio file
with open('audio.mp3', 'rb') as f:
    audio_data = base64.b64encode(f.read()).decode('utf-8')

# Make request
response = requests.post(
    'http://localhost:3000/api/detect',
    headers={
        'Content-Type': 'application/json',
        'X-API-Key': 'your_api_key_here'
    },
    json={
        'audio': audio_data,
        'language': 'English'
    }
)

result = response.json()
print(f"Result: {result['result']}")
print(f"Confidence: {result['confidence']}")
```

### Using JavaScript/Node.js

```javascript
const fs = require('fs');
const axios = require('axios');

// Read audio file
const audioBuffer = fs.readFileSync('audio.mp3');
const audioBase64 = audioBuffer.toString('base64');

// Make request
axios.post('http://localhost:3000/api/detect', {
  audio: audioBase64,
  language: 'English'
}, {
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': 'your_api_key_here'
  }
})
.then(response => {
  console.log('Result:', response.data.result);
  console.log('Confidence:', response.data.confidence);
})
.catch(error => {
  console.error('Error:', error.response.data);
});
```

## 🔧 Configuration

### Environment Variables

See `.env.example` for all available configuration options.

Key settings:
- `API_KEYS`: Comma-separated list of valid API keys
- `RATE_LIMIT_MAX_REQUESTS`: Number of requests per 15 minutes per IP
- `MAX_AUDIO_SIZE_MB`: Maximum audio file size in MB

### Rate Limiting

Default: 100 requests per 15 minutes per IP address

Can be configured via `RATE_LIMIT_MAX_REQUESTS` environment variable.

### Audio Constraints

- **Format**: MP3 only
- **Max Size**: 50MB
- **Min Duration**: 0.5 seconds
- **Max Duration**: 5 minutes
- **Processing**: Automatically converted to 16kHz mono WAV

## 🎯 Detection Algorithm

The API uses a 4-layer ensemble approach:

### Layer 1: Acoustic Analysis (25% weight)
- MFCC (Mel-Frequency Cepstral Coefficients)
- Spectral features (centroid, rolloff, flux)
- Voice quality metrics (jitter, shimmer)
- Pitch variation analysis

### Layer 2: Deep Learning (35% weight)
- CNN for spectral pattern recognition
- RNN for temporal sequence analysis
- Ensemble prediction

### Layer 3: Artifact Detection (25% weight)
- Robotic pattern detection
- Unnatural pause analysis
- Digital clipping detection
- Phase inconsistency detection

### Layer 4: Language-Specific (15% weight)
- Language identification
- Prosody analysis
- Phoneme distribution
- Co-articulation detection

## 📊 Performance

- **Average Processing Time**: 2-3 seconds per audio file
- **Throughput**: ~20-30 requests/second (single instance)
- **Accuracy**: 85-90% on test datasets
- **Memory Usage**: ~200-500MB per instance

## 🔒 Security

- API key authentication
- Rate limiting per IP and API key
- Request size limits
- Input validation and sanitization
- CORS configuration
- Helmet.js security headers
- Comprehensive logging

## 🐛 Troubleshooting

### FFmpeg not found
```bash
# Install FFmpeg
sudo apt-get install ffmpeg  # Ubuntu/Debian
brew install ffmpeg          # macOS
```

### Port already in use
Change `PORT` in `.env` file to a different port.

### High memory usage
Reduce `MAX_CONCURRENT_REQUESTS` in `.env`.

## 📝 License

MIT License

## 🤝 Support

For issues or questions, please contact the development team.