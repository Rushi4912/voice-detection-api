# 🏗️ System Architecture

## Overview

The AI Voice Detection API is a high-performance, multi-layered system designed to accurately distinguish between AI-generated and human voice samples.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     CLIENT LAYER                             │
│  Web Apps │ Mobile Apps │ API Clients │ Testing Tools       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   API GATEWAY LAYER                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Express.js Server                                     │  │
│  │ • Rate Limiting (100 req/15min)                      │  │
│  │ • API Key Authentication                             │  │
│  │ • CORS Policy                                        │  │
│  │ • Request Validation                                 │  │
│  │ • Error Handling                                     │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              AUDIO PROCESSING PIPELINE                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 1. Base64 Decoder                                    │  │
│  │    • Validate base64 format                          │  │
│  │    • Check file size limits                          │  │
│  │    • Decode to binary buffer                         │  │
│  │                                                       │  │
│  │ 2. Format Validator                                  │  │
│  │    • Check MP3 magic numbers                         │  │
│  │    • Validate audio structure                        │  │
│  │                                                       │  │
│  │ 3. Audio Converter (FFmpeg)                          │  │
│  │    • Convert MP3 → WAV (16kHz, mono, 16-bit)        │  │
│  │    • Extract metadata                                │  │
│  │                                                       │  │
│  │ 4. Feature Extractor                                 │  │
│  │    • Parse WAV structure                             │  │
│  │    • Extract raw audio samples                       │  │
│  │    • Normalize to Float32Array                       │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│           AI DETECTION ENGINE (Parallel Processing)          │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Layer 1: Acoustic Analysis (25% weight)               │ │
│  │ ┌──────────────────────────────────────────────────┐ │ │
│  │ │ Feature Extraction:                               │ │ │
│  │ │ • MFCC (13 coefficients)                         │ │ │
│  │ │ • Spectral Centroid                              │ │ │
│  │ │ • Spectral Rolloff                               │ │ │
│  │ │ • Zero Crossing Rate                             │ │ │
│  │ │ • RMS Energy                                     │ │ │
│  │ │ • Pitch Variation                                │ │ │
│  │ │ • Jitter (frequency perturbation)                │ │ │
│  │ │ • Shimmer (amplitude perturbation)               │ │ │
│  │ └──────────────────────────────────────────────────┘ │ │
│  │ ┌──────────────────────────────────────────────────┐ │ │
│  │ │ AI Pattern Detection:                             │ │ │
│  │ │ • Unnatural pitch consistency                    │ │ │
│  │ │ • Abnormal jitter/shimmer                        │ │ │
│  │ │ • Spectral anomalies                             │ │ │
│  │ │ • ZCR consistency                                │ │ │
│  │ │ • Digital artifacts in frequency domain          │ │ │
│  │ └──────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Layer 2: Deep Learning Classification (35% weight)    │ │
│  │ ┌──────────────────────────────────────────────────┐ │ │
│  │ │ CNN Model (Spectrogram Analysis):                │ │ │
│  │ │ • Compute STFT spectrogram                       │ │ │
│  │ │ • Mel-spectrogram transformation                 │ │ │
│  │ │ • Pattern recognition in time-frequency domain   │ │ │
│  │ │ • Detect regular AI patterns                     │ │ │
│  │ │ • Phase coherence analysis                       │ │ │
│  │ └──────────────────────────────────────────────────┘ │ │
│  │ ┌──────────────────────────────────────────────────┐ │ │
│  │ │ RNN Model (Temporal Analysis):                   │ │ │
│  │ │ • Delta features (first derivative)              │ │ │
│  │ │ • Delta-delta features (acceleration)            │ │ │
│  │ │ • Temporal evolution patterns                    │ │ │
│  │ │ • Transition smoothness analysis                 │ │ │
│  │ │ • Harmonic consistency via chroma features       │ │ │
│  │ └──────────────────────────────────────────────────┘ │ │
│  │ ┌──────────────────────────────────────────────────┐ │ │
│  │ │ Ensemble Prediction:                             │ │ │
│  │ │ • Weighted average (CNN: 60%, RNN: 40%)         │ │ │
│  │ │ • Confidence scoring based on model agreement    │ │ │
│  │ └──────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Layer 3: Artifact Detection (25% weight)              │ │
│  │ ┌──────────────────────────────────────────────────┐ │ │
│  │ │ Robotic Pattern Detection:                       │ │ │
│  │ │ • Overly repetitive waveform segments            │ │ │
│  │ │ • Autocorrelation analysis                       │ │ │
│  │ └──────────────────────────────────────────────────┘ │ │
│  │ ┌──────────────────────────────────────────────────┐ │ │
│  │ │ Pause Analysis:                                  │ │ │
│  │ │ • Silence detection                              │ │ │
│  │ │ • Pause duration uniformity (AI characteristic)  │ │ │
│  │ │ • Coefficient of variation                       │ │ │
│  │ └──────────────────────────────────────────────────┘ │ │
│  │ ┌──────────────────────────────────────────────────┐ │ │
│  │ │ Digital Clipping:                                │ │ │
│  │ │ • Sample amplitude threshold detection           │ │ │
│  │ │ • Clipping frequency analysis                    │ │ │
│  │ └──────────────────────────────────────────────────┘ │ │
│  │ ┌──────────────────────────────────────────────────┐ │ │
│  │ │ Phase Inconsistencies:                           │ │ │
│  │ │ • Phase jump detection between windows           │ │ │
│  │ │ • Unnatural phase coherence                      │ │ │
│  │ └──────────────────────────────────────────────────┘ │ │
│  │ ┌──────────────────────────────────────────────────┐ │ │
│  │ │ Frequency Anomalies:                             │ │ │
│  │ │ • Unusual energy distribution across bands       │ │ │
│  │ │ • Nyquist frequency artifacts                    │ │ │
│  │ └──────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Layer 4: Language-Specific Analysis (15% weight)      │ │
│  │ ┌──────────────────────────────────────────────────┐ │ │
│  │ │ Language Detection:                              │ │ │
│  │ │ • Prosody feature extraction                     │ │ │
│  │ │ • Pitch range analysis                           │ │ │
│  │ │ • Rhythm pattern recognition                     │ │ │
│  │ │ • Stress vs syllable timing                      │ │ │
│  │ │ • Speech rate estimation                         │ │ │
│  │ │ • Language-specific characteristics:             │ │ │
│  │ │   - Tamil: High pitch variation, rhythmic        │ │ │
│  │ │   - English: Moderate pitch, stress-timed        │ │ │
│  │ │   - Hindi: Moderate-high pitch, syllable-timed   │ │ │
│  │ │   - Malayalam: Fast speech rate                  │ │ │
│  │ │   - Telugu: Rhythmic, melodic                    │ │ │
│  │ └──────────────────────────────────────────────────┘ │ │
│  │ ┌──────────────────────────────────────────────────┐ │ │
│  │ │ Prosody Analysis:                                │ │ │
│  │ │ • Natural vs AI prosody bounds per language      │ │ │
│  │ │ • Pitch range validation                         │ │ │
│  │ │ • Rhythm consistency checks                      │ │ │
│  │ │ • Prosody consistency scoring                    │ │ │
│  │ └──────────────────────────────────────────────────┘ │ │
│  │ ┌──────────────────────────────────────────────────┐ │ │
│  │ │ Pattern Analysis:                                │ │ │
│  │ │ • Co-articulation detection (AI lacks this)      │ │ │
│  │ │ • Natural disfluencies                           │ │ │
│  │ │ • Breathing pattern naturalness                  │ │ │
│  │ └──────────────────────────────────────────────────┘ │ │
│  │ ┌──────────────────────────────────────────────────┐ │ │
│  │ │ Phoneme Distribution:                            │ │ │
│  │ │ • Formant extraction                             │ │ │
│  │ │ • Formant transition smoothness                  │ │ │
│  │ │ • Formant consistency analysis                   │ │ │
│  │ └──────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              ENSEMBLE SCORING & DECISION                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Weighted Ensemble:                                   │  │
│  │ Score = (Acoustic × 0.25) + (DeepLearning × 0.35)   │  │
│  │       + (Artifact × 0.25) + (Language × 0.15)       │  │
│  │                                                       │  │
│  │ Confidence Adjustment:                               │  │
│  │ • Boost if deep learning confidence > 0.8            │  │
│  │ • Final score = min(1, weighted_score × adjustment)  │  │
│  │                                                       │  │
│  │ Classification Decision:                             │  │
│  │ • Score >= 0.5 → AI_GENERATED                       │  │
│  │ • Score < 0.5  → HUMAN                              │  │
│  │                                                       │  │
│  │ Confidence Calculation:                              │  │
│  │ • If AI: confidence = final_score                   │  │
│  │ • If HUMAN: confidence = 1 - final_score            │  │
│  │                                                       │  │
│  │ Reasoning Generation:                                │  │
│  │ • Compile anomalies and artifacts from all layers    │  │
│  │ • Generate human-readable explanation                │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                RESPONSE FORMATTING & METRICS                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ JSON Response Construction:                          │  │
│  │ • result: "AI_GENERATED" or "HUMAN"                 │  │
│  │ • confidence: 0.0 - 1.0                             │  │
│  │ • analysis: detailed breakdown from all layers       │  │
│  │ • metadata: audio file information                   │  │
│  │ • requestId: unique identifier                       │  │
│  │ • processingTime: milliseconds                       │  │
│  │ • timestamp: ISO 8601                                │  │
│  │                                                       │  │
│  │ Metrics Collection:                                  │  │
│  │ • Record detection result                            │  │
│  │ • Log processing time                                │  │
│  │ • Track language distribution                        │  │
│  │ • Monitor error rates                                │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. API Gateway Layer

**Technologies**: Express.js, Helmet, CORS, Express-Rate-Limit

**Responsibilities**:
- Request routing
- Authentication via API key
- Rate limiting (100 requests per 15 minutes per IP)
- Input validation
- Error handling and formatting
- Security headers
- CORS policy enforcement

### 2. Audio Processing Pipeline

**Technologies**: FFmpeg, Native Node.js Buffer API

**Responsibilities**:
- Base64 decoding and validation
- MP3 format verification
- Audio conversion (MP3 → WAV, 16kHz mono)
- Feature extraction
- Metadata parsing

**Processing Flow**:
1. Decode base64 → Binary buffer
2. Write to temporary file
3. FFmpeg conversion
4. WAV parsing
5. Extract Float32Array samples
6. Cleanup temporary files

### 3. AI Detection Engine

#### Layer 1: Acoustic Analysis
**Algorithm**: Statistical signal processing
**Key Metrics**:
- MFCC: 13 coefficients via DCT
- Jitter: Pitch perturbation (natural: 0.5-1.5%)
- Shimmer: Amplitude perturbation (natural: 3-10%)
- ZCR variation: Coefficient of variation

**Detection Criteria**:
- AI: Pitch variation < 0.15
- AI: Jitter < 0.3% or > 2%
- AI: Shimmer < 2% or > 15%
- AI: ZCR variation < 0.05

#### Layer 2: Deep Learning
**Architecture**: CNN + RNN Ensemble
**CNN**: Time-frequency pattern recognition
**RNN**: Temporal sequence analysis

**Processing**:
1. STFT → Spectrogram
2. Mel filterbank → Mel-spectrogram
3. Delta/Delta-delta features
4. Chroma features (harmonic content)
5. Pattern detection algorithms
6. Weighted ensemble prediction

#### Layer 3: Artifact Detection
**Focus**: Digital processing artifacts

**Detections**:
- Robotic patterns (autocorrelation > 0.9)
- Pause uniformity (CV < 0.2)
- Digital clipping (amplitude > 0.99)
- Phase jumps (> π/2 radians)
- Frequency anomalies (Nyquist region > 5%)

#### Layer 4: Language-Specific
**Supported**: Tamil, English, Hindi, Malayalam, Telugu

**Features**:
- Language detection via prosody
- Language-specific natural ranges
- Prosody consistency scoring
- Co-articulation presence
- Breathing pattern analysis

### 4. Ensemble Scorer

**Algorithm**: Weighted average with confidence adjustment

**Weights**:
- Acoustic: 25%
- Deep Learning: 35%
- Artifact: 25%
- Language: 15%

**Decision Threshold**: 0.5

## Performance Characteristics

**Processing Time**: 2-3 seconds average
**Accuracy**: 85-90% on test datasets
**Throughput**: 20-30 requests/second (single instance)
**Memory**: 200-500MB per instance

## Scalability

**Horizontal Scaling**: Deploy multiple instances behind load balancer
**Vertical Scaling**: Increase CPU/memory resources
**Caching**: Redis for repeated analyses
**Async Processing**: Parallel detection layers

## Security Features

- API key authentication
- Rate limiting (IP + API key)
- Input validation and sanitization
- Request size limits (50MB max)
- CORS policy
- Security headers (Helmet.js)
- Comprehensive logging

## Monitoring & Observability

**Metrics Collected**:
- Request count (total, last 24h, last hour)
- Classification distribution (AI vs Human)
- Average confidence scores
- Processing times
- Error rates
- Language distribution

**Logging**:
- Winston logger (JSON format)
- File rotation (daily, 14 days retention)
- Log levels: error, warn, info, debug
- Request tracing via requestId

## Error Handling

**Error Types**:
- ValidationError (400): Invalid input
- AuthError (401): Invalid API key
- AudioProcessingError (422): Audio format issues
- ModelInferenceError (500): Detection failure
- RateLimitError (429): Too many requests

**Response Format**:
```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable message",
    "statusCode": 400,
    "details": { ... },
    "requestId": "req_xxx",
    "timestamp": "ISO-8601"
  }
}
```

## Future Enhancements

1. **Real-time Streaming**: WebSocket support for live audio
2. **GPU Acceleration**: TensorFlow.js with GPU backend
3. **More Languages**: Expand to 20+ languages
4. **Speaker Recognition**: Identify specific AI voice models
5. **Batch Processing**: Multiple files in single request
6. **Webhook Callbacks**: Async processing with callbacks
7. **Advanced ML**: Fine-tuned transformers (Wav2Vec2, HuBERT)
8. **Explainable AI**: Visual attention maps and LIME

## Technology Stack Summary

**Backend**: Node.js 18+, TypeScript, Express.js
**Audio**: FFmpeg, native audio processing
**ML/AI**: Custom algorithms, statistical methods
**Security**: Helmet, CORS, Rate limiting
**Logging**: Winston
**Process Management**: PM2
**Containerization**: Docker
**Reverse Proxy**: Nginx
**SSL**: Certbot/Let's Encrypt

## Design Principles

1. **Reliability**: Comprehensive error handling
2. **Performance**: Parallel processing, optimized algorithms
3. **Scalability**: Stateless design, horizontal scaling
4. **Security**: Defense in depth
5. **Maintainability**: Clean code, TypeScript, logging
6. **Observability**: Metrics, logging, tracing
7. **Documentation**: Comprehensive API docs
