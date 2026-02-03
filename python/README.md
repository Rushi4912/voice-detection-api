---
title: Voice Detection API
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# AI Voice Detection API

Detects whether audio is AI-generated or human voice.

## API Endpoint

```
POST /api/detect
Header: x-api-key: sk_test_123456789
Content-Type: application/json

{
  "language": "English",
  "audio_base64": "<base64 encoded audio>"
}
```

## Supported Languages
- English
- Hindi
- Tamil
- Telugu
- Malayalam
