# Training Data Structure

## Overview
This folder contains audio samples for training the AI voice detection models.
The models are **language-agnostic** - they detect AI vs Human based on acoustic patterns, 
not language content. However, organizing by language helps ensure balanced training.

## Folder Structure

```
training_data/
├── human/                    # Real human voice recordings
│   ├── tamil/               # Tamil human voice samples
│   ├── english/             # English human voice samples
│   ├── hindi/               # Hindi human voice samples
│   ├── malayalam/           # Malayalam human voice samples
│   └── telugu/              # Telugu human voice samples
│
├── ai/                       # AI-generated audio samples
│   ├── tamil/               # Tamil AI-generated samples
│   ├── english/             # English AI-generated samples
│   ├── hindi/               # Hindi AI-generated samples
│   ├── malayalam/           # Malayalam AI-generated samples
│   └── telugu/              # Telugu AI-generated samples
│
└── README.md                 # This file
```

## How to Collect Training Data

### Human Voice Samples (Label: 0)
Collect real human voice recordings:
- Record yourself speaking
- Use voice recordings from phone calls
- Use publicly available speech datasets (Common Voice, etc.)
- Record friends/family speaking in different languages

**Requirements:**
- Duration: 3-10 seconds per sample
- Format: WAV, MP3, FLAC, OGG, M4A
- Quality: Clear audio without too much background noise
- Aim for: 50-100 samples per language

### AI-Generated Samples (Label: 1)
Generate AI voice samples using:
- **ElevenLabs** (https://elevenlabs.io)
- **Google Cloud TTS** 
- **Amazon Polly**
- **OpenAI TTS**
- **Microsoft Azure Speech**

**Steps:**
1. Go to any TTS service
2. Type text in the target language
3. Generate and download the audio
4. Save to `ai/{language}/` folder

**Tips:**
- Use various TTS voices (male, female, different styles)
- Use different text content
- Aim for: 50-100 samples per language

## Sample Counts Target

For 90% accuracy, aim for:

| Language   | Human Samples | AI Samples |
|------------|---------------|------------|
| Tamil      | 50-100        | 50-100     |
| English    | 50-100        | 50-100     |
| Hindi      | 50-100        | 50-100     |
| Malayalam  | 50-100        | 50-100     |
| Telugu     | 50-100        | 50-100     |
| **Total**  | **250-500**   | **250-500** |

More samples = better accuracy!

## File Naming Convention

Use descriptive names:
```
human/english/recording_001.wav
human/english/john_greeting.mp3
ai/english/elevenlabs_sample_01.mp3
ai/tamil/google_tts_news.wav
```

## Training Command

Once you have collected samples, run:

```bash
cd /home/rushikesh/Desktop/voice-detection-api/python
./venv/bin/python3 train_models.py --data-dir training_data --epochs 10
```

## Important Notes

1. **Balance is key**: Keep similar numbers of human and AI samples
2. **Diversity matters**: Use different speakers, TTS services, recording conditions
3. **Quality over quantity**: Clean samples are better than noisy ones
4. **Min 3 seconds**: Audio should be at least 3 seconds long
