# NVIDIA Speech NIM — OpenClaw Skill

Free, open-source speech AI skill for OpenClaw using NVIDIA's hosted NIM APIs on [build.nvidia.com](https://build.nvidia.com/).

- **🎙️ Speech-to-Text** — Nemotron ASR Streaming (600M params) — English transcription, punctuation, streaming support
- **🗣️ Text-to-Speech** — Magpie TTS Multilingual (357M params) — 9 languages, 5 voices

## Quick Start

```bash
# Install deps
pip install nvidia-riva-client

# Set your API key
export NVIDIA_API_KEY=nvapi-your-key-here

# Transcribe audio
python scripts/process_audio.py voice_note.ogg

# Synthesize speech
python scripts/synthesize.py "Hello world" output.wav
```

Get a free API key at [build.nvidia.com](https://build.nvidia.com/).

## Models

| Model | Type | Params | Function ID | Languages |
|---|---|---|---|---|
| Nemotron ASR Streaming | STT | 600M | `bb0837de...` | English |
| Magpie TTS Multilingual | TTS | 357M | `877104f7...` | 9 languages |

## WhatsApp / Chat Integration

This skill auto-detects and converts any audio format (OGG Opus, MP3, M4A, WAV) to 16 kHz mono WAV for ASR processing. Perfect for voice notes from messaging platforms.

## Scripts

- **`process_audio.py`** — Universal audio processor: auto-converts & transcribes
- **`transcribe.py`** — Transcribe WAV files with NVIDIA ASR
- **`synthesize.py`** — Generate speech from text with NVIDIA TTS

## License

Apache 2.0
