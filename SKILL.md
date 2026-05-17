---
name: nvidia-speech
description: Use NVIDIA's free hosted speech AI APIs on build.nvidia.com — Nemotron ASR for speech-to-text and Magpie TTS for text-to-speech. Trigger when the user requests transcription of audio, voice input, speech-to-text, text-to-speech, voice synthesis, or any task involving converting speech ↔ text using NVIDIA's free NIM API tier.
---

# NVIDIA Speech NIM

Free hosted API endpoints for **Nemotron ASR** (STT) and **Magpie TTS** using NVIDIA's NIM platform at build.nvidia.com. Requires an `nvapi-...` key from the [NVIDIA Developer Program](https://build.nvidia.com/).

## Quick Reference

| Model | Type | Params | Function ID | Mode |
|---|---|---|---|---|
| **Canary-1B** ★ | Speech-to-Text | 1B | `b0e8b4a5-217c-40b7-9b96-17d84e666317` | Offline (fastest) |
| Nemotron ASR Streaming | Speech-to-Text | 600M | `bb0837de-8c7b-481f-9ec8-ef5663e9c1fa` | Streaming |
| Magpie TTS Multilingual | Text-to-Speech | 357M | `877104f7-e885-42b9-8de8-f6e4c6303969` | gRPC |

**gRPC endpoint:** `grpc.nvcf.nvidia.com:443`

**As of 2026-05:** Canary-1B (multilingual) is the recommended STT model. It uses `offline_recognize` (no chunking, single API call) and handles diverse accents better than Nemotron.

### Available TTS Voices (Magpie Multilingual)

Speaker names usable with Magpie TTS — 9 languages (En, Es, De, Fr, Vi, It, Zh, Hi, Ja):

- `Magpie-Multilingual.EN-US.Sofia`
- `Magpie-Multilingual.EN-US.Aria`
- `Magpie-Multilingual.EN-US.Jason`
- `Magpie-Multilingual.EN-US.Leo`
- `Magpie-Multilingual.EN-US.JohnVanStan`

## Setup

```bash
pip install nvidia-riva-client
```

Set the API key as an environment variable:

```bash
export NVIDIA_API_KEY=nvapi-your-key-here
```

## Speech-to-Text (STT) — Nemotron ASR

### Offline transcription (single file)

```python
import riva.client

auth = riva.client.Auth(
    uri="grpc.nvcf.nvidia.com:443",
    use_ssl=True,
    metadata_args=[
        ["function-id", "bb0837de-8c7b-481f-9ec8-ef5663e9c1fa"],
        ["authorization", "Bearer nvapi-YOUR_API_KEY"],
    ],
)
asr_service = riva.client.ASRService(auth)

with open("audio.wav", "rb") as f:
    audio = f.read()

config = riva.client.RecognitionConfig(
    language_code="en-US",
    max_alternatives=1,
    enable_automatic_punctuation=True,
    enable_word_time_offsets=True,
)

response = asr_service.offline_recognize(audio, config)
print(response.results[0].alternatives[0].transcript)
```

### Streaming transcription

See `references/asr-streaming.md` for streaming recognition with microphone input or live audio chunks.

**Supported audio:** 16 kHz mono WAV/PCM. The model is English-only (en-US). Streaming supports chunk sizes of 80ms, 160ms, 560ms, 1120ms.

## Text-to-Speech (TTS) — Magpie TTS

### Basic speech synthesis

```python
import wave
import riva.client
from riva.client.proto.riva_audio_pb2 import AudioEncoding

auth = riva.client.Auth(
    uri="grpc.nvcf.nvidia.com:443",
    use_ssl=True,
    metadata_args=[
        ["function-id", "877104f7-e885-42b9-8de8-f6e4c6303969"],
        ["authorization", "Bearer nvapi-YOUR_API_KEY"],
    ],
)
service = riva.client.SpeechSynthesisService(auth)

sample_rate_hz = 22050
resp = service.synthesize(
    "Hello from the Magpie multilingual hosted API.",
    "Magpie-Multilingual.EN-US.Sofia",
    "en-US",
    sample_rate_hz=sample_rate_hz,
    encoding=AudioEncoding.LINEAR_PCM,
)

with wave.open("output.wav", "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sample_rate_hz)
    wf.writeframesraw(resp.audio)
```

### Language codes

| Language | Code |
|---|---|
| English | `en-US` |
| Spanish | `es-ES` |
| German | `de-DE` |
| French | `fr-FR` |
| Vietnamese | `vi-VN` |
| Italian | `it-IT` |
| Mandarin Chinese | `zh-CN` |
| Hindi | `hi-IN` |
| Japanese | `ja-JP` |

## Scripts

### `scripts/transcribe.py`
Transcribe a WAV audio file to text using the hosted Nemotron ASR API.

### `scripts/synthesize.py`
Generate a WAV audio file from text using the hosted Magpie TTS API.

### `scripts/process_audio.py`
**Primary entry point for incoming audio.** Auto-detects format, converts any audio (OGG, MP3, WAV, M4A, etc.) to 16 kHz mono WAV via ffmpeg, then transcribes with Nemotron ASR Streaming. Handles long audio by splitting into 3-second overlapping chunks (the streaming model has a maximum input length per session). Uses trailing silence to flush final results.

Both `transcribe.py` and `process_audio.py` read `NVIDIA_API_KEY` from the environment. Run any script with `--help` for options.

## WhatsApp / Messaging Integration

When a user sends audio via WhatsApp, OpenClaw provides:

- `{{MediaPath}}` — local temp path to the downloaded audio file
- `{{MediaUrl}}` — pseudo-URL for the inbound media

**In the main session**, when you receive inbound audio:

1. The file path is available in the message context
2. Call `process_audio.py` on it: `python3 skills/nvidia-speech/scripts/process_audio.py <path>`
3. Return the transcription as a reply

WhatsApp voice notes (PTT) arrive as `.ogg` (Opus). The script handles conversion automatically.

**Environment setup required:**

```bash
export NVIDIA_API_KEY=nvapi-...
```

## References

- `references/asr-streaming.md` — Details on streaming ASR, chunk sizes, microphone integration
- `references/tts-voices.md` — Full voice list, emotional styles, SSML support
