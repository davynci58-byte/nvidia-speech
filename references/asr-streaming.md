# Nemotron ASR Streaming — Reference

**Function ID:** `bb0837de-8c7b-481f-9ec8-ef5663e9c1fa`
**Endpoint:** `grpc.nvcf.nvidia.com:443`
**Model:** FastConformer-CacheAware-RNNT, 600M parameters

## Capabilities

- English-only transcription (en-US)
- Native punctuation & capitalization
- Word-level time offsets (optional)
- Configurable latency: 80ms, 160ms, 560ms, 1120ms chunk sizes
- Cache-aware streaming (no redundant computation between chunks)

## Streaming Recognition

For low-latency applications (voice agents, live captioning):

```python
import riva.client
from riva.client import StreamingRecognitionConfig, RecognitionConfig

auth = riva.client.Auth(
    uri="grpc.nvcf.nvidia.com:443",
    use_ssl=True,
    metadata_args=[
        ["function-id", "bb0837de-8c7b-481f-9ec8-ef5663e9c1fa"],
        ["authorization", "Bearer nvapi-YOUR_API_KEY"],
    ],
)
asr_service = riva.client.ASRService(auth)

# Configure streaming
config = RecognitionConfig(
    language_code="en-US",
    max_alternatives=1,
    enable_automatic_punctuation=True,
    enable_word_time_offsets=False,
)
stream_config = StreamingRecognitionConfig(interim_results=True, config=config)

# Start streaming
generator = asr_service.streaming_response_generator(stream_config)

# Send audio chunks (16 kHz mono PCM, ~320 bytes per 20ms frame)
# generator(audio_chunk) -> yields StreamingRecognitionResult
```

## Audio Requirements

| Property | Value |
|---|---|
| Sample rate | 16000 Hz (16 kHz) |
| Channels | 1 (mono) |
| Format | PCM 16-bit signed |
| Codec | WAV / raw PCM |

## Using the CLI Script

```bash
# Install Riva client
pip install nvidia-riva-client

# Transcribe a file
python scripts/transcribe.py meeting_recording.wav

# Save to file
python scripts/transcribe.py meeting_recording.wav -o transcript.txt
```
