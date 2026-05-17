# Magpie TTS Multilingual — Voice Reference

**Function ID:** `877104f7-e885-42b9-8de8-f6e4c6303969`
**Endpoint:** `grpc.nvcf.nvidia.com:443`
**Model:** Transformer encoder-decoder, 357M parameters

## Voices

All 5 speakers support all 9 languages.

| Voice Name | Gender | Notes |
|---|---|---|
| `Magpie-Multilingual.EN-US.Sofia` | Female | Default, natural tone |
| `Magpie-Multilingual.EN-US.Aria` | Female | Clear, expressive |
| `Magpie-Multilingual.EN-US.Jason` | Male | Deep, professional |
| `Magpie-Multilingual.EN-US.Leo` | Male | Warm, friendly |
| `Magpie-Multilingual.EN-US.JohnVanStan` | Male | Public domain (LibriVox) |

## Languages

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

## Audio Output

- Default sample rate: 22050 Hz
- Encoding: LINEAR_PCM (16-bit signed, mono WAV)
- Also supports OGG OPUS via `encoding=AudioEncoding.OGGOPUS`

## Using the CLI Script

```bash
# Install Riva client
pip install nvidia-riva-client

# Basic synthesis
python scripts/synthesize.py "Hello world" hello.wav

# Different voice and language
python scripts/synthesize.py "Bonjour le monde" bonjour.wav \
    --voice "Magpie-Multilingual.EN-US.Aria" \
    --language fr-FR

# Higher quality
python scripts/synthesize.py "Text" output.wav --sample-rate 44100
```
