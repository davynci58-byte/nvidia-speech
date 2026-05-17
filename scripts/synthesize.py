#!/usr/bin/env python3
"""
Synthesize speech from text using NVIDIA Magpie TTS Multilingual (hosted on build.nvidia.com).

Usage:
    export NVIDIA_API_KEY=nvapi-...
    python synthesize.py "Hello world" output.wav
    python synthesize.py "Hola mundo" output.wav --language es-ES --voice "Magpie-Multilingual.EN-US.Sofia"

Requires: pip install nvidia-riva-client
"""
import argparse
import os
import sys
import wave

import riva.client
from riva.client.proto.riva_audio_pb2 import AudioEncoding

# Available voices
VOICES = [
    "Magpie-Multilingual.EN-US.Sofia",
    "Magpie-Multilingual.EN-US.Aria",
    "Magpie-Multilingual.EN-US.Jason",
    "Magpie-Multilingual.EN-US.Leo",
    "Magpie-Multilingual.EN-US.JohnVanStan",
]

LANGUAGE_MAP = {
    "en-US": "en-US",
    "es-ES": "es-ES",
    "de-DE": "de-DE",
    "fr-FR": "fr-FR",
    "vi-VN": "vi-VN",
    "it-IT": "it-IT",
    "zh-CN": "zh-CN",
    "hi-IN": "hi-IN",
    "ja-JP": "ja-JP",
}


def build_auth(api_key: str) -> riva.client.Auth:
    return riva.client.Auth(
        uri="grpc.nvcf.nvidia.com:443",
        use_ssl=True,
        metadata_args=[
            ["function-id", "877104f7-e885-42b9-8de8-f6e4c6303969"],
            ["authorization", f"Bearer {api_key}"],
        ],
    )


def synthesize(
    text: str,
    output_path: str,
    voice: str = "Magpie-Multilingual.EN-US.Sofia",
    language: str = "en-US",
    sample_rate: int = 22050,
):
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        sys.exit("ERROR: Set NVIDIA_API_KEY environment variable")

    if language not in LANGUAGE_MAP:
        sys.exit(f"ERROR: Unsupported language '{language}'. Supported: {', '.join(LANGUAGE_MAP.keys())}")

    auth = build_auth(api_key)
    service = riva.client.SpeechSynthesisService(auth)

    resp = service.synthesize(
        text,
        voice,
        language,
        sample_rate_hz=sample_rate,
        encoding=AudioEncoding.LINEAR_PCM,
    )

    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframesraw(resp.audio)

    print(f"Audio saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Synthesize speech with NVIDIA Magpie TTS")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("output", help="Output WAV file path")
    parser.add_argument(
        "--voice", "-v",
        default="Magpie-Multilingual.EN-US.Sofia",
        help=f"Voice name. Available: {', '.join(VOICES)} (default: Sofia)",
    )
    parser.add_argument(
        "--language", "-l",
        default="en-US",
        help=f"Language code. Supported: {', '.join(LANGUAGE_MAP.keys())} (default: en-US)",
    )
    parser.add_argument("--sample-rate", type=int, default=22050, help="Sample rate in Hz (default: 22050)")
    args = parser.parse_args()

    synthesize(args.text, args.output, args.voice, args.language, args.sample_rate)


if __name__ == "__main__":
    main()
