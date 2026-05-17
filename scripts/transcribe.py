#!/usr/bin/env python3
"""
Transcribe a WAV audio file using NVIDIA Nemotron ASR (hosted on build.nvidia.com).

Usage:
    export NVIDIA_API_KEY=nvapi-...
    python transcribe.py audio.wav
    python transcribe.py audio.wav --language en-US --output transcript.txt

Requires: pip install nvidia-riva-client
"""
import argparse
import os
import sys

import riva.client


def build_auth(api_key: str) -> riva.client.Auth:
    return riva.client.Auth(
        uri="grpc.nvcf.nvidia.com:443",
        use_ssl=True,
        metadata_args=[
            ["function-id", "bb0837de-8c7b-481f-9ec8-ef5663e9c1fa"],
            ["authorization", f"Bearer {api_key}"],
        ],
    )


def transcribe(audio_path: str, language: str = "en-US") -> str:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        sys.exit("ERROR: Set NVIDIA_API_KEY environment variable")

    auth = build_auth(api_key)
    asr_service = riva.client.ASRService(auth)

    with open(audio_path, "rb") as f:
        audio = f.read()

    config = riva.client.RecognitionConfig(
        language_code=language,
        max_alternatives=1,
        enable_automatic_punctuation=True,
        enable_word_time_offsets=False,
    )

    response = asr_service.offline_recognize(audio, config)
    if not response.results:
        return ""

    return response.results[0].alternatives[0].transcript


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio with NVIDIA Nemotron ASR")
    parser.add_argument("audio", help="Path to WAV audio file (16 kHz mono recommended)")
    parser.add_argument("--language", default="en-US", help="Language code (default: en-US)")
    parser.add_argument("--output", "-o", help="Write transcript to file instead of stdout")
    args = parser.parse_args()

    if not os.path.isfile(args.audio):
        sys.exit(f"ERROR: File not found: {args.audio}")

    text = transcribe(args.audio, language=args.language)

    if args.output:
        with open(args.output, "w") as f:
            f.write(text + "\n")
        print(f"Transcript written to {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
