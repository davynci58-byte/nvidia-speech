#!/usr/bin/env python3
"""
Fast audio transcription using NVIDIA Canary (multilingual ASR).
Converts any audio to 16kHz mono WAV via ffmpeg, transcribes via offline API.

Fastest NVIDIA option: single API call, no chunking, multilingual.

Usage:
    export NVIDIA_API_KEY=nvapi-...
    python process_audio.py path/to/audio.ogg
"""
import argparse
import os
import subprocess
import sys
import tempfile

import riva.client

# Canary-1B (multilingual ASR, offline recognition)
FUNCTION_ID = "b0e8b4a5-217c-40b7-9b96-17d84e666317"
ASR_RATE = 16000

_service = None


def _get_service():
    global _service
    if _service is None:
        api_key = os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            sys.exit("ERROR: Set NVIDIA_API_KEY environment variable")
        auth = riva.client.Auth(
            uri="grpc.nvcf.nvidia.com:443",
            use_ssl=True,
            metadata_args=[
                ["function-id", FUNCTION_ID],
                ["authorization", f"Bearer {api_key}"],
            ],
        )
        _service = riva.client.ASRService(auth)
    return _service


def convert(input_path, output_path):
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path,
         "-ar", str(ASR_RATE), "-ac", "1",
         "-sample_fmt", "s16", "-loglevel", "error", output_path],
        check=True, capture_output=True,
    )


def transcribe(audio_path):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav = f.name
    convert(audio_path, wav)
    with open(wav, "rb") as f:
        data = f.read()
    os.unlink(wav)

    config = riva.client.RecognitionConfig(
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        sample_rate_hertz=ASR_RATE,
        language_code="en-US",
        max_alternatives=1,
        enable_automatic_punctuation=True,
    )

    resp = _get_service().offline_recognize(data, config)
    if not resp.results:
        return ""
    return resp.results[0].alternatives[0].transcript


def main():
    p = argparse.ArgumentParser()
    p.add_argument("audio", help="Audio file path")
    p.add_argument("-o", "--output", help="Output file path")
    args = p.parse_args()
    if not os.path.isfile(args.audio):
        sys.exit(f"ERROR: file not found: {args.audio}")
    t = transcribe(args.audio)
    if args.output:
        with open(args.output, "w") as f:
            f.write(t + "\n")
    else:
        print(t)


if __name__ == "__main__":
    main()
