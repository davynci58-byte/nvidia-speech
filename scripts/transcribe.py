#!/usr/bin/env python3
"""
Transcribe a WAV audio file using NVIDIA Nemotron ASR Streaming (hosted on build.nvidia.com).

Usage:
    export NVIDIA_API_KEY=nvapi-...
    python transcribe.py audio.wav
    python transcribe.py audio.wav --language en-US --output transcript.txt

Requires: pip install nvidia-riva-client
"""
import argparse
import os
import sys
import wave

import riva.client


ASR_SAMPLE_RATE = 16000


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

    # Read the WAV file
    with wave.open(audio_path, "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw_audio = wf.readframes(wf.getnframes())

    # Streaming config
    config = riva.client.RecognitionConfig(
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        sample_rate_hertz=sample_rate,
        language_code=language,
        max_alternatives=1,
        enable_automatic_punctuation=True,
        enable_word_time_offsets=False,
    )
    streaming_config = riva.client.StreamingRecognitionConfig(
        config=config,
        interim_results=False,
    )

    # Break audio into chunks (~100ms each)
    chunk_bytes = 1600 * sampwidth * n_channels
    chunks = [raw_audio[i:i + chunk_bytes] for i in range(0, len(raw_audio), chunk_bytes)]

    # Stream
    full_transcript = ""
    for response in asr_service.streaming_response_generator(chunks, streaming_config):
        for result in response.results:
            if result.is_final:
                for alt in result.alternatives:
                    full_transcript += alt.transcript

    return full_transcript


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
