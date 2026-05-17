#!/usr/bin/env python3
"""Transcribe a WAV file using NVIDIA Nemotron ASR Streaming."""
import argparse
import os
import sys
import wave

import riva.client

FUNCTION_ID = "bb0837de-8c7b-481f-9ec8-ef5663e9c1fa"
RATE = 16000


def build_auth(api_key):
    return riva.client.Auth(
        uri="grpc.nvcf.nvidia.com:443", use_ssl=True,
        metadata_args=[
            ["function-id", FUNCTION_ID],
            ["authorization", f"Bearer {api_key}"],
        ],
    )


def transcribe(audio_path, language="en-US"):
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        sys.exit("ERROR: Set NVIDIA_API_KEY")

    with wave.open(audio_path, "rb") as wf:
        sr = wf.getframerate()
        sw = wf.getsampwidth()
        nc = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())

    # Add trailing silence
    raw = raw + b'\x00\x00' * int(RATE * 0.5)

    config = riva.client.RecognitionConfig(
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        sample_rate_hertz=sr, language_code=language,
        max_alternatives=1, enable_automatic_punctuation=True,
    )
    sc = riva.client.StreamingRecognitionConfig(config=config, interim_results=False)

    chunk = 5120
    chunks = [raw[i:i + chunk] for i in range(0, len(raw), chunk)]

    asr = riva.client.ASRService(build_auth(api_key))
    out = []
    for resp in asr.streaming_response_generator(chunks, sc):
        for r in resp.results:
            if r.is_final:
                for a in r.alternatives:
                    t = a.transcript.strip()
                    if t:
                        out.append(t)
    return " ".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("audio")
    p.add_argument("--language", default="en-US")
    p.add_argument("-o", "--output")
    args = p.parse_args()
    if not os.path.isfile(args.audio):
        sys.exit(f"ERROR: file not found: {args.audio}")
    t = transcribe(args.audio, args.language)
    if args.output:
        with open(args.output, "w") as f:
            f.write(t + "\n")
    else:
        print(t)


if __name__ == "__main__":
    main()
