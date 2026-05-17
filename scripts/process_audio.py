#!/usr/bin/env python3
"""
Audio transcription using NVIDIA Nemotron ASR Streaming.
Converts any audio to 16kHz mono WAV, transcribes via streaming API.
Splits long audio into 3s chunks for the streaming model.

Usage:
    export NVIDIA_API_KEY=nvapi-...
    python process_audio.py path/to/audio.ogg
"""
import argparse
import os
import subprocess
import sys
import tempfile
import wave

import riva.client

FUNCTION_ID = "bb0837de-8c7b-481f-9ec8-ef5663e9c1fa"
RATE = 16000

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
         "-ar", str(RATE), "-ac", "1",
         "-sample_fmt", "s16", "-loglevel", "error", output_path],
        check=True, capture_output=True,
    )


def transcribe_segment(data: bytes) -> str:
    """Transcribe one PCM segment. Adds trailing silence to flush results."""
    data = data + b'\x00\x00' * int(RATE * 0.5)  # 0.5s silence

    config = riva.client.RecognitionConfig(
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        sample_rate_hertz=RATE,
        language_code="en-US",
        max_alternatives=1,
        enable_automatic_punctuation=True,
    )
    sc = riva.client.StreamingRecognitionConfig(config=config, interim_results=False)

    chunk = 5120
    chunks = [data[i:i + chunk] for i in range(0, len(data), chunk)]

    out = []
    for resp in _get_service().streaming_response_generator(chunks, sc):
        for r in resp.results:
            if r.is_final:
                for a in r.alternatives:
                    t = a.transcript.strip()
                    if t:
                        out.append(t)
    return " ".join(out)


def transcribe(audio_path: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav = f.name
    convert(audio_path, wav)
    with wave.open(wav, "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    os.unlink(wav)

    dur_s = len(raw) / (RATE * 2)

    # Short audio (<=5s): one shot
    if dur_s <= 5:
        return transcribe_segment(raw)

    # Long audio: split into 3s chunks with 1s overlap
    cf = RATE * 3
    sf = RATE * 2
    results = []
    for i in range(0, len(raw), sf * 2):
        seg = raw[i:i + cf * 2]
        if len(seg) < RATE * 2:
            break
        t = transcribe_segment(seg)
        if t:
            results.append(t)
    return " ".join(results)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("audio")
    p.add_argument("-o", "--output")
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
