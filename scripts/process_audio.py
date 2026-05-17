#!/usr/bin/env python3
"""
Process incoming audio for transcription using NVIDIA Nemotron ASR Streaming.
Converts any audio format to 16kHz mono WAV, then transcribes via streaming RPC.
Handles long audio by splitting into 3-second chunks.

Usage:
    export NVIDIA_API_KEY=nvapi-...
    python process_audio.py path/to/voice_note.ogg
    python process_audio.py path/to/audio.wav --language en-US
    python process_audio.py path/to/audio.mp3 -o transcript.txt
"""
import argparse
import os
import subprocess
import sys
import tempfile
import wave

import riva.client

NVIDIA_FUNCTION_ID = "bb0837de-8c7b-481f-9ec8-ef5663e9c1fa"
ASR_SAMPLE_RATE = 16000
CHUNK_DURATION_S = 3       # split long audio into 3s segments
OVERLAP_S = 1              # 1s overlap between segments
TRAILING_SILENCE_S = 0.5   # silence appended to flush final results


def build_auth(api_key: str) -> riva.client.Auth:
    return riva.client.Auth(
        uri="grpc.nvcf.nvidia.com:443",
        use_ssl=True,
        metadata_args=[
            ["function-id", NVIDIA_FUNCTION_ID],
            ["authorization", f"Bearer {api_key}"],
        ],
    )


def convert_audio(input_path: str, output_path: str) -> None:
    """Convert any audio to 16 kHz mono WAV using ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ar", str(ASR_SAMPLE_RATE),
        "-ac", "1",
        "-sample_fmt", "s16",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")


def transcribe_segment(audio_segment: bytes, asr_service) -> str:
    """Transcribe a single audio segment via streaming ASR."""
    silence = b'\x00\x00' * int(ASR_SAMPLE_RATE * TRAILING_SILENCE_S)
    data = audio_segment + silence

    config = riva.client.RecognitionConfig(
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        sample_rate_hertz=ASR_SAMPLE_RATE,
        language_code="en-US",
        max_alternatives=1,
        enable_automatic_punctuation=True,
    )
    streaming_config = riva.client.StreamingRecognitionConfig(
        config=config,
        interim_results=False,
    )

    chunk_bytes = 3200  # 100ms
    chunks = [data[i:i + chunk_bytes] for i in range(0, len(data), chunk_bytes)]

    texts = []
    for response in asr_service.streaming_response_generator(chunks, streaming_config):
        for result in response.results:
            if result.is_final:
                for alt in result.alternatives:
                    t = alt.transcript.strip()
                    if t:
                        texts.append(t)
    return " ".join(texts)


def transcribe_audio(audio_path: str) -> str:
    """Transcribe a WAV file, splitting into chunks if longer than CHUNK_DURATION_S."""
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        sys.exit("ERROR: Set NVIDIA_API_KEY environment variable")

    auth = build_auth(api_key)
    asr_service = riva.client.ASRService(auth)

    with wave.open(audio_path, "rb") as wf:
        sample_rate = wf.getframerate()
        raw_audio = wf.readframes(wf.getnframes())
        duration = wf.getnframes() / sample_rate

    # If short enough, transcribe in one shot
    if duration <= CHUNK_DURATION_S + 0.5:
        return transcribe_segment(raw_audio, asr_service)

    # Split into overlapping chunks
    chunk_frames = CHUNK_DURATION_S * sample_rate
    overlap_frames = OVERLAP_S * sample_rate
    step_frames = chunk_frames - overlap_frames
    bytes_per_frame = 2  # 16-bit mono
    step_bytes = step_frames * bytes_per_frame
    chunk_bytes = chunk_frames * bytes_per_frame

    all_text = []
    for i in range(0, len(raw_audio), step_bytes):
        seg = raw_audio[i:i + chunk_bytes]
        if len(seg) < sample_rate * bytes_per_frame:  # skip < 1s leftovers
            break
        seg_text = transcribe_segment(seg, asr_service)
        if seg_text:
            all_text.append(seg_text)

    return " ".join(all_text)


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio using NVIDIA Nemotron ASR Streaming"
    )
    parser.add_argument("audio", help="Path to audio file (WAV, OGG, MP3, etc.)")
    parser.add_argument("--output", "-o", help="Write transcript to file instead of stdout")
    args = parser.parse_args()

    if not os.path.isfile(args.audio):
        sys.exit(f"ERROR: File not found: {args.audio}")

    # Convert to 16kHz mono WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        converted_path = tmp.name

    try:
        convert_audio(args.audio, converted_path)
    except RuntimeError as e:
        if os.path.exists(converted_path):
            os.unlink(converted_path)
        sys.exit(f"ERROR: {e}")

    text = transcribe_audio(converted_path)

    if os.path.exists(converted_path):
        os.unlink(converted_path)

    if args.output:
        with open(args.output, "w") as f:
            f.write(text + "\n")
        print(f"Transcript written to {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
