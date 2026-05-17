#!/usr/bin/env python3
"""
Process incoming audio for transcription — convert to proper format then transcribe with NVIDIA Nemotron ASR.

This is the primary entry point for handling audio from messaging channels (WhatsApp, Telegram, etc.).

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

import riva.client


NVIDIA_FUNCTION_ID = "bb0837de-8c7b-481f-9ec8-ef5663e9c1fa"
ASR_SAMPLE_RATE = 16000  # Hz — what the ASR model expects


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
    """
    Convert any audio file to 16 kHz mono WAV for the ASR model.
    Uses ffmpeg for conversion.
    """
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
        sys.exit(f"ERROR: ffmpeg conversion failed:\n{result.stderr}")


def transcribe(audio_path: str, language: str = "en-US") -> str:
    """Transcribe a WAV file using NVIDIA Nemotron ASR."""
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
    parser = argparse.ArgumentParser(description="Transcribe audio using NVIDIA Nemotron ASR")
    parser.add_argument("audio", help="Path to audio file (WAV, OGG, MP3, etc.)")
    parser.add_argument("--language", "-l", default="en-US", help="Language code (default: en-US)")
    parser.add_argument("--output", "-o", help="Write transcript to file instead of stdout")
    args = parser.parse_args()

    if not os.path.isfile(args.audio):
        sys.exit(f"ERROR: File not found: {args.audio}")

    audio_path = args.audio

    # Convert to proper WAV format if needed
    _, ext = os.path.splitext(audio_path)
    needs_conversion = ext.lower() not in (".wav",)

    if needs_conversion:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            converted_path = tmp.name
        try:
            convert_audio(audio_path, converted_path)
            audio_path = converted_path
        except Exception as e:
            # Clean up the temp file on error
            if os.path.exists(converted_path):
                os.unlink(converted_path)
            sys.exit(f"ERROR: Audio conversion failed: {e}")
    else:
        # Still resample to ensure 16kHz mono
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            converted_path = tmp.name
        try:
            convert_audio(audio_path, converted_path)
            audio_path = converted_path
        except Exception:
            # If conversion fails, try the original file directly
            pass

    try:
        text = transcribe(audio_path, language=args.language)

        if args.output:
            with open(args.output, "w") as f:
                f.write(text + "\n")
            print(f"Transcript written to {args.output}")
        else:
            print(text)
    finally:
        # Clean up temp file if we created one
        if needs_conversion or audio_path != args.audio:
            if os.path.exists(converted_path):
                os.unlink(converted_path)


if __name__ == "__main__":
    main()
