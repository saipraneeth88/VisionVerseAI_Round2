import os
import subprocess
from faster_whisper import WhisperModel

# -----------------------------------------
# Step 1: Extract Audio from Video
# -----------------------------------------
def extract_audio(video_path, video_hash):
    """
    Extracts audio from the given video file using FFmpeg.

    Parameters:
        video_path (str) : Path to the video file.
        video_hash (str) : Unique identifier for the video (used in output filename).

    Returns:
        str | None : Path to the extracted audio file (.wav) or None if failed.
    """
    try:
        output_audio_path = f"data/temp/audio/{video_hash}.wav"

        # FFmpeg command to extract mono 16kHz PCM audio
        command = [
            "ffmpeg", "-i", video_path,
            "-vn",                   # No video
            "-acodec", "pcm_s16le",  # WAV format
            "-ar", "16000",          # Sample rate 16kHz
            "-ac", "1",              # Mono channel
            "-y", output_audio_path  # Overwrite if exists
        ]

        subprocess.run(command, check=True, capture_output=True)
        return output_audio_path

    except Exception:
        return None


# -----------------------------------------
# Step 2: Speech-to-Text with faster-whisper
# -----------------------------------------
def run_audio_processing(video_path, video_hash):
    """
    Runs the audio processing pipeline:
    1. Extracts audio from video.
    2. Transcribes it using faster-whisper.

    Parameters:
        video_path (str) : Path to the video file.
        video_hash (str) : Unique identifier for the video.

    Returns:
        list[dict] : A list of transcript segments with timestamps and dialogue text.
    """
    print("-> Running Audio Pipeline (faster-whisper)...")

    # Extract the audio
    audio_path = extract_audio(video_path, video_hash)
    if not audio_path:
        return []

    try:
        # Load the Whisper model (Medium English version)
        model = WhisperModel("medium.en", device="cuda", compute_type="float16")

        # Transcribe the audio
        segments, _ = model.transcribe(audio_path)

        # Convert transcription results to structured events
        transcripts = [
            {"time": seg.start, "type": "dialogue", "content": seg.text.strip()}
            for seg in segments
        ]

    except Exception as e:
        print(f"   -> Whisper Error: {e}")
        transcripts = []

    finally:
        # Clean up temporary audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)

    return transcripts
