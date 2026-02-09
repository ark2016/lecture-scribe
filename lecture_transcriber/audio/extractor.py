"""Extract audio from video using ffmpeg."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


def extract_audio(
    video_path: str | Path,
    output_path: str | Path | None = None,
    sample_rate: int = 16000,
) -> Path:
    """Run ffmpeg to extract mono WAV audio from *video_path*.

    Returns the path to the produced WAV file.
    """
    video_path = Path(video_path)
    if output_path is None:
        output_path = video_path.with_suffix(".wav")
    output_path = Path(output_path)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        str(output_path),
    ]

    log.info("Extracting audio: %s -> %s", video_path, output_path)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")

    log.info("Audio extracted: %s", output_path)
    return output_path
