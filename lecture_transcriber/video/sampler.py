"""Extract frames from video at a fixed FPS using OpenCV."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


def sample_frames(
    video_path: str | Path,
    fps: float = 2.0,
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (timestamp_seconds, frame) tuples at the requested *fps*."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30.0
    frame_interval = int(round(video_fps / fps))
    if frame_interval < 1:
        frame_interval = 1

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / video_fps
                yield (timestamp, frame)
            frame_idx += 1
    finally:
        cap.release()

    log.info("Sampled %d frames from %s at %.1f fps", frame_idx // frame_interval, video_path, fps)
