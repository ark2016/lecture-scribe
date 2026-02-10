"""Keyframe tracking and event classification — Phase 2 stub."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrackedKeyframe:
    id: str
    timestamp_first: float
    timestamp_last: float
    frame: np.ndarray
    history: list[tuple[float, np.ndarray]] = field(default_factory=list)


@dataclass
class VisualEvent:
    timestamp: float
    event_type: str  # "new_content" | "update_existing" | "return_to_previous"
    keyframe_id: str
    frame: np.ndarray
    previous_frame: np.ndarray | None = None


class KeyframeTracker:
    """Buffer-based keyframe tracker — stub for Phase 2.

    In Phase 1 every keyframe is treated as ``new_content``.
    """

    def __init__(
        self,
        buffer_size: int = 50,
        match_threshold: float = 0.80,
        update_threshold: float = 0.90,
    ):
        self.buffer_size = buffer_size
        self.match_threshold = match_threshold
        self.update_threshold = update_threshold
        self._counter = 0

    def process_event(self, timestamp: float, frame: np.ndarray) -> VisualEvent:
        """Phase 1: always returns new_content."""
        self._counter += 1
        kf_id = f"kf_{self._counter:04d}"
        return VisualEvent(
            timestamp=timestamp,
            event_type="new_content",
            keyframe_id=kf_id,
            frame=frame,
            previous_frame=None,
        )
