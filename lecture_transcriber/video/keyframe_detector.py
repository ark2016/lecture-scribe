"""Detect keyframes from a stream of sampled frames using SSIM + flicker filtering."""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass

import cv2
import numpy as np

from lecture_transcriber.utils.image import compute_ssim, compute_change_region

log = logging.getLogger(__name__)

_MIN_BRIGHTNESS = 40  # mean pixel value; TG dark screens are typically < 30


def _is_dark(frame: np.ndarray, threshold: int = _MIN_BRIGHTNESS) -> bool:
    """Return True if frame is too dark to contain useful lecture content."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return float(gray.mean()) < threshold


@dataclass
class KeyframeEvent:
    timestamp: float
    frame: np.ndarray
    change_type: str  # "significant" | "minor" | "flicker"
    change_region: tuple[int, int, int, int] | None
    ssim_vs_prev: float


def detect_keyframes(
    frames: Iterator[tuple[float, np.ndarray]],
    ssim_threshold_significant: float = 0.85,
    ssim_threshold_minor: float = 0.95,
    flicker_window: int = 3,
    flicker_recovery_threshold: float = 0.95,
) -> Iterator[KeyframeEvent]:
    """Yield KeyframeEvent for every real content change, filtering out flicker."""

    # We need a lookahead buffer to detect flicker (frames that revert quickly).
    buf: deque[tuple[float, np.ndarray]] = deque()
    last_stable: np.ndarray | None = None

    def _is_flicker(candidate_idx: int) -> bool:
        """Check whether buf[candidate_idx] is a transient flicker frame."""
        if last_stable is None:
            return False
        # Look at frames after the candidate to see if they revert to last_stable
        for offset in range(1, flicker_window):
            check_idx = candidate_idx + offset
            if check_idx < len(buf):
                recovery = compute_ssim(buf[check_idx][1], last_stable)
                if recovery >= flicker_recovery_threshold:
                    return True
        return False

    # Fill buffer first, then process with lookahead
    for ts, frame in frames:
        buf.append((ts, frame))

        # Keep buffer bounded — process when we have enough lookahead
        while len(buf) > flicker_window + 1:
            ts_cur, frame_cur = buf.popleft()

            if _is_dark(frame_cur):
                log.debug("Dark frame at %.2fs, skipping", ts_cur)
                continue

            if last_stable is None:
                last_stable = frame_cur
                continue

            score = compute_ssim(frame_cur, last_stable)

            if score >= ssim_threshold_minor:
                # No meaningful change
                continue

            # There's a change — check if it's flicker
            # We need to check if the *next* frames revert to last_stable
            is_flick = False
            for offset in range(min(flicker_window, len(buf))):
                recovery = compute_ssim(buf[offset][1], last_stable)
                if recovery >= flicker_recovery_threshold:
                    is_flick = True
                    break

            if is_flick:
                log.debug("Flicker at %.2fs (ssim=%.3f), skipping", ts_cur, score)
                continue

            # Real change
            change_type = "significant" if score < ssim_threshold_significant else "minor"
            region = compute_change_region(last_stable, frame_cur)

            yield KeyframeEvent(
                timestamp=ts_cur,
                frame=frame_cur,
                change_type=change_type,
                change_region=region,
                ssim_vs_prev=score,
            )
            last_stable = frame_cur

    # Drain remaining buffer
    while buf:
        ts_cur, frame_cur = buf.popleft()
        if _is_dark(frame_cur):
            log.debug("Dark frame at %.2fs, skipping", ts_cur)
            continue
        if last_stable is None:
            last_stable = frame_cur
            continue
        score = compute_ssim(frame_cur, last_stable)
        if score >= ssim_threshold_minor:
            continue
        change_type = "significant" if score < ssim_threshold_significant else "minor"
        region = compute_change_region(last_stable, frame_cur)
        yield KeyframeEvent(
            timestamp=ts_cur,
            frame=frame_cur,
            change_type=change_type,
            change_region=region,
            ssim_vs_prev=score,
        )
        last_stable = frame_cur
