"""Image utility helpers — SSIM, diff, region detection."""

from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2


def compute_ssim(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Compute SSIM between two frames (converted to grayscale)."""
    gray_a = _to_gray(frame_a)
    gray_b = _to_gray(frame_b)
    return float(ssim(gray_a, gray_b))


def compute_change_region(
    frame_a: np.ndarray, frame_b: np.ndarray, threshold: int = 30
) -> tuple[int, int, int, int] | None:
    """Return bounding box (y_min, y_max, x_min, x_max) of changed region, or None."""
    diff = cv2.absdiff(_to_gray(frame_a), _to_gray(frame_b))
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    return (y, y + h, x, x + w)


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame
