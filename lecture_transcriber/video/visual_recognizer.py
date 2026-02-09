"""Recognize visual content from keyframes using Mistral Vision."""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from lecture_transcriber.utils.api_client import MistralClient

log = logging.getLogger(__name__)


PROMPT_NEW_CONTENT = """\
Ты анализируешь кадр из видеолекции. На экране — рукописный текст и/или формулы,
написанные в графическом редакторе (Paint).

Задача:
1. Запиши весь видимый рукописный текст (русский язык)
2. Все математические формулы запиши в LaTeX (обёрнуты в $..$ или $$..$$)
3. Если есть схемы или диаграммы — опиши их структуру и содержание
4. Сохраняй порядок сверху вниз, слева направо

Формат ответа — JSON:
{
    "text": "распознанный текст с формулами в LaTeX",
    "latex_blocks": ["\\\\frac{a}{b}", ...],
    "diagrams": ["описание схемы 1", ...]
}
"""


@dataclass
class VisualSegment:
    keyframe_id: str
    timestamp: float
    event_type: str
    text: str
    latex: str
    diagram_description: str
    raw_response: str


def recognize_visual_content(
    events: list[tuple[str, float, str, np.ndarray]],
    client: MistralClient,
    model: str = "mistral-small-latest",
    save_keyframes_dir: str | Path | None = None,
) -> list[VisualSegment]:
    """Recognize content for a list of (keyframe_id, timestamp, event_type, frame) tuples.

    Phase 1: only uses PROMPT_NEW_CONTENT for every event.
    """
    segments: list[VisualSegment] = []

    for kf_id, timestamp, event_type, frame in events:
        img_path = _save_frame(frame, kf_id, save_keyframes_dir)
        try:
            raw = client.analyze_image([img_path], PROMPT_NEW_CONTENT, model=model)
        except Exception:
            log.exception("Vision API failed for %s at %.2fs", kf_id, timestamp)
            continue

        parsed = _parse_response(raw)
        segments.append(
            VisualSegment(
                keyframe_id=kf_id,
                timestamp=timestamp,
                event_type=event_type,
                text=parsed.get("text", ""),
                latex="; ".join(parsed.get("latex_blocks", [])),
                diagram_description="; ".join(parsed.get("diagrams", [])),
                raw_response=raw,
            )
        )
        log.info("Recognized %s at %.2fs", kf_id, timestamp)

    return segments


_MAX_VISION_DIM = 1024  # downscale to fit within this; keeps tokens reasonable


def _save_frame(
    frame: np.ndarray,
    name: str,
    directory: str | Path | None,
) -> Path:
    frame = _downscale(frame, _MAX_VISION_DIM)
    if directory is not None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{name}.png"
    else:
        path = Path(tempfile.mktemp(suffix=".png"))
    cv2.imwrite(str(path), frame)
    return path


def _downscale(frame: np.ndarray, max_dim: int) -> np.ndarray:
    """Downscale frame so the longest side is at most *max_dim* pixels."""
    h, w = frame.shape[:2]
    if max(h, w) <= max_dim:
        return frame
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _parse_response(raw: str) -> dict:
    """Best-effort JSON parse of the VLM response."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try extracting JSON block from markdown fences
        import re
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
    return {"text": raw, "latex_blocks": [], "diagrams": []}
