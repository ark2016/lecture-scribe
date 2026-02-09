"""Synchronize audio transcript and visual segments by timestamp."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from lecture_transcriber.audio.transcriber import TranscriptSegment
from lecture_transcriber.video.visual_recognizer import VisualSegment

log = logging.getLogger(__name__)


@dataclass
class LectureBlock:
    timestamp_start: float
    timestamp_end: float
    audio_text: str
    visual_content: str
    latex_blocks: list[str] = field(default_factory=list)
    diagram_descriptions: list[str] = field(default_factory=list)
    block_type: str = "content"  # "content" | "revision" | "return"


def synchronize(
    transcript: list[TranscriptSegment],
    visual_segments: list[VisualSegment],
    merge_window: float = 5.0,
) -> list[LectureBlock]:
    """Align audio and visual segments into LectureBlocks (Phase 1: naive timestamp sync)."""

    blocks: list[LectureBlock] = []

    # Index audio segments that have been claimed
    used_audio: set[int] = set()

    for vs in visual_segments:
        # Gather audio segments within the merge window
        matched_audio: list[TranscriptSegment] = []
        for i, ts in enumerate(transcript):
            if i in used_audio:
                continue
            if abs(ts.start - vs.timestamp) <= merge_window:
                matched_audio.append(ts)
                used_audio.add(i)

        audio_text = " ".join(seg.text for seg in matched_audio)
        latex = [b for b in vs.latex.split("; ") if b] if vs.latex else []
        diagrams = [d for d in vs.diagram_description.split("; ") if d] if vs.diagram_description else []

        blocks.append(
            LectureBlock(
                timestamp_start=vs.timestamp,
                timestamp_end=vs.timestamp,
                audio_text=audio_text,
                visual_content=vs.text,
                latex_blocks=latex,
                diagram_descriptions=diagrams,
                block_type="content",
            )
        )

    # Remaining audio segments not matched to any visual
    for i, ts in enumerate(transcript):
        if i in used_audio:
            continue
        blocks.append(
            LectureBlock(
                timestamp_start=ts.start,
                timestamp_end=ts.end,
                audio_text=ts.text,
                visual_content="",
                block_type="content",
            )
        )

    blocks.sort(key=lambda b: b.timestamp_start)
    log.info("Synchronized into %d blocks", len(blocks))
    return blocks
