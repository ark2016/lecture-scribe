"""Transcribe audio using the Mistral Voxtral API."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from lecture_transcriber.utils.api_client import MistralClient

log = logging.getLogger(__name__)


@dataclass
class TranscriptWord:
    text: str
    start: float
    end: float


@dataclass
class TranscriptSegment:
    text: str
    start: float
    end: float
    words: list[TranscriptWord]


def transcribe(
    audio_path: str | Path,
    client: MistralClient,
    language: str = "ru",
    context_bias: list[str] | None = None,
) -> list[TranscriptSegment]:
    """Transcribe *audio_path* via Voxtral and return structured segments."""
    log.info("Transcribing %s", audio_path)
    raw = client.transcribe_audio(audio_path, language=language, context_bias=context_bias)

    segments: list[TranscriptSegment] = []
    for seg in raw.get("segments", []):
        words = [
            TranscriptWord(text=w["text"], start=w["start"], end=w["end"])
            for w in seg.get("words", [])
        ]
        segments.append(
            TranscriptSegment(
                text=seg["text"],
                start=seg["start"],
                end=seg["end"],
                words=words,
            )
        )

    log.info("Got %d transcript segments", len(segments))
    return segments
