"""Orchestrator — connects all pipeline stages."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np
from rich.progress import Progress

from lecture_transcriber.config import Config
from lecture_transcriber.utils.api_client import MistralClient
from lecture_transcriber.audio.extractor import extract_audio
from lecture_transcriber.audio.transcriber import transcribe, TranscriptSegment, TranscriptWord
from lecture_transcriber.video.sampler import sample_frames
from lecture_transcriber.video.keyframe_detector import detect_keyframes
from lecture_transcriber.video.keyframe_tracker import KeyframeTracker
from lecture_transcriber.video.visual_recognizer import recognize_visual_content, VisualSegment
from lecture_transcriber.merge.synchronizer import synchronize
from lecture_transcriber.merge.formatter import format_markdown

log = logging.getLogger(__name__)


def run_pipeline(
    video_path: str | Path,
    output_path: str | Path,
    config: Config,
    *,
    save_keyframes: str | Path | None = None,
    save_transcript: str | Path | None = None,
    save_visual: str | Path | None = None,
    no_audio: bool = False,
    title: str = "Конспект лекции",
    cache: bool = False,
) -> Path:
    """Run the full transcription pipeline and write Markdown output."""
    video_path = Path(video_path)
    output_path = Path(output_path)

    client = MistralClient(
        api_key=config.mistral_api_key,
        rps_limit=config.rps_limit,
    )

    with Progress() as progress:
        # --- Stage 1: Audio ---------------------------------------------------
        transcript: list[TranscriptSegment] = []
        if not no_audio:
            cached_transcript = _load_cached_transcript(save_transcript) if cache else None
            if cached_transcript is not None:
                transcript = cached_transcript
                log.info("Loaded %d transcript segments from cache", len(transcript))
            else:
                task_audio = progress.add_task("Audio extraction", total=2)
                wav_path = extract_audio(video_path, sample_rate=config.audio_sample_rate)
                progress.advance(task_audio)

                transcript = transcribe(
                    wav_path, client,
                    language=config.transcription_language,
                    context_bias=config.context_bias_terms or None,
                )
                progress.advance(task_audio)

                if save_transcript:
                    Path(save_transcript).write_text(
                        json.dumps([_seg_to_dict(s) for s in transcript],
                                   ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

        # --- Stage 2: Visual --------------------------------------------------
        cached_visual = _load_cached_visual(save_visual) if cache else None
        if cached_visual is not None:
            visual_segments = cached_visual
            log.info("Loaded %d visual segments from cache", len(visual_segments))
        else:
            task_visual = progress.add_task("Visual pipeline", total=3)

            # Check if keyframes are already saved on disk
            kf_on_disk = _load_cached_keyframes(save_keyframes) if cache else None
            if kf_on_disk is not None:
                vision_events = kf_on_disk
                log.info("Loaded %d keyframes from cache", len(vision_events))
                progress.advance(task_visual, 2)
            else:
                frames = sample_frames(video_path, fps=config.sample_fps)
                progress.advance(task_visual)

                keyframe_events = list(detect_keyframes(
                    frames,
                    ssim_threshold_significant=config.ssim_threshold_significant,
                    ssim_threshold_minor=config.ssim_threshold_minor,
                    flicker_window=config.flicker_window,
                    flicker_recovery_threshold=config.flicker_recovery_threshold,
                ))
                progress.advance(task_visual)
                log.info("Detected %d keyframes", len(keyframe_events))

                tracker = KeyframeTracker(
                    buffer_size=config.buffer_size,
                    match_threshold=config.match_threshold,
                    update_threshold=config.update_threshold,
                )
                vision_events = []
                for kf in keyframe_events:
                    ve = tracker.process_event(kf.timestamp, kf.frame)
                    vision_events.append((ve.keyframe_id, ve.timestamp, ve.event_type, ve.frame))

            visual_segments = recognize_visual_content(
                vision_events, client,
                model=config.vision_model,
                save_keyframes_dir=save_keyframes,
            )
            progress.advance(task_visual)

            if save_visual:
                Path(save_visual).write_text(
                    json.dumps([_vs_to_dict(s) for s in visual_segments],
                               ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

        # --- Stage 3: Merge & output ------------------------------------------
        task_merge = progress.add_task("Merge & format", total=1)

        blocks = synchronize(transcript, visual_segments, merge_window=config.merge_window_seconds)
        markdown = format_markdown(
            blocks,
            title=title,
            include_timestamps=config.include_timestamps,
            include_audio=config.include_audio_text,
        )
        output_path.write_text(markdown, encoding="utf-8")
        progress.advance(task_merge)

    log.info("Output written to %s", output_path)
    return output_path


# --- Cache helpers -----------------------------------------------------------

def _load_cached_transcript(path: str | Path | None) -> list[TranscriptSegment] | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return [
            TranscriptSegment(
                text=s["text"], start=s["start"], end=s["end"],
                words=[TranscriptWord(**w) for w in s.get("words", [])],
            )
            for s in data
        ]
    except Exception:
        log.warning("Failed to load cached transcript from %s", path)
        return None


def _load_cached_visual(path: str | Path | None) -> list[VisualSegment] | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return [
            VisualSegment(
                keyframe_id=s["keyframe_id"], timestamp=s["timestamp"],
                event_type=s["event_type"], text=s["text"],
                latex=s["latex"], diagram_description=s["diagram_description"],
                raw_response=s["raw_response"],
            )
            for s in data
        ]
    except Exception:
        log.warning("Failed to load cached visual from %s", path)
        return None


def _load_cached_keyframes(dir_path: str | Path | None) -> list[tuple] | None:
    """Load keyframe images from disk if they exist."""
    if dir_path is None:
        return None
    d = Path(dir_path)
    if not d.exists():
        return None
    pngs = sorted(d.glob("kf_*.png"))
    if not pngs:
        return None
    try:
        events = []
        for png in pngs:
            kf_id = png.stem
            # Extract timestamp from the visual.json if available, otherwise use 0
            frame = cv2.imread(str(png))
            if frame is None:
                continue
            events.append((kf_id, 0.0, "new_content", frame))
        return events if events else None
    except Exception:
        log.warning("Failed to load cached keyframes from %s", dir_path)
        return None


def _seg_to_dict(s: TranscriptSegment) -> dict:
    return {
        "text": s.text, "start": s.start, "end": s.end,
        "words": [{"text": w.text, "start": w.start, "end": w.end} for w in s.words],
    }


def _vs_to_dict(s: VisualSegment) -> dict:
    return {
        "keyframe_id": s.keyframe_id, "timestamp": s.timestamp,
        "event_type": s.event_type, "text": s.text,
        "latex": s.latex, "diagram_description": s.diagram_description,
        "raw_response": s.raw_response,
    }
