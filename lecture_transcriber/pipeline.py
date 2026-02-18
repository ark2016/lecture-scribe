"""Orchestrator — connects all pipeline stages."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
from rich.progress import Progress

from lecture_transcriber.config import Config
from lecture_transcriber.utils.api_client import MistralClient
from lecture_transcriber.audio.extractor import extract_audio
from lecture_transcriber.audio.transcriber import transcribe, TranscriptSegment, TranscriptWord
from lecture_transcriber.video.sampler import sample_frames
from lecture_transcriber.video.keyframe_detector import detect_keyframes
from lecture_transcriber.video.keyframe_tracker import KeyframeTracker
from lecture_transcriber.video.visual_recognizer import recognize_visual_content, deduplicate_visual_segments
from lecture_transcriber.merge.synchronizer import synchronize
from lecture_transcriber.merge.formatter import format_markdown, format_html

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
        vision_max_tokens=config.vision_max_tokens,
        vision_min_interval=config.vision_min_interval,
        max_429_retries=config.vision_max_429_retries,
        rate_limit_base_wait=config.vision_rate_limit_base_wait,
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

            # Save keyframe metadata so timestamps survive caching
            if save_keyframes:
                _save_keyframe_metadata(save_keyframes, vision_events)

        visual_segments = recognize_visual_content(
            vision_events, client,
            model=config.vision_model,
            save_keyframes_dir=save_keyframes,
            save_visual_path=save_visual,  # incremental save + resume
            abort_on_rate_limit=config.vision_abort_on_rate_limit,
        )
        visual_segments = deduplicate_visual_segments(visual_segments)
        progress.advance(task_visual)

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

        html_path = output_path.with_suffix(".html")
        html_doc = format_html(
            blocks,
            title=title,
            include_timestamps=config.include_timestamps,
            include_audio=config.include_audio_text,
        )
        html_path.write_text(html_doc, encoding="utf-8")
        progress.advance(task_merge)

    log.info("Output written to %s (+ %s)", output_path, html_path)
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



def _save_keyframe_metadata(
    dir_path: str | Path,
    events: list[tuple],
) -> None:
    """Save keyframe metadata (timestamps, event types) alongside PNGs."""
    d = Path(dir_path)
    d.mkdir(parents=True, exist_ok=True)
    meta = {
        kf_id: {"timestamp": ts, "event_type": etype}
        for kf_id, ts, etype, _frame in events
    }
    (d / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8",
    )


def _load_cached_keyframes(dir_path: str | Path | None) -> list[tuple] | None:
    """Load keyframe images + metadata from disk if they exist."""
    if dir_path is None:
        return None
    d = Path(dir_path)
    if not d.exists():
        return None
    pngs = sorted(d.glob("kf_*.png"))
    if not pngs:
        return None

    # Load metadata for timestamps
    meta: dict = {}
    meta_path = d / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            log.warning("Failed to load keyframe metadata, timestamps will be 0")

    try:
        events = []
        for png in pngs:
            kf_id = png.stem
            frame = cv2.imread(str(png))
            if frame is None:
                continue
            kf_meta = meta.get(kf_id, {})
            ts = kf_meta.get("timestamp", 0.0)
            etype = kf_meta.get("event_type", "new_content")
            events.append((kf_id, ts, etype, frame))
        return events if events else None
    except Exception:
        log.warning("Failed to load cached keyframes from %s", dir_path)
        return None


def _seg_to_dict(s: TranscriptSegment) -> dict:
    return {
        "text": s.text, "start": s.start, "end": s.end,
        "words": [{"text": w.text, "start": w.start, "end": w.end} for w in s.words],
    }


# --- Audio-only pipeline ----------------------------------------------------

AUDIO_EXTENSIONS = frozenset((
    ".aac", ".mp3", ".m4a", ".wav", ".ogg", ".flac", ".wma", ".opus",
))


def _build_frontmatter(subject: str, source: str) -> str:
    """Build YAML frontmatter for Obsidian."""
    from datetime import date

    lines = ["---"]
    if subject:
        lines.append(f'subject: "{subject}"')
    lines.append(f'source: "{source}"')
    lines.append(f"date: {date.today().isoformat()}")
    lines.append("type: lecture-notes")
    lines.append("---\n\n")
    return "\n".join(lines)


def run_audio_pipeline(
    audio_path: str | Path,
    output_path: str | Path,
    config: Config,
    *,
    subject: str = "",
    save_transcript: str | Path | None = None,
    cache: bool = False,
) -> Path:
    """Run the audio-only pipeline: transcribe + LLM post-process -> Obsidian notes."""
    audio_path = Path(audio_path)
    output_path = Path(output_path)

    client = MistralClient(
        api_key=config.mistral_api_key,
        rps_limit=config.rps_limit,
    )

    with Progress() as progress:
        # --- Stage 1: Convert to WAV if needed --------------------------------
        task_audio = progress.add_task("Audio preparation", total=2)

        if audio_path.suffix.lower() == ".wav":
            wav_path = audio_path
        else:
            wav_path = extract_audio(audio_path, sample_rate=config.audio_sample_rate)
        progress.advance(task_audio)

        # --- Stage 2: Transcription -------------------------------------------
        cached_transcript = _load_cached_transcript(save_transcript) if cache else None
        if cached_transcript is not None:
            transcript = cached_transcript
            log.info("Loaded %d transcript segments from cache", len(transcript))
        else:
            transcript = transcribe(
                wav_path,
                client,
                language=config.transcription_language,
                context_bias=config.context_bias_terms or None,
            )
            if save_transcript:
                Path(save_transcript).write_text(
                    json.dumps(
                        [_seg_to_dict(s) for s in transcript],
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
        progress.advance(task_audio)

        # --- Stage 3: LLM post-processing ------------------------------------
        task_llm = progress.add_task("LLM post-processing", total=1)

        seg_dicts = [
            {"text": s.text, "start": s.start, "end": s.end}
            for s in transcript
        ]

        from lecture_transcriber.llm.postprocessor import postprocess_transcript

        notes_md = postprocess_transcript(
            segments=seg_dicts,
            subject=subject or config.subject,
            client=client,
            model=config.postprocess_model,
            target_chunk_words=config.postprocess_chunk_words,
        )
        progress.advance(task_llm)

        # --- Stage 4: Write output --------------------------------------------
        frontmatter = _build_frontmatter(
            subject=subject or config.subject,
            source=audio_path.name,
        )
        output_path.write_text(frontmatter + notes_md, encoding="utf-8")

    log.info("Output written to %s", output_path)
    return output_path
