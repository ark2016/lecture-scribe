"""Recognize visual content from keyframes using Mistral Vision."""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np

from lecture_transcriber.utils.api_client import MistralClient, RateLimitExhaustedError

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

PROMPT_STRUCTURE_OCR = """\
Ниже — OCR-текст из кадра видеолекции. Лекция записана через Telegram-звонок,
лектор пишет от руки в Paint. OCR мог допустить ошибки в рукописном тексте.

Правила:
1. Исправь ошибки OCR, восстанови осмысленный русский текст и математику
2. Все формулы запиши в LaTeX ($..$ для inline, $$...$$ для display)
3. Если есть схемы/графики — кратко опиши
4. Порядок: сверху вниз, слева направо
5. ИГНОРИРУЙ элементы интерфейса Telegram: "... is speaking", имена участников,
   "N участников", кнопки, аватарки, "BMSTU ...", "img-0.jpeg" и т.п.
   Эти строки — мусор от захвата экрана, не часть лекции.
6. ВАЖНО для JSON: все обратные слэши в LaTeX удваивай: \\\\frac, \\\\omega, \\\\( ... \\\\)

Ответ — строго JSON, без пояснений и без markdown-обёрток:
{"text": "...", "latex_blocks": ["..."], "diagrams": ["..."]}

OCR-текст:
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
    save_visual_path: str | Path | None = None,
    abort_on_rate_limit: bool = True,
) -> list[VisualSegment]:
    """Recognize content for a list of (keyframe_id, timestamp, event_type, frame) tuples.

    Phase 1: only uses PROMPT_NEW_CONTENT for every event.

    If *save_visual_path* is given and already contains results, keyframes that
    have already been recognised are skipped (resume after rate-limit crash).
    New results are appended to the file after each successful API call so
    progress is never lost.
    """
    # Load existing results for resume
    existing: dict[str, dict] = {}
    if save_visual_path:
        p = Path(save_visual_path)
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                existing = {s["keyframe_id"]: s for s in data}
            except Exception:
                log.warning("Could not load existing visual.json, starting fresh")

    segments: list[VisualSegment] = []
    cache_repaired = False

    for kf_id, timestamp, event_type, frame in events:
        # Resume: skip already-recognised keyframes
        if kf_id in existing:
            s = existing[kf_id]
            text = s["text"]
            latex = s["latex"]
            diagrams_desc = s["diagram_description"]

            # Repair: if text looks like raw JSON, re-parse it
            stripped = text.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                reparsed = _parse_response(stripped)
                new_text = reparsed.get("text", "")
                if new_text and not new_text.strip().startswith("{"):
                    text = new_text
                    latex = "; ".join(reparsed.get("latex_blocks", [])) or latex
                    diagrams_desc = "; ".join(reparsed.get("diagrams", [])) or diagrams_desc
                    cache_repaired = True
                    log.info("Repaired cached text for %s", kf_id)

            segments.append(VisualSegment(
                keyframe_id=s["keyframe_id"], timestamp=s["timestamp"],
                event_type=s["event_type"], text=text,
                latex=latex, diagram_description=diagrams_desc,
                raw_response=s["raw_response"],
            ))
            log.info("Reusing cached result for %s", kf_id)
            continue

        img_path = _save_frame(frame, kf_id, save_keyframes_dir)
        try:
            # Step 1: OCR — extract raw text from image
            ocr_text = client.ocr_image(img_path)
            log.info("OCR for %s: %d chars", kf_id, len(ocr_text))

            if not ocr_text.strip():
                log.warning("OCR returned empty text for %s, skipping", kf_id)
                continue

            # Step 2: Text LLM — structure OCR output into JSON
            raw = client.chat_text(
                PROMPT_STRUCTURE_OCR + ocr_text,
                model=model,
            )
        except RateLimitExhaustedError as exc:
            retry_hint = ""
            if exc.retry_after_seconds is not None:
                retry_hint = f" Retry after about {exc.retry_after_seconds:.0f}s."
            log.error(
                "Vision API rate limit persisted for %s at %.2fs.%s",
                kf_id,
                timestamp,
                retry_hint,
            )
            if abort_on_rate_limit:
                log.error("Stopping visual recognition early to preserve progress.")
                break
            log.warning("Continuing with next keyframe after rate-limit failure.")
            continue
        except Exception:
            log.exception("Vision API failed for %s at %.2fs, skipping", kf_id, timestamp)
            continue

        parsed = _parse_response(raw)
        seg = VisualSegment(
            keyframe_id=kf_id,
            timestamp=timestamp,
            event_type=event_type,
            text=parsed.get("text", ""),
            latex="; ".join(parsed.get("latex_blocks", [])),
            diagram_description="; ".join(parsed.get("diagrams", [])),
            raw_response=raw,
        )
        segments.append(seg)
        log.info("Recognized %s at %.2fs", kf_id, timestamp)

        # Save incrementally so progress survives crashes
        if save_visual_path:
            _save_visual_incremental(segments, save_visual_path)

    # Persist repaired cache entries
    if cache_repaired and save_visual_path:
        _save_visual_incremental(segments, save_visual_path)
        log.info("Saved repaired visual cache to %s", save_visual_path)

    return segments


def _vs_to_dict(s: VisualSegment) -> dict:
    return {
        "keyframe_id": s.keyframe_id, "timestamp": s.timestamp,
        "event_type": s.event_type, "text": s.text,
        "latex": s.latex, "diagram_description": s.diagram_description,
        "raw_response": s.raw_response,
    }


def _save_visual_incremental(segments: list[VisualSegment], path: str | Path) -> None:
    Path(path).write_text(
        json.dumps([_vs_to_dict(s) for s in segments], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def deduplicate_visual_segments(
    segments: list[VisualSegment],
    similarity_threshold: float = 0.7,
    lookback: int = 5,
) -> list[VisualSegment]:
    """Remove near-duplicate visual segments, keeping the latest version.

    Compares each segment's text against the last *lookback* kept segments.
    If similar enough, replaces the older duplicate (the newer capture is
    usually more complete).  Returns a new list.
    """
    if not segments:
        return segments

    deduped: list[VisualSegment] = [segments[0]]
    for seg in segments[1:]:
        if not seg.text.strip():
            continue
        replaced = False
        for i in range(max(0, len(deduped) - lookback), len(deduped)):
            ratio = SequenceMatcher(None, deduped[i].text, seg.text).ratio()
            if ratio >= similarity_threshold:
                deduped[i] = seg  # keep newer (more complete) version
                replaced = True
                break
        if not replaced:
            deduped.append(seg)

    log.info("Dedup: %d segments -> %d unique", len(segments), len(deduped))
    return deduped


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


def _fix_json_escapes(s: str) -> str:
    r"""Fix LaTeX backslash sequences that break JSON parsing.

    The LLM often writes ``\frac`` (single backslash) inside JSON strings,
    which is invalid JSON.  This function doubles lone backslashes that are
    *not* valid JSON escape sequences.

    Tricky overlap: ``\n``, ``\t``, ``\f``, ``\r``, ``\b`` are valid JSON
    escapes, but ``\nu``, ``\theta``, ``\frac``, ``\rho``, ``\bar`` are LaTeX.
    We use a suffix whitelist to distinguish them.
    """
    # Letters that commonly follow \b, \f, \n, \r, \t to form LaTeX commands.
    # If the char after the escape letter is NOT in this set → JSON escape.
    _latex_suffixes: dict[str, frozenset[str]] = {
        'b': frozenset('aeimorux'),   # bar, big, binom, boldsymbol, bot, boxed, bullet, bmod
        'f': frozenset('lor'),        # flat, frac, forall  (NOT 'a' — \fa isn't common LaTeX)
        'n': frozenset('aeou'),       # nabla, neg, neq, not, nu, nolimits  (NOT 'f','i',etc.)
        'r': frozenset('acehio'),     # rangle, rceil, rfloor, rho, right
        't': frozenset('aehiou'),     # tan, tanh, tau, text, theta, tilde, times, to, top, triangle
    }

    result: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch != '\\':
            result.append(ch)
            i += 1
            continue

        # At a backslash — look at next char
        if i + 1 >= n:
            result.append(ch)
            i += 1
            continue

        nxt = s[i + 1]

        # Already-doubled backslash
        if nxt == '\\':
            result.append('\\\\')
            i += 2
            continue

        # Unambiguous JSON escapes — keep as-is
        if nxt in '"/':
            result.append('\\')
            result.append(nxt)
            i += 2
            continue

        # \uXXXX — unicode escape
        if nxt == 'u' and i + 5 < n and all(c in '0123456789abcdefABCDEF' for c in s[i + 2 : i + 6]):
            result.append(s[i : i + 6])
            i += 6
            continue

        # Ambiguous: \b \f \n \r \t — check suffix whitelist
        if nxt in _latex_suffixes:
            is_latex = (
                i + 2 < n
                and s[i + 2].isalpha()
                and s[i + 2].lower() in _latex_suffixes[nxt]
            )
            if is_latex:
                result.append('\\\\')
            else:
                result.append('\\')
            result.append(nxt)
            i += 2
            continue

        # Any other \X where X is not a valid JSON escape — double it
        result.append('\\\\')
        result.append(nxt)
        i += 2

    return ''.join(result)


def _try_json_loads(s: str) -> dict | None:
    """Try json.loads, then retry with fixed LaTeX escapes."""
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(_fix_json_escapes(s))
    except json.JSONDecodeError:
        return None


def _parse_response(raw: str) -> dict:
    """Best-effort JSON parse of the LLM response."""
    import re

    # 1. Direct parse (with escape fix fallback)
    result = _try_json_loads(raw)
    if result is not None:
        return result

    # 2. Extract from markdown fences
    m = re.search(r"```(?:json)?\s*(\{.+\})\s*```", raw, re.DOTALL)
    if m:
        result = _try_json_loads(m.group(1))
        if result is not None:
            return result

    # 3. Find first { and match balanced braces
    start = raw.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(raw)):
            if raw[i] == "{":
                depth += 1
            elif raw[i] == "}":
                depth -= 1
                if depth == 0:
                    result = _try_json_loads(raw[start : i + 1])
                    if result is not None:
                        return result
                    break

    # 4. Fallback — strip any markdown/fences and use as plain text
    cleaned = re.sub(r"```(?:json)?\s*", "", raw)
    cleaned = re.sub(r"```", "", cleaned).strip()
    return {"text": cleaned, "latex_blocks": [], "diagrams": []}
