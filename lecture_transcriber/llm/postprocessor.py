"""LLM post-processing: raw transcript -> structured Obsidian notes."""

from __future__ import annotations

import logging
import re

from lecture_transcriber.utils.api_client import MistralClient

log = logging.getLogger(__name__)

CHUNK_TARGET_WORDS = 8000
CHUNK_OVERLAP_WORDS = 500
DEFAULT_MODEL = "mistral-large-latest"

PROMPT_MODES = ("lecture", "lab")

# ── Lecture mode ─────────────────────────────────────────────────────────────

_SYSTEM_LECTURE = (
    "Ты — расшифровщик аудиозаписей лекций. Твоя задача — оформить сырой транскрипт "
    "в читаемый текст в формате Obsidian Markdown, максимально близко к тому, что было сказано."
)

_CHUNK_LECTURE = """\
Предмет: {subject}
{chunk_context}

Ниже — фрагмент транскрипта лекции{chunk_label}.
Временной диапазон: примерно {time_start} — {time_end}.

ТРАНСКРИПТ:
{transcript_text}

---

Оформи этот транскрипт в читаемый текст. Правила:

1. ГЛАВНОЕ ПРАВИЛО — ВЕРНОСТЬ ИСТОЧНИКУ:
   - Пиши ТОЛЬКО то, что сказано в транскрипте
   - НИКОГДА не добавляй информацию от себя, не дополняй, не «улучшай» содержание
   - Если что-то в транскрипте непонятно или обрывается — оставь как есть
   - Не придумывай примеры, определения или пояснения, которых нет в записи
   - Лучше оставить неполное предложение, чем додумать за лектора

2. ФОРМАТИРОВАНИЕ:
   - Раздели текст на абзацы по смыслу (по смене темы в речи лектора)
   - Заголовки ## и ### — только если лектор явно обозначает новую тему/раздел
   - Маркированные списки — только если лектор явно перечисляет пункты

3. МАТЕМАТИКА:
   - Inline формулы: $формула$
   - Блочные формулы: $$формула$$
   - Оформляй в LaTeX только те выражения, которые лектор произносит

4. CALLOUT-блоки Obsidian — используй ТОЛЬКО когда лектор явно обозначает:
   - > [!definition] — лектор говорит «определение», «определим»
   - > [!theorem] — лектор говорит «теорема»
   - > [!example] — лектор говорит «пример», «рассмотрим пример»
   - > [!important] — лектор говорит «важно», «обратите внимание», «запомните»
   - Если нет явного маркера — НЕ оборачивай в callout

5. СТИЛЬ:
   - Сохраняй речь лектора близко к оригиналу
   - Убирай только явные слова-паразиты (э-э, ну, вот, значит) и дословные повторы
   - НЕ переписывай своими словами, НЕ «улучшай» формулировки
   - Если лектор говорит «мы» — оставляй «мы»

6. ВРЕМЕННЫЕ МЕТКИ: НЕ добавляй таймстемпы.

Верни ТОЛЬКО оформленный текст в Obsidian Markdown, без обёрток в ``` и без пояснений.\
"""

_MERGE_LECTURE = """\
Ниже — несколько частей расшифровки одной лекции по предмету «{subject}».
Они были сгенерированы отдельно по частям транскрипта.

Задача: склей их в один цельный текст.

Правила:
1. Убери дословно повторяющийся текст на стыках частей (из-за overlap)
2. НЕ переписывай и не «улучшай» текст — только склей
3. Сохрани всё форматирование: заголовки, callout-блоки, формулы ($, $$)
4. НЕ добавляй и НЕ убирай содержательную информацию

Части расшифровки:

{chunks_text}

Верни ТОЛЬКО склеенный текст в Obsidian Markdown, без обёрток и пояснений.\
"""

# ── Lab mode ─────────────────────────────────────────────────────────────────

_SYSTEM_LAB = (
    "Ты — расшифровщик аудиозаписей. Твоя задача — оформить транскрипт занятия, "
    "где обсуждаются условия лабораторной работы, в читаемый текст Obsidian Markdown."
)

_CHUNK_LAB = """\
Предмет: {subject}
{chunk_context}

Ниже — фрагмент транскрипта занятия{chunk_label}, на котором преподаватель \
обсуждает лабораторную работу (условия, нотации, требования).
Временной диапазон: примерно {time_start} — {time_end}.

ТРАНСКРИПТ:
{transcript_text}

---

Оформи этот транскрипт в читаемый текст. Правила:

1. ГЛАВНОЕ ПРАВИЛО — ВЕРНОСТЬ ИСТОЧНИКУ:
   - Пиши ТОЛЬКО то, что сказано в транскрипте
   - НИКОГДА не добавляй информацию от себя — ни условия, ни формулы, ни пояснения
   - Если что-то обрывается или непонятно — оставь как есть
   - Лучше неполное предложение, чем додуманное

2. ФОРМАТИРОВАНИЕ:
   - Раздели текст на абзацы по смыслу
   - Заголовки ## и ### — только если преподаватель явно обозначает новый раздел
   - Нумерованные списки — если преподаватель перечисляет пункты задания
   - Таблицы нотаций — только если преподаватель явно вводит набор обозначений

3. МАТЕМАТИКА:
   - Inline формулы: $формула$
   - Блочные формулы: $$формула$$
   - Оформляй в LaTeX только выражения, которые преподаватель произносит

4. CALLOUT-блоки Obsidian — используй ТОЛЬКО когда преподаватель явно обозначает:
   - > [!task] — преподаватель формулирует конкретное задание
   - > [!warning] — преподаватель говорит «ограничение», «нельзя», «обязательно»
   - > [!example] — преподаватель приводит конкретный пример
   - > [!important] — преподаватель говорит «важно», «обратите внимание»
   - Если нет явного маркера — НЕ оборачивай в callout

5. СТИЛЬ:
   - Сохраняй речь преподавателя близко к оригиналу
   - Убирай только явные слова-паразиты и дословные повторы
   - НЕ переписывай своими словами, НЕ структурируй больше, чем было в речи

6. ВРЕМЕННЫЕ МЕТКИ: НЕ добавляй таймстемпы.

Верни ТОЛЬКО оформленный текст в Obsidian Markdown, без обёрток в ``` и без пояснений.\
"""

_MERGE_LAB = """\
Ниже — несколько частей расшифровки занятия по предмету «{subject}» (лабораторная работа).
Они были сгенерированы отдельно по частям транскрипта.

Задача: склей их в один цельный текст.

Правила:
1. Убери дословно повторяющийся текст на стыках частей (из-за overlap)
2. НЕ переписывай и не «улучшай» текст — только склей
3. Сохрани всё форматирование: заголовки, callout-блоки, формулы ($, $$), таблицы
4. НЕ добавляй и НЕ убирай содержательную информацию

Части расшифровки:

{chunks_text}

Верни ТОЛЬКО склеенный текст в Obsidian Markdown, без обёрток и пояснений.\
"""

# ── Mode dispatch ────────────────────────────────────────────────────────────

_PROMPTS = {
    "lecture": (_SYSTEM_LECTURE, _CHUNK_LECTURE, _MERGE_LECTURE),
    "lab":     (_SYSTEM_LAB,     _CHUNK_LAB,     _MERGE_LAB),
}


def _format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _count_words(text: str) -> int:
    return len(text.split())


def _clean_llm_output(text: str) -> str:
    """Strip markdown code fences the LLM may wrap its response in."""
    text = text.strip()
    if text.startswith("```"):
        # Remove opening fence (possibly ```markdown)
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text, count=1)
    if text.endswith("```"):
        text = text[: -len("```")]
    return text.strip()


def chunk_transcript(
    segments: list[dict],
    target_words: int = CHUNK_TARGET_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> list[tuple[str, float, float]]:
    """Split transcript segments into overlapping text chunks.

    Returns list of ``(chunk_text, time_start, time_end)``.
    """
    if not segments:
        return []

    chunks: list[tuple[str, float, float]] = []
    current_texts: list[str] = []
    current_word_count = 0
    chunk_start_time = segments[0].get("start", 0.0)
    segment_buffer: list[dict] = []

    for seg in segments:
        seg_text = seg["text"].strip()
        if not seg_text:
            continue
        seg_words = _count_words(seg_text)
        current_texts.append(seg_text)
        current_word_count += seg_words
        segment_buffer.append(seg)

        if current_word_count >= target_words:
            chunk_text = " ".join(current_texts)
            chunk_end_time = seg.get("end", 0.0)
            chunks.append((chunk_text, chunk_start_time, chunk_end_time))

            # Build overlap from tail of current buffer
            overlap_texts: list[str] = []
            overlap_count = 0
            overlap_segs: list[dict] = []
            for s in reversed(segment_buffer):
                s_words = _count_words(s["text"].strip())
                if overlap_count + s_words > overlap_words:
                    break
                overlap_texts.insert(0, s["text"].strip())
                overlap_segs.insert(0, s)
                overlap_count += s_words

            current_texts = list(overlap_texts)
            current_word_count = overlap_count
            segment_buffer = list(overlap_segs)
            chunk_start_time = (
                overlap_segs[0].get("start", 0.0) if overlap_segs else seg.get("end", 0.0)
            )

    # Remaining text
    if current_texts:
        chunk_text = " ".join(current_texts)
        chunk_end_time = segments[-1].get("end", 0.0)
        # Avoid emitting a tiny final chunk that duplicates the overlap
        if chunks and _count_words(chunk_text) < overlap_words:
            # Merge into last chunk
            prev_text, prev_start, _ = chunks[-1]
            chunks[-1] = (prev_text + " " + chunk_text, prev_start, chunk_end_time)
        else:
            chunks.append((chunk_text, chunk_start_time, chunk_end_time))

    return chunks


def postprocess_transcript(
    segments: list[dict],
    subject: str,
    client: MistralClient,
    model: str = DEFAULT_MODEL,
    target_chunk_words: int = CHUNK_TARGET_WORDS,
    mode: str = "lecture",
) -> str:
    """Convert raw transcript segments into structured Obsidian Markdown notes.

    Args:
        segments: list of dicts with ``text``, ``start``, ``end`` keys.
        subject: lecture subject name (e.g. ``"Математический анализ"``).
        client: :class:`MistralClient` instance.
        model: LLM model to use for structuring.
        target_chunk_words: target word count per chunk.
        mode: ``"lecture"`` for lecture notes, ``"lab"`` for lab assignment conditions.

    Returns:
        Structured Obsidian Markdown string.
    """
    if mode not in _PROMPTS:
        raise ValueError(f"Unknown mode {mode!r}, expected one of {list(_PROMPTS)}")
    system_prompt, chunk_template, merge_template = _PROMPTS[mode]

    chunks = chunk_transcript(segments, target_words=target_chunk_words)
    if not chunks:
        return ""

    total = len(chunks)
    log.info("Processing transcript in %d chunk(s), mode=%s", total, mode)

    note_parts: list[str] = []

    for i, (text, t_start, t_end) in enumerate(chunks):
        chunk_num = i + 1
        chunk_context = ""
        chunk_label = ""
        if total > 1:
            chunk_context = (
                f"Это часть {chunk_num} из {total}. "
                "Пиши конспект только для этой части, но учитывай контекст."
            )
            chunk_label = f" (часть {chunk_num} из {total})"

        prompt = chunk_template.format(
            subject=subject or "(не указан)",
            chunk_context=chunk_context,
            chunk_label=chunk_label,
            chunk_num=chunk_num,
            total_chunks=total,
            time_start=_format_time(t_start),
            time_end=_format_time(t_end),
            transcript_text=text,
        )

        log.info(
            "Processing chunk %d/%d (%d words, %s — %s)",
            chunk_num,
            total,
            _count_words(text),
            _format_time(t_start),
            _format_time(t_end),
        )

        result = client.chat_text(
            prompt,
            model=model,
            system_prompt=system_prompt,
        )
        note_parts.append(_clean_llm_output(result))
        log.info("Chunk %d/%d done", chunk_num, total)

    # Single chunk — use directly
    if len(note_parts) == 1:
        return note_parts[0]

    # Multiple chunks — merge pass
    log.info("Running merge pass for %d chunks", len(note_parts))
    chunks_text = ""
    for i, part in enumerate(note_parts):
        chunks_text += f"\n\n---\n**[Часть {i + 1}]**\n\n{part}"

    merge_prompt = merge_template.format(
        subject=subject or "(не указан)",
        chunks_text=chunks_text,
    )
    merged = client.chat_text(
        merge_prompt,
        model=model,
        system_prompt=system_prompt,
    )
    return _clean_llm_output(merged)
