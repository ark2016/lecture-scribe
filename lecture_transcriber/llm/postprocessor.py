"""LLM post-processing: raw transcript -> structured Obsidian notes."""

from __future__ import annotations

import logging
import re

from lecture_transcriber.utils.api_client import MistralClient

log = logging.getLogger(__name__)

CHUNK_TARGET_WORDS = 8000
CHUNK_OVERLAP_WORDS = 500
DEFAULT_MODEL = "mistral-large-latest"

SYSTEM_PROMPT = (
    "Ты — ассистент-конспектолог. Твоя задача — превратить сырой транскрипт лекции "
    "в структурированный конспект в формате Obsidian Markdown."
)

CHUNK_PROMPT_TEMPLATE = """\
Предмет: {subject}
{chunk_context}

Ниже — фрагмент транскрипта лекции{chunk_label}.
Временной диапазон: примерно {time_start} — {time_end}.

ТРАНСКРИПТ:
{transcript_text}

---

Создай структурированный конспект этого фрагмента лекции. Правила:

1. СТРУКТУРА:
   - Используй заголовки ## и ### для разделов и подразделов
   - Используй маркированные и нумерованные списки
   - Группируй связанные мысли в параграфы

2. МАТЕМАТИКА:
   - Inline формулы: $формула$
   - Блочные формулы: $$формула$$
   - Записывай все математические выражения в LaTeX

3. CALLOUT-блоки Obsidian (используй по смыслу):
   - > [!definition] Определение — для определений терминов и понятий
   - > [!theorem] Теорема — для формулировок теорем
   - > [!example] Пример — для примеров и задач
   - > [!note] Заметка — для важных пояснений лектора
   - > [!important] Важно — для ключевых выводов и акцентов
   - > [!proof]- Доказательство — для доказательств (свёрнуто по умолчанию)

4. СТИЛЬ:
   - Пиши от третьего лица или безлично (не «мы рассмотрим», а «рассмотрим»)
   - Исправляй речевые ошибки и повторы из устной речи
   - Убирай слова-паразиты и повторения
   - Сохраняй ВСЕ содержательные идеи, ничего не пропускай
   - Не добавляй информацию, которой нет в транскрипте

5. ВРЕМЕННЫЕ МЕТКИ: НЕ добавляй таймстемпы в текст конспекта.

Верни ТОЛЬКО текст конспекта в Obsidian Markdown, без обёрток в ``` и без пояснений.\
"""

MERGE_PROMPT_TEMPLATE = """\
Ниже — несколько частей конспекта одной лекции по предмету «{subject}».
Они были сгенерированы отдельно по частям транскрипта.

Задача: объедини их в один цельный конспект.

Правила:
1. Убери дублирующиеся заголовки и повторяющийся материал на стыках частей
2. Обеспечь плавные переходы между разделами
3. Сохрани всю структуру: заголовки, callout-блоки, формулы ($, $$)
4. Если одна тема начинается в одной части и продолжается в другой — объедини
5. Не добавляй и не убирай содержательную информацию

Части конспекта:

{chunks_text}

Верни ТОЛЬКО итоговый конспект в Obsidian Markdown, без обёрток и пояснений.\
"""


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
) -> str:
    """Convert raw transcript segments into structured Obsidian Markdown notes.

    Args:
        segments: list of dicts with ``text``, ``start``, ``end`` keys.
        subject: lecture subject name (e.g. ``"Математический анализ"``).
        client: :class:`MistralClient` instance.
        model: LLM model to use for structuring.
        target_chunk_words: target word count per chunk.

    Returns:
        Structured Obsidian Markdown string.
    """
    chunks = chunk_transcript(segments, target_words=target_chunk_words)
    if not chunks:
        return ""

    total = len(chunks)
    log.info("Processing transcript in %d chunk(s)", total)

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

        prompt = CHUNK_PROMPT_TEMPLATE.format(
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
            system_prompt=SYSTEM_PROMPT,
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

    merge_prompt = MERGE_PROMPT_TEMPLATE.format(
        subject=subject or "(не указан)",
        chunks_text=chunks_text,
    )
    merged = client.chat_text(
        merge_prompt,
        model=model,
        system_prompt=SYSTEM_PROMPT,
    )
    return _clean_llm_output(merged)
