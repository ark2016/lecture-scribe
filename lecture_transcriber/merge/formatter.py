"""Generate Markdown output from synchronized lecture blocks."""

from __future__ import annotations

from lecture_transcriber.merge.synchronizer import LectureBlock


def _format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_markdown(
    blocks: list[LectureBlock],
    title: str = "Конспект лекции",
    include_timestamps: bool = True,
    include_audio: bool = True,
) -> str:
    """Render a list of LectureBlocks as a Markdown document."""
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")

    for block in blocks:
        ts = _format_timestamp(block.timestamp_start)
        if include_timestamps:
            lines.append(f"## [{ts}]")
        else:
            lines.append("## ---")
        lines.append("")

        if include_audio and block.audio_text:
            lines.append(f"**Лектор:** {block.audio_text}")
            lines.append("")

        if block.visual_content:
            lines.append("**На доске:**")
            lines.append("")
            lines.append(block.visual_content)
            lines.append("")

        for diagram in block.diagram_descriptions:
            lines.append(f"> **Схема:** {diagram}")
            lines.append("")

        lines.append("")

    return "\n".join(lines)
