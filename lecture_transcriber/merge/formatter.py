"""Generate Markdown and HTML output from synchronized lecture blocks."""

from __future__ import annotations

import html

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


# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script>
MathJax = {{
  tex: {{ inlineMath: [['$','$'], ['\\\\(','\\\\)']], displayMath: [['$$','$$'], ['\\\\[','\\\\]']] }},
  svg: {{ fontCache: 'global' }}
}};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js" async></script>
<style>
  :root {{ --bg: #fff; --fg: #222; --muted: #666; --accent: #2563eb; --border: #e5e7eb; --block-bg: #f9fafb; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg: #1a1a2e; --fg: #e0e0e0; --muted: #999; --accent: #60a5fa; --border: #333; --block-bg: #16213e; }}
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: "Segoe UI", system-ui, sans-serif; background: var(--bg); color: var(--fg); line-height: 1.7; padding: 2rem; max-width: 900px; margin: 0 auto; }}
  h1 {{ font-size: 1.8rem; margin-bottom: 1.5rem; border-bottom: 2px solid var(--accent); padding-bottom: 0.5rem; }}
  .block {{ background: var(--block-bg); border: 1px solid var(--border); border-radius: 8px; padding: 1.2rem 1.5rem; margin-bottom: 1.2rem; }}
  .timestamp {{ font-size: 0.85rem; color: var(--muted); font-family: monospace; margin-bottom: 0.5rem; }}
  .speaker {{ margin-bottom: 0.6rem; }}
  .speaker strong {{ color: var(--accent); }}
  .board {{ background: var(--bg); border-left: 3px solid var(--accent); padding: 0.6rem 1rem; margin-bottom: 0.6rem; white-space: pre-wrap; }}
  .diagram {{ border-left: 3px solid #f59e0b; padding: 0.4rem 1rem; margin-bottom: 0.6rem; font-style: italic; color: var(--muted); }}
</style>
</head>
<body>
<h1>{title}</h1>
{blocks}
</body>
</html>
"""


def _escape(text: str) -> str:
    """HTML-escape text but preserve LaTeX delimiters."""
    return html.escape(text, quote=False)


def _render_block_html(block: LectureBlock, include_timestamps: bool, include_audio: bool) -> str:
    parts: list[str] = ['<div class="block">']

    if include_timestamps:
        ts = _format_timestamp(block.timestamp_start)
        parts.append(f'  <div class="timestamp">[{ts}]</div>')

    if include_audio and block.audio_text:
        parts.append(f'  <div class="speaker"><strong>Лектор:</strong> {_escape(block.audio_text)}</div>')

    if block.visual_content:
        parts.append(f'  <div class="board">{_escape(block.visual_content)}</div>')

    for diagram in block.diagram_descriptions:
        parts.append(f'  <div class="diagram"><strong>Схема:</strong> {_escape(diagram)}</div>')

    parts.append('</div>')
    return "\n".join(parts)


def format_html(
    blocks: list[LectureBlock],
    title: str = "Конспект лекции",
    include_timestamps: bool = True,
    include_audio: bool = True,
) -> str:
    """Render a list of LectureBlocks as an HTML document with MathJax support."""
    rendered = "\n".join(
        _render_block_html(b, include_timestamps, include_audio)
        for b in blocks
    )
    return _HTML_TEMPLATE.format(title=_escape(title), blocks=rendered)
