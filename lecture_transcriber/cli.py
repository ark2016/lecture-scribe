"""CLI entry point using Click."""

from __future__ import annotations

import logging
from pathlib import Path

import click
from rich.logging import RichHandler

from lecture_transcriber.config import Config
from lecture_transcriber.pipeline import run_pipeline


@click.command()
@click.argument("video", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", type=click.Path(path_type=Path), default=None, help="Output Markdown path.")
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=None, help="YAML config file.")
@click.option("--terms", default=None, help="Comma-separated context bias terms for transcription.")
@click.option("--vision-model", default=None, help="Mistral vision model name.")
@click.option("--fps", type=float, default=None, help="Frame sampling rate (fps).")
@click.option("--no-audio", is_flag=True, default=False, help="Skip audio pipeline (visual only).")
@click.option("--title", default="Конспект лекции", help="Title for the output document.")
@click.option("--debug", is_flag=True, default=False, help="Enable debug logging.")
@click.option("--save-keyframes", type=click.Path(path_type=Path), default=None, help="Directory to save keyframe images.")
@click.option("--save-transcript", type=click.Path(path_type=Path), default=None, help="Path to save raw transcript JSON.")
@click.option("--save-visual", type=click.Path(path_type=Path), default=None, help="Path to save visual segments JSON.")
@click.option("--cache", is_flag=True, default=False, help="Reuse cached intermediate files (transcript, keyframes, visual) if they exist.")
def main(
    video: Path,
    output: Path | None,
    config_path: Path | None,
    terms: str | None,
    vision_model: str | None,
    fps: float | None,
    no_audio: bool,
    title: str,
    debug: bool,
    save_keyframes: Path | None,
    save_transcript: Path | None,
    save_visual: Path | None,
    cache: bool,
) -> None:
    """Transcribe a lecture VIDEO into structured Markdown notes."""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        handlers=[RichHandler(rich_tracebacks=True)],
        format="%(message)s",
    )

    cfg = Config.load(config_path)

    # CLI overrides
    if terms:
        cfg.context_bias_terms = [t.strip() for t in terms.split(",")]
    if vision_model:
        cfg.vision_model = vision_model
    if fps is not None:
        cfg.sample_fps = fps

    if not cfg.mistral_api_key:
        raise click.ClickException(
            "MISTRAL_API_KEY not set. Export it or add to .env / config YAML."
        )

    if output is None:
        output = video.with_suffix(".md")

    run_pipeline(
        video_path=video,
        output_path=output,
        config=cfg,
        save_keyframes=save_keyframes,
        save_transcript=save_transcript,
        save_visual=save_visual,
        no_audio=no_audio,
        title=title,
        cache=cache,
    )

    click.echo(f"Done! Output: {output}")


if __name__ == "__main__":
    main()
