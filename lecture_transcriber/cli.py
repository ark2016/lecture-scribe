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
@click.option("--vision-max-tokens", type=click.IntRange(1, None), default=None, help="Cap tokens generated per vision response.")
@click.option("--vision-interval", type=float, default=None, help="Minimum seconds between vision API calls.")
@click.option("--vision-rate-retries", type=int, default=None, help="How many 429 retries to allow per vision call.")
@click.option(
    "--continue-after-rate-limit",
    is_flag=True,
    default=False,
    help="Keep processing the next keyframe after a hard 429 failure.",
)
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
    vision_max_tokens: int | None,
    vision_interval: float | None,
    vision_rate_retries: int | None,
    continue_after_rate_limit: bool,
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
        cfg.context_bias_terms = [t.strip() for t in terms.split(",") if t.strip()]
    if vision_model:
        cfg.vision_model = vision_model
    if vision_max_tokens is not None:
        cfg.vision_max_tokens = vision_max_tokens
    if vision_interval is not None:
        cfg.vision_min_interval = vision_interval
    if vision_rate_retries is not None:
        cfg.vision_max_429_retries = vision_rate_retries
    if continue_after_rate_limit:
        cfg.vision_abort_on_rate_limit = False
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

    html_output = output.with_suffix(".html")
    click.echo(f"Done! Output: {output} + {html_output}")


if __name__ == "__main__":
    main()
