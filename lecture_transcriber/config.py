"""Configuration management — dataclass + env/.yaml loading."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv


@dataclass
class Config:
    # API
    mistral_api_key: str = ""
    rps_limit: float = 1.0

    # Audio
    audio_sample_rate: int = 16000
    transcription_language: str = "ru"
    context_bias_terms: list[str] = field(default_factory=list)

    # Video sampling
    sample_fps: float = 2.0

    # Keyframe detection
    ssim_threshold_significant: float = 0.85
    ssim_threshold_minor: float = 0.95
    flicker_window: int = 3
    flicker_recovery_threshold: float = 0.95

    # Keyframe tracking (Phase 2)
    buffer_size: int = 50
    match_threshold: float = 0.80
    update_threshold: float = 0.90
    use_phash_prefilter: bool = True

    # Visual recognition
    vision_model: str = "mistral-large-latest"
    vision_max_tokens: int | None = None
    vision_min_interval: float = 8.0
    vision_max_429_retries: int = 3
    vision_rate_limit_base_wait: float = 20.0
    vision_abort_on_rate_limit: bool = True
    skip_minor_updates: bool = True

    # Merge
    merge_window_seconds: float = 5.0

    # Output
    include_timestamps: bool = True
    include_audio_text: bool = True
    output_format: str = "markdown"

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> Config:
        """Load config from YAML file, then overlay environment variables."""
        data: dict = {}

        load_dotenv()  # load .env into os.environ

        if config_path is not None:
            path = Path(config_path)
            if path.exists():
                with open(path) as f:
                    data = yaml.safe_load(f) or {}

        # Environment overrides
        env_key = os.environ.get("MISTRAL_API_KEY")
        if env_key:
            data.setdefault("mistral_api_key", env_key)

        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
