"""Mistral API wrapper with retry and rate limiting."""

from __future__ import annotations

import base64
import logging
import time
from pathlib import Path

from mistralai import Mistral
from mistralai.models import SDKError

log = logging.getLogger(__name__)

# Vision calls: ~1100 input tokens + ~500 output ≈ 1600 tokens per call.
# Free tier: 50K tokens/min → max ~31 calls/min → 1 call per ~2s.
# We use 4s to leave margin.
_VISION_INTERVAL = 4.0
_RATE_LIMIT_BASE_WAIT = 60  # initial 429 backoff in seconds


class MistralClient:
    """Thin wrapper around the Mistral SDK adding rate limiting and retries."""

    def __init__(self, api_key: str, max_retries: int = 10, rps_limit: float = 1.0):
        self._client = Mistral(api_key=api_key)
        self._max_retries = max_retries
        self._min_interval = 1.0 / rps_limit if rps_limit > 0 else 0.0
        self._last_call: float = 0.0

    def _wait(self, interval: float) -> None:
        now = time.monotonic()
        elapsed = now - self._last_call
        if elapsed < interval:
            time.sleep(interval - elapsed)
        self._last_call = time.monotonic()

    def _call_with_retry(self, fn, *args, min_interval: float | None = None, **kwargs):
        """Call *fn* with retry logic that respects 429 rate limits."""
        interval = min_interval if min_interval is not None else self._min_interval
        last_exc = None
        for attempt in range(1, self._max_retries + 1):
            self._wait(interval)
            try:
                result = fn(*args, **kwargs)
                return result
            except SDKError as exc:
                last_exc = exc
                if "429" in str(exc) or "rate" in str(exc).lower():
                    # Progressive backoff: 60s, 90s, 120s, ...
                    wait = _RATE_LIMIT_BASE_WAIT + 30 * (attempt - 1)
                    log.warning(
                        "Rate limited (attempt %d/%d), waiting %ds...",
                        attempt, self._max_retries, wait,
                    )
                    time.sleep(wait)
                    self._last_call = time.monotonic()
                    continue
                backoff = min(2 ** attempt, 30)
                log.warning("API error (attempt %d/%d): %s — retrying in %ds", attempt, self._max_retries, exc, backoff)
                time.sleep(backoff)
            except Exception as exc:
                last_exc = exc
                backoff = min(2 ** attempt, 30)
                log.warning("Error (attempt %d/%d): %s — retrying in %ds", attempt, self._max_retries, exc, backoff)
                time.sleep(backoff)
        raise last_exc  # type: ignore[misc]

    # -- Audio transcription ---------------------------------------------------

    def transcribe_audio(
        self,
        audio_path: str | Path,
        language: str = "ru",
        context_bias: list[str] | None = None,
    ) -> dict:
        log.info("Transcribing %s", audio_path)
        audio_path = Path(audio_path)

        def _do_transcribe():
            kwargs: dict = {
                "model": "voxtral-mini-latest",
                "file": {
                    "file_name": audio_path.name,
                    "content": open(audio_path, "rb"),
                },
                "timestamp_granularities": ["segment"],
                "language": language,
            }
            if context_bias:
                kwargs["context_bias"] = context_bias
            return self._client.audio.transcriptions.complete(**kwargs)

        result = self._call_with_retry(_do_transcribe)
        return result.model_dump()

    # -- Vision ----------------------------------------------------------------

    def analyze_image(
        self,
        images: list[str | Path],
        prompt: str,
        model: str = "mistral-small-latest",
    ) -> str:
        """Send one or more images to Mistral Vision and return the text response.

        Each entry in *images* is a filesystem path; images are base64-encoded
        before sending.
        """
        content: list[dict] = []
        for img_path in images:
            raw = Path(img_path).read_bytes()
            b64 = base64.b64encode(raw).decode()
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
            )
        content.append({"type": "text", "text": prompt})

        resp = self._call_with_retry(
            self._client.chat.complete,
            model=model,
            messages=[{"role": "user", "content": content}],
            timeout_ms=60_000,  # 60s max per request — avoids 504 hangs
            min_interval=_VISION_INTERVAL,
        )
        return resp.choices[0].message.content
