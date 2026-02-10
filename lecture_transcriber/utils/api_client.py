"""Mistral API wrapper with retry and rate limiting."""

from __future__ import annotations

import base64
import json
import logging
import random
import time
from email.utils import parsedate_to_datetime
from pathlib import Path

from mistralai import Mistral
from mistralai.models import SDKError

log = logging.getLogger(__name__)

_DEFAULT_VISION_INTERVAL = 8.0
_MAX_VISION_INTERVAL = 60.0
_DEFAULT_RATE_LIMIT_BASE_WAIT = 20.0
_MAX_RATE_LIMIT_BACKOFF = 300.0
_DEFAULT_MAX_429_RETRIES = 3


class RateLimitExhaustedError(RuntimeError):
    """Raised when API calls keep returning 429 after all retries."""

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class MistralClient:
    """Thin wrapper around the Mistral SDK adding rate limiting and retries."""

    def __init__(
        self,
        api_key: str,
        max_retries: int = 10,
        rps_limit: float = 1.0,
        vision_max_tokens: int | None = None,
        vision_min_interval: float = _DEFAULT_VISION_INTERVAL,
        max_429_retries: int = _DEFAULT_MAX_429_RETRIES,
        rate_limit_base_wait: float = _DEFAULT_RATE_LIMIT_BASE_WAIT,
    ):
        self._client = Mistral(api_key=api_key)
        self._max_retries = max_retries
        self._default_min_interval = 1.0 / rps_limit if rps_limit > 0 else 0.0
        self._vision_max_tokens = vision_max_tokens if (vision_max_tokens is None or vision_max_tokens > 0) else None
        self._vision_base_interval = max(vision_min_interval, self._default_min_interval)
        self._vision_interval = self._vision_base_interval
        self._max_429_retries = max(0, max_429_retries)
        self._rate_limit_base_wait = max(1.0, rate_limit_base_wait)
        self._last_call: float = 0.0

    def _wait(self, interval: float) -> None:
        now = time.monotonic()
        elapsed = now - self._last_call
        if elapsed < interval:
            time.sleep(interval - elapsed)
        self._last_call = time.monotonic()

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        if getattr(exc, "status_code", None) == 429:
            return True
        msg = str(exc).lower()
        return "status 429" in msg or "rate limit" in msg or "rate_limited" in msg

    @staticmethod
    def _parse_retry_after(value: str | None) -> float | None:
        if not value:
            return None
        value = value.strip()
        try:
            wait_seconds = float(value)
            if wait_seconds >= 0:
                return wait_seconds
        except ValueError:
            pass

        try:
            dt = parsedate_to_datetime(value)
            wait_seconds = dt.timestamp() - time.time()
            if wait_seconds > 0:
                return wait_seconds
        except (TypeError, ValueError):
            return None
        return None

    def _extract_retry_after(self, exc: Exception) -> float | None:
        headers = getattr(exc, "headers", None)
        if headers:
            retry_after = headers.get("retry-after") or headers.get("Retry-After")
            parsed = self._parse_retry_after(retry_after)
            if parsed is not None:
                return parsed

            reset = headers.get("x-ratelimit-reset") or headers.get("X-RateLimit-Reset")
            if reset:
                try:
                    reset_seconds = float(reset) - time.time()
                    if reset_seconds > 0:
                        return reset_seconds
                except ValueError:
                    pass

        body = getattr(exc, "body", "")
        if body:
            try:
                data = json.loads(body)
                for key in ("retry_after", "retry_after_seconds"):
                    val = data.get(key)
                    if isinstance(val, (int, float)) and val > 0:
                        return float(val)
            except (json.JSONDecodeError, AttributeError):
                pass
        return None

    def _compute_rate_limit_wait(self, exc: Exception, hit: int) -> tuple[float, float | None]:
        hinted = self._extract_retry_after(exc)
        exponential = min(
            self._rate_limit_base_wait * (2 ** (hit - 1)),
            _MAX_RATE_LIMIT_BACKOFF,
        )
        wait = max(hinted or 0.0, exponential)
        wait += random.uniform(0.25, 1.0)
        return wait, hinted

    def _call_with_retry(self, fn, *args, min_interval: float | None = None, **kwargs):
        """Call *fn* with retry logic that respects 429 rate limits."""
        interval = min_interval if min_interval is not None else self._default_min_interval
        is_vision_call = min_interval is not None
        last_exc = None
        rate_limit_hits = 0
        for attempt in range(1, self._max_retries + 1):
            self._wait(interval)
            try:
                result = fn(*args, **kwargs)
                if is_vision_call and self._vision_interval > self._vision_base_interval:
                    self._vision_interval = max(
                        self._vision_base_interval,
                        self._vision_interval * 0.9,
                    )
                return result
            except SDKError as exc:
                last_exc = exc
                if self._is_rate_limit_error(exc):
                    rate_limit_hits += 1
                    if is_vision_call:
                        interval = min(interval * 1.35, _MAX_VISION_INTERVAL)
                        self._vision_interval = interval
                        log.warning(
                            "Rate limit detected, increasing vision interval to %.1fs",
                            interval,
                        )

                    if rate_limit_hits > self._max_429_retries:
                        retry_after = self._extract_retry_after(exc)
                        raise RateLimitExhaustedError(
                            f"Rate limited {rate_limit_hits} times in a row",
                            retry_after_seconds=retry_after,
                        ) from exc

                    wait, hinted = self._compute_rate_limit_wait(exc, rate_limit_hits)
                    hint_text = f" (server hint: {hinted:.1f}s)" if hinted is not None else ""
                    log.warning(
                        "Rate limited (%d/%d), waiting %.1fs%s...",
                        rate_limit_hits,
                        self._max_429_retries,
                        wait,
                        hint_text,
                    )
                    time.sleep(wait)
                    self._last_call = time.monotonic()
                    continue

                backoff = min(2 ** attempt, 30)
                log.warning(
                    "API error (attempt %d/%d): %s - retrying in %ds",
                    attempt,
                    self._max_retries,
                    exc,
                    backoff,
                )
                time.sleep(backoff)
            except Exception as exc:
                last_exc = exc
                backoff = min(2 ** attempt, 30)
                log.warning(
                    "Error (attempt %d/%d): %s - retrying in %ds",
                    attempt,
                    self._max_retries,
                    exc,
                    backoff,
                )
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
            with open(audio_path, "rb") as f:
                file_bytes = f.read()
            kwargs: dict = {
                "model": "voxtral-mini-latest",
                "file": {
                    "file_name": audio_path.name,
                    "content": file_bytes,
                },
                "timestamp_granularities": ["segment"],
                "language": language,
            }
            if context_bias:
                kwargs["context_bias"] = context_bias
            return self._client.audio.transcriptions.complete(**kwargs)

        result = self._call_with_retry(_do_transcribe)
        return result.model_dump()

    # -- OCR -------------------------------------------------------------------

    def ocr_image(self, image_path: str | Path) -> str:
        """Run Mistral OCR on an image and return extracted markdown text."""
        raw_bytes = Path(image_path).read_bytes()
        b64 = base64.b64encode(raw_bytes).decode()

        result = self._call_with_retry(
            self._client.ocr.process,
            model="mistral-ocr-latest",
            document={
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            },
            min_interval=self._vision_interval,
        )
        pages = result.pages if hasattr(result, "pages") else []
        return "\n".join(p.markdown for p in pages).strip()

    # -- Text chat (for structuring OCR output) --------------------------------

    def chat_text(
        self,
        prompt: str,
        model: str = "mistral-small-latest",
        max_tokens: int | None = None,
    ) -> str:
        """Send a text-only prompt to the chat API and return the response."""
        kwargs: dict = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        resp = self._call_with_retry(self._client.chat.complete, **kwargs)
        return resp.choices[0].message.content

    # -- Vision (legacy) -------------------------------------------------------

    def analyze_image(
        self,
        images: list[str | Path],
        prompt: str,
        model: str = "mistral-small-latest",
        max_tokens: int | None = None,
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

        target_max_tokens = self._vision_max_tokens if max_tokens is None else max_tokens
        request_kwargs: dict = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "timeout_ms": 60_000,  # 60s max per request to avoid long hangs
            "min_interval": self._vision_interval,
        }
        if target_max_tokens is not None:
            request_kwargs["max_tokens"] = target_max_tokens

        resp = self._call_with_retry(
            self._client.chat.complete,
            **request_kwargs,
        )
        return resp.choices[0].message.content
