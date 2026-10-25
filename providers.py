"""Provider clients for the TTFT and token-count arms.

Design choices
--------------
* No vendor SDKs. One OpenAI-compatible streaming client covers OpenAI,
  vLLM, Together, Fireworks and OpenRouter. Gemini uses its REST API.
* TTFT is measured from immediately before the HTTP request until the first
  non-empty *visible* content delta. The time to the first SSE event and the
  time to the first reasoning delta are recorded separately.
* Retries use exponential backoff with jitter and honour Retry-After.
* API keys are read from environment variables and never written to logs or
  result files.

These clients are intentionally small. The benchmark's scientific protocol is
in core.py / docs/METHODOLOGY.md, not hidden in a vendor SDK.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from typing import Any

import requests

log = logging.getLogger(__name__)


class ProviderError(RuntimeError):
    """A provider returned an error that is not a transient rate limit."""


class ProviderUnavailable(ProviderError):
    """The provider is disabled, unconfigured, or missing an API key."""


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


def _api_key(spec: dict[str, Any]) -> str:
    env_name = spec.get("api_key_env")
    key = os.environ.get(env_name, "") if env_name else ""
    if not key:
        key = str(spec.get("api_key_default") or "")
    if not key:
        label = spec.get("name") or spec.get("model") or spec.get("kind")
        raise ProviderUnavailable(
            f"provider {label!r}: missing environment variable {env_name!r}"
        )
    return key


def _backoff_seconds(attempt: int, response: requests.Response | None = None) -> float:
    if response is not None:
        retry_after = response.headers.get("retry-after")
        if retry_after:
            try:
                return max(0.5, min(60.0, float(retry_after)))
            except ValueError:
                pass
    # 1, 2, 4, 8, ... seconds, with jitter, capped at 30 s.
    return min(30.0, float(2 ** attempt)) * (0.5 + random.random())


def _request(
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    json_body: dict[str, Any] | None = None,
    timeout_s: float = 120.0,
    max_retries: int = 5,
    stream: bool = False,
) -> requests.Response:
    """Issue a request, retrying transient failures.

    The returned response is open. The caller must close it, especially for
    streaming responses.
    """
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        response: requests.Response | None = None
        try:
            response = requests.request(
                method,
                url,
                headers=headers,
                json=json_body,
                timeout=(10.0, timeout_s),
                stream=stream,
            )
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            delay = _backoff_seconds(attempt)
            log.warning("network error (%s); retrying in %.1fs", exc, delay)
            time.sleep(delay)
            continue

        if response.status_code in RETRYABLE_STATUS:
            delay = _backoff_seconds(attempt, response)
            body = _safe_body(response)
            response.close()
            last_error = ProviderError(
                f"HTTP {response.status_code}: {body[:400]}"
            )
            if attempt >= max_retries:
                break
            log.warning(
                "transient HTTP %s; retrying in %.1fs",
                response.status_code,
                delay,
            )
            time.sleep(delay)
            continue

        if response.status_code >= 400:
            body = _safe_body(response)
            response.close()
            raise ProviderError(f"HTTP {response.status_code}: {body[:800]}")

        return response

    raise ProviderError(f"request failed after {max_retries + 1} attempts: {last_error}")


def _safe_body(response: requests.Response) -> str:
    try:
        return response.text
    except Exception:  # pragma: no cover - defensive
        return "<unreadable body>"


def _text_from_content(value: Any) -> str:
    """Normalise OpenAI-style content, which may be a string or a list."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


# ---------------------------------------------------------------------------
# OpenAI-compatible chat completions (OpenAI, vLLM, Together, ...)
# ---------------------------------------------------------------------------


class OpenAICompatibleClient:
    kind = "openai"

    def __init__(self, spec: dict[str, Any]):
        self.spec = spec
        self.name = str(spec.get("name") or spec.get("model") or "openai-compatible")
        self.model = str(spec["model"])
        self.base_url = str(spec.get("base_url") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = _api_key(spec)
        self.timeout_s = float(spec.get("timeout_s", 120))
        self.max_retries = int(spec.get("max_retries", 5))
        self.extra_body = dict(spec.get("extra_body") or {})
        self.extra_headers = dict(spec.get("extra_headers") or {})
        self.max_tokens_field = str(spec.get("max_tokens_field", "max_tokens"))
        self.include_usage = bool(spec.get("include_usage", True))

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        headers.update(self.extra_headers)
        return headers

    def _payload(
        self,
        user_text: str,
        *,
        system_prompt: str | None,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_text})
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
            self.max_tokens_field: max_tokens,
        }
        if self.include_usage:
            payload["stream_options"] = {"include_usage": True}
        # User-supplied fields win, so a provider-specific switch can be added
        # without touching the code.
        payload.update(self.extra_body)
        return payload

    def stream_chat(
        self,
        user_text: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int = 1,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        payload = self._payload(
            user_text,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        t0 = time.perf_counter()
        response = _request(
            "POST",
            url,
            headers=self._headers(),
            json_body=payload,
            timeout_s=self.timeout_s,
            max_retries=self.max_retries,
            stream=True,
        )

        first_event_ms: float | None = None
        first_any_ms: float | None = None
        first_content_ms: float | None = None
        pieces: list[str] = []
        usage: dict[str, Any] | None = None
        request_id = response.headers.get("x-request-id") or response.headers.get("request-id")
        malformed_lines = 0

        try:
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", "replace") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data:
                    continue
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    malformed_lines += 1
                    log.debug("skipping malformed SSE line: %.200s", line)
                    continue

                now_ms = (time.perf_counter() - t0) * 1000.0
                if first_event_ms is None:
                    first_event_ms = now_ms

                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]

                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    content = _text_from_content(delta.get("content"))
                    reasoning = _text_from_content(
                        delta.get("reasoning_content") or delta.get("reasoning")
                    )
                    if reasoning and first_any_ms is None:
                        first_any_ms = now_ms
                    if content:
                        if first_any_ms is None:
                            first_any_ms = now_ms
                        if first_content_ms is None:
                            first_content_ms = now_ms
                        pieces.append(content)
        finally:
            response.close()

        total_ms = (time.perf_counter() - t0) * 1000.0
        text = "".join(pieces)
        if first_content_ms is None:
            raise ProviderError(
                f"provider {self.name!r} ended the stream without visible content "
                f"(malformed SSE lines: {malformed_lines})"
            )

        input_tokens = _usage_int(usage, "prompt_tokens", "input_tokens")
        output_tokens = _usage_int(usage, "completion_tokens", "output_tokens")
        cached_tokens = None
        if isinstance(usage, dict):
            details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details")
            if isinstance(details, dict):
                cached_tokens = _as_int(details.get("cached_tokens"))

        return {
            "status": "ok",
            "provider": self.name,
            "kind": self.kind,
            "model": self.model,
            "base_url": self.base_url,
            "ttft_ms": first_content_ms,
            "first_event_ms": first_event_ms,
            "first_any_delta_ms": first_any_ms,
            "total_ms": total_ms,
            "text": text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cached_tokens,
            "usage": usage,
            "request_id": request_id,
            "malformed_sse_lines": malformed_lines,
        }


# ---------------------------------------------------------------------------
# Gemini REST API
# ---------------------------------------------------------------------------


class GeminiClient:
    kind = "gemini"

    def __init__(self, spec: dict[str, Any]):
        self.spec = spec
        self.name = str(spec.get("name") or spec.get("model") or "gemini")
        self.model = str(spec["model"])
        self.base_url = str(
            spec.get("base_url") or "https://generativelanguage.googleapis.com/v1beta"
        ).rstrip("/")
        self.api_key = _api_key(spec)
        self.timeout_s = float(spec.get("timeout_s", 120))
        self.max_retries = int(spec.get("max_retries", 5))
        self.thinking_budget = spec.get("thinking_budget", 0)

    def _model_name(self) -> str:
        return self.model.removeprefix("models/")

    def _headers(self) -> dict[str, str]:
        return {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

    def count_tokens(self, text: str, model: str | None = None) -> int:
        model_name = (model or self.model).removeprefix("models/")
        url = f"{self.base_url}/models/{model_name}:countTokens"
        response = _request(
            "POST",
            url,
            headers=self._headers(),
            json_body={"contents": [{"parts": [{"text": text}]}]},
            timeout_s=self.timeout_s,
            max_retries=self.max_retries,
        )
        try:
            payload = response.json()
        finally:
            response.close()
        total = payload.get("totalTokens")
        if total is None:
            raise ProviderError(f"Gemini countTokens returned no totalTokens: {payload}")
        return int(total)

    def _stream_once(
        self,
        user_text: str,
        *,
        system_prompt: str | None,
        max_tokens: int,
        temperature: float,
        thinking_budget: int | None,
    ) -> dict[str, Any]:
        model_name = self._model_name()
        url = f"{self.base_url}/models/{model_name}:streamGenerateContent?alt=sse"
        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": user_text}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if system_prompt:
            body["systemInstruction"] = {
                "role": "system",
                "parts": [{"text": system_prompt}],
            }
        if thinking_budget is not None:
            body["generationConfig"]["thinkingConfig"] = {
                "thinkingBudget": int(thinking_budget)
            }

        t0 = time.perf_counter()
        response = _request(
            "POST",
            url,
            headers=self._headers(),
            json_body=body,
            timeout_s=self.timeout_s,
            max_retries=self.max_retries,
            stream=True,
        )

        first_event_ms: float | None = None
        first_any_ms: float | None = None
        first_content_ms: float | None = None
        pieces: list[str] = []
        usage: dict[str, Any] | None = None
        model_version: str | None = None
        block_reason: str | None = None
        malformed_lines = 0

        try:
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", "replace") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data:
                    continue
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    malformed_lines += 1
                    continue

                now_ms = (time.perf_counter() - t0) * 1000.0
                if first_event_ms is None:
                    first_event_ms = now_ms
                if isinstance(event.get("usageMetadata"), dict):
                    usage = event["usageMetadata"]
                if event.get("modelVersion"):
                    model_version = event["modelVersion"]

                feedback = event.get("promptFeedback") or {}
                if feedback.get("blockReason"):
                    block_reason = str(feedback["blockReason"])

                for candidate in event.get("candidates") or []:
                    content = candidate.get("content") or {}
                    for part in content.get("parts") or []:
                        text = part.get("text")
                        if isinstance(text, str) and text:
                            if first_any_ms is None:
                                first_any_ms = now_ms
                            if first_content_ms is None:
                                first_content_ms = now_ms
                            pieces.append(text)
        finally:
            response.close()

        total_ms = (time.perf_counter() - t0) * 1000.0
        if first_content_ms is None:
            raise ProviderError(
                f"Gemini {self.name!r} ended without visible content"
                + (f" (blocked: {block_reason})" if block_reason else "")
                + f" (malformed SSE lines: {malformed_lines})"
            )

        input_tokens = _usage_int(usage, "promptTokenCount")
        output_tokens = _usage_int(usage, "candidatesTokenCount")
        cached_tokens = _usage_int(usage, "cachedContentTokenCount")
        return {
            "status": "ok",
            "provider": self.name,
            "kind": self.kind,
            "model": self.model,
            "model_version": model_version,
            "base_url": self.base_url,
            "ttft_ms": first_content_ms,
            "first_event_ms": first_event_ms,
            "first_any_delta_ms": first_any_ms,
            "total_ms": total_ms,
            "text": "".join(pieces),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cached_tokens,
            "usage": usage,
            "request_id": None,
            "malformed_sse_lines": malformed_lines,
        }

    def stream_chat(
        self,
        user_text: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int = 1,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        try:
            return self._stream_once(
                user_text,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                thinking_budget=self.thinking_budget,
            )
        except ProviderError as exc:
            # Some Gemini models reject thinkingConfig. Retry once without it
            # so an otherwise valid run does not die on a model-specific field.
            if self.thinking_budget is not None and "HTTP 400" in str(exc):
                log.warning(
                    "Gemini rejected thinkingConfig; retrying without it: %s", exc
                )
                return self._stream_once(
                    user_text,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    thinking_budget=None,
                )
            raise


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_provider(spec: dict[str, Any]):
    # Enabled/selection is handled by the caller (run_ttft). An explicit
    # `--providers <name>` is allowed to run an entry whose `enabled` flag is
    # false, so this factory must not re-apply the config filter.
    kind = spec.get("kind")
    if kind == "openai":
        return OpenAICompatibleClient(spec)
    if kind == "gemini":
        return GeminiClient(spec)
    raise ProviderUnavailable(f"unknown provider kind: {kind!r}")


def provider_label(spec: dict[str, Any]) -> str:
    return str(spec.get("name") or spec.get("model") or spec.get("kind"))


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _usage_int(usage: dict[str, Any] | None, *keys: str) -> int | None:
    if not isinstance(usage, dict):
        return None
    for key in keys:
        value = _as_int(usage.get(key))
        if value is not None:
            return value
    return None
