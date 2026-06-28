"""DeepSeek chat client (OpenAI-compatible), implemented without a live call.

Security contract:
- The API key is read from the environment variable named by
  ``deepseek.api_key_env`` — never from the config file or the repo.
- The key is never logged, cached, put in ``repr``, or included in any
  exception message.
- The network transport is injectable so unit tests run fully offline.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from os import environ
from typing import Any, Callable

from ..core.config import resolve_stage_llm
from .client import LLMNotConfiguredError

# (url, headers, body, timeout) -> (status_code, response_text)
Transport = Callable[[str, dict[str, str], bytes, float], tuple[int, str]]

_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


class DeepSeekError(RuntimeError):
    """A DeepSeek API failure. Never carries the API key."""


def _urllib_transport(
    url: str, headers: dict[str, str], body: bytes, timeout: float
) -> tuple[int, str]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:  # 4xx/5xx with a body
        try:
            text = exc.read().decode("utf-8")
        except Exception:  # pragma: no cover - defensive
            text = ""
        return exc.code, text


class DeepSeekClient:
    """Callable ``(system, user) -> str`` returning the raw model message text."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        timeout: float = 120.0,
        max_retries: int = 3,
        temperature: float = 0.3,
        response_format_json: bool = True,
        thinking: str | None = None,
        max_tokens: int | None = None,
        stage: str | None = None,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key:
            # Defensive: callers should use from_config, which validates first.
            raise LLMNotConfiguredError("DeepSeek API key is empty")
        self._api_key = api_key  # never logged / repr'd / serialized
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = float(timeout)
        self.max_retries = max(0, int(max_retries))
        self.temperature = float(temperature)
        self.response_format_json = bool(response_format_json)
        self.thinking = thinking if thinking in {"enabled", "disabled"} else None
        self.max_tokens = int(max_tokens) if max_tokens is not None else None
        self.stage = stage
        self._transport = transport or _urllib_transport
        self._sleep = sleep

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        stage: str | None = None,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> "DeepSeekClient":
        deepseek = (config or {}).get("deepseek", {}) if isinstance(config, dict) else {}
        settings = {
            **deepseek,
            **(resolve_stage_llm(config, stage) if stage else {}),
        }
        api_key_env = str(settings.get("api_key_env", "DEEPSEEK_API_KEY"))
        api_key = environ.get(api_key_env, "").strip()
        if not api_key:
            raise LLMNotConfiguredError(
                f"DeepSeek API key not found: set the {api_key_env} environment "
                "variable (the key is read from the environment, never from config)"
            )
        return cls(
            api_key,
            base_url=str(settings.get("base_url", "https://api.deepseek.com")),
            model=str(settings.get("model", "deepseek-chat")),
            timeout=float(settings.get("timeout", 120)),
            max_retries=int(settings.get("max_retries", 3)),
            temperature=float(settings.get("temperature", 0.3)),
            response_format_json=bool(settings.get("response_format_json", True)),
            thinking=str(settings.get("thinking")) if settings.get("thinking") is not None else None,
            max_tokens=(
                int(settings.get("max_tokens", settings.get("max_output_tokens")))
                if settings.get("max_tokens", settings.get("max_output_tokens")) is not None
                else None
            ),
            stage=stage,
            transport=transport,
            sleep=sleep,
        )

    def __repr__(self) -> str:  # never expose the key
        suffix = f", stage={self.stage!r}" if self.stage else ""
        return f"DeepSeekClient(model={self.model!r}, base_url={self.base_url!r}{suffix})"

    @property
    def url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _payload(self, system: str, user: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.response_format_json:
            payload["response_format"] = {"type": "json_object"}
        if self.thinking is not None:
            payload["thinking"] = {"type": self.thinking}
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        return payload

    def __call__(self, system: str, user: str) -> str:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        body = json.dumps(self._payload(system, user), ensure_ascii=False).encode("utf-8")

        last_error: str | None = None
        for attempt in range(self.max_retries + 1):
            try:
                status, text = self._transport(self.url, headers, body, self.timeout)
            except (urllib.error.URLError, TimeoutError, OSError):
                # Network-level failure: retry without surfacing any header/key.
                last_error = "network error"
                status, text = -1, ""
            else:
                if status == 200:
                    return self._extract_content(text)
                if status not in _RETRY_STATUS:
                    raise DeepSeekError(
                        f"DeepSeek request failed with status {status}: "
                        f"{_safe_snippet(text)}"
                    )
                last_error = f"status {status}"

            if attempt < self.max_retries:
                self._sleep(min(2.0 * (attempt + 1), 8.0))

        raise DeepSeekError(
            f"DeepSeek request failed after {self.max_retries + 1} attempts ({last_error})"
        )

    @staticmethod
    def _extract_content(text: str) -> str:
        try:
            data = json.loads(text)
            return str(data["choices"][0]["message"]["content"])
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise DeepSeekError(
                f"Unexpected DeepSeek response shape: {type(exc).__name__}"
            ) from None


def _safe_snippet(text: str, limit: int = 300) -> str:
    snippet = (text or "").strip().replace("\n", " ")
    return snippet[:limit]
