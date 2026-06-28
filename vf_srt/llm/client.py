"""LLM client contract and configured-client resolver for diagnosis stages."""
from __future__ import annotations

from typing import Any, Callable

# A client is any callable taking the system + user prompt and returning the
# raw model text (expected to contain a JSON object).
LLMClient = Callable[[str, str], str]


class LLMNotConfiguredError(RuntimeError):
    """Raised when a real LLM client is required but none was injected."""


def resolve_client(
    config: dict[str, Any] | None = None, *, stage: str | None = None,
) -> LLMClient | None:
    """Return a configured client, or ``None`` when no key is available.

    Builds a :class:`~vf_srt.llm.deepseek.DeepSeekClient` when the configured
    API-key environment variable is set. Returns ``None`` (rather than raising)
    when it is not, so the diagnosis stage degrades to its explicit
    :class:`LLMNotConfiguredError`. Constructing the client performs **no**
    network call; the request only happens when the client is invoked.
    """
    from .deepseek import DeepSeekClient

    try:
        return DeepSeekClient.from_config(config or {}, stage=stage)
    except LLMNotConfiguredError:
        return None


def request(*args: Any, **kwargs: Any) -> str:
    raise NotImplementedError(
        "Use resolve_client(config, stage=...) or inject an LLMClient"
    )
