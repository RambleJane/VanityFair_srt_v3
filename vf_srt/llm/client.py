"""LLM client contract for the diagnosis stages.

This milestone deliberately ships *no* network client. The diagnosis stage
takes an injected ``LLMClient`` callable so tests can run with a mock and the
real DeepSeek wiring can be added later without touching the stage logic.
"""
from __future__ import annotations

from typing import Any, Callable

# A client is any callable taking the system + user prompt and returning the
# raw model text (expected to contain a JSON object).
LLMClient = Callable[[str, str], str]


class LLMNotConfiguredError(RuntimeError):
    """Raised when a real LLM client is required but none was injected."""


def resolve_client(config: dict[str, Any] | None = None) -> LLMClient | None:
    """Return a configured client, or ``None`` when offline.

    No real network client is implemented in this milestone, so this always
    returns ``None``. Callers that receive ``None`` and still have work to do
    should raise :class:`LLMNotConfiguredError` rather than reaching the network.
    """
    del config
    return None


def request(*args: Any, **kwargs: Any) -> str:
    raise NotImplementedError(
        "Real LLM calls are not part of the offline milestone; inject an LLMClient"
    )
