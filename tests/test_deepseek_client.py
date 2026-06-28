from __future__ import annotations

import copy
import json

import pytest

from vf_srt.core.config import DEFAULT_CONFIG
from vf_srt.llm.client import LLMNotConfiguredError, resolve_client
from vf_srt.llm.deepseek import DeepSeekClient, DeepSeekError

_FAKE_KEY = "sk-test-DO-NOT-LOG-1234567890"


def _config() -> dict:
    return copy.deepcopy(DEFAULT_CONFIG)


class FakeTransport:
    """Captures every request and returns scripted (status, text) tuples."""

    def __init__(self, responses) -> None:
        self.calls: list[dict] = []
        self._responses = list(responses)

    def __call__(self, url, headers, body, timeout):
        self.calls.append({
            "url": url, "headers": headers, "body": body, "timeout": timeout,
        })
        item = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        if isinstance(item, Exception):
            raise item
        return item


def _chat_response(content: str) -> tuple[int, str]:
    return 200, json.dumps({"choices": [{"message": {"content": content}}]})


def test_from_config_missing_env_raises_clear_error(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(LLMNotConfiguredError) as excinfo:
        DeepSeekClient.from_config(_config())
    message = str(excinfo.value)
    assert "DEEPSEEK_API_KEY" in message
    assert _FAKE_KEY not in message  # never leak any key material


def test_call_builds_openai_compatible_request(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", _FAKE_KEY)
    transport = FakeTransport([_chat_response('{"ok": true}')])
    client = DeepSeekClient.from_config(_config(), transport=transport, sleep=lambda _s: None)

    out = client("SYS PROMPT", "USER PROMPT")

    assert out == '{"ok": true}'
    call = transport.calls[0]
    assert call["url"] == "https://api.deepseek.com/chat/completions"
    assert call["headers"]["Authorization"] == f"Bearer {_FAKE_KEY}"
    payload = json.loads(call["body"].decode("utf-8"))
    assert payload["model"] == "deepseek-chat"
    assert payload["temperature"] == 0.3
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["messages"][0] == {"role": "system", "content": "SYS PROMPT"}
    assert payload["messages"][1] == {"role": "user", "content": "USER PROMPT"}


def test_key_never_leaks_in_repr_or_errors(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", _FAKE_KEY)
    transport = FakeTransport([(401, '{"error": "unauthorized"}')])
    client = DeepSeekClient.from_config(_config(), transport=transport, sleep=lambda _s: None)

    assert _FAKE_KEY not in repr(client)
    with pytest.raises(DeepSeekError) as excinfo:
        client("s", "u")
    assert _FAKE_KEY not in str(excinfo.value)
    assert "401" in str(excinfo.value)


def test_retries_on_transient_status_then_succeeds(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", _FAKE_KEY)
    transport = FakeTransport([
        (503, "service unavailable"),
        (429, "rate limited"),
        _chat_response("recovered"),
    ])
    client = DeepSeekClient.from_config(
        _config(), transport=transport, sleep=lambda _s: None
    )

    assert client("s", "u") == "recovered"
    assert len(transport.calls) == 3


def test_retries_exhausted_raises_without_key(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", _FAKE_KEY)
    transport = FakeTransport([(500, "boom")])
    config = _config()
    config["deepseek"]["max_retries"] = 2
    client = DeepSeekClient.from_config(config, transport=transport, sleep=lambda _s: None)

    with pytest.raises(DeepSeekError) as excinfo:
        client("s", "u")
    assert len(transport.calls) == 3  # 1 initial + 2 retries
    assert _FAKE_KEY not in str(excinfo.value)


def test_4xx_does_not_retry(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", _FAKE_KEY)
    transport = FakeTransport([(400, "bad request")])
    client = DeepSeekClient.from_config(_config(), transport=transport, sleep=lambda _s: None)

    with pytest.raises(DeepSeekError):
        client("s", "u")
    assert len(transport.calls) == 1  # no retry on 400


def test_malformed_response_raises_deepseek_error(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", _FAKE_KEY)
    transport = FakeTransport([(200, "not json at all")])
    client = DeepSeekClient.from_config(_config(), transport=transport, sleep=lambda _s: None)

    with pytest.raises(DeepSeekError):
        client("s", "u")


def test_resolve_client_none_when_env_missing(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert resolve_client(_config()) is None


def test_resolve_client_builds_deepseek_when_configured(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", _FAKE_KEY)
    client = resolve_client(_config())
    assert isinstance(client, DeepSeekClient)
    # Constructing the client must not perform any network call (nothing to assert
    # beyond: we got here without one, since the default transport was never invoked).


def test_respects_custom_base_url_and_model(monkeypatch) -> None:
    monkeypatch.setenv("DS_KEY", _FAKE_KEY)
    config = _config()
    config["deepseek"].update({
        "api_key_env": "DS_KEY",
        "base_url": "https://proxy.example.com/v1/",
        "model": "deepseek-reasoner",
        "response_format_json": False,
    })
    transport = FakeTransport([_chat_response("x")])
    client = DeepSeekClient.from_config(config, transport=transport, sleep=lambda _s: None)

    client("s", "u")
    call = transport.calls[0]
    assert call["url"] == "https://proxy.example.com/v1/chat/completions"
    payload = json.loads(call["body"].decode("utf-8"))
    assert payload["model"] == "deepseek-reasoner"
    assert "response_format" not in payload


def test_pre_review_stage_settings_reach_request_payload(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", _FAKE_KEY)
    config = _config()
    config["deepseek"]["stages"]["pre_review_diagnosis"].update({
        "model": "stage-model",
        "thinking": "enabled",
        "max_tokens": 4321,
        "temperature": 0.15,
        "timeout": 77,
        "max_retries": 1,
    })
    transport = FakeTransport([_chat_response("{}")])
    client = DeepSeekClient.from_config(
        config, stage="pre_review_diagnosis",
        transport=transport, sleep=lambda _s: None,
    )

    client("s", "u")

    call = transport.calls[0]
    payload = json.loads(call["body"].decode("utf-8"))
    assert client.stage == "pre_review_diagnosis"
    assert call["timeout"] == 77
    assert client.max_retries == 1
    assert payload["model"] == "stage-model"
    assert payload["thinking"] == {"type": "enabled"}
    assert payload["max_tokens"] == 4321
    assert payload["temperature"] == 0.15


def test_resolve_client_uses_requested_stage(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", _FAKE_KEY)
    client = resolve_client(_config(), stage="pre_review_diagnosis")
    assert isinstance(client, DeepSeekClient)
    assert client.stage == "pre_review_diagnosis"
    assert client.thinking == "enabled"
    assert client.max_tokens == 4000
