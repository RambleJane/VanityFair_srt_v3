from __future__ import annotations

from typing import Any


def get_result_utterances(result: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        utterances = result["body"]["result"]["utterances"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Doubao result missing body.result.utterances") from exc
    if not isinstance(utterances, list):
        raise ValueError("Doubao body.result.utterances must be a list")
    return utterances


def validate_doubao_result(result: dict[str, Any]) -> None:
    for utterance_index, utterance in enumerate(get_result_utterances(result), start=1):
        if not isinstance(utterance, dict):
            raise ValueError(f"Utterance {utterance_index} must be an object")
        missing = [key for key in ("start_time", "end_time", "text", "words") if key not in utterance]
        if missing:
            raise ValueError(f"Utterance {utterance_index} missing: {', '.join(missing)}")
        if not isinstance(utterance["words"], list):
            raise ValueError(f"Utterance {utterance_index}.words must be a list")
        for word_index, word in enumerate(utterance["words"], start=1):
            if not isinstance(word, dict):
                raise ValueError(f"Utterance {utterance_index} word {word_index} must be an object")
            missing = [key for key in ("start_time", "end_time", "text") if key not in word]
            if missing:
                raise ValueError(
                    f"Utterance {utterance_index} word {word_index} missing: {', '.join(missing)}"
                )


def get_audio_duration(result: dict[str, Any]) -> float | None:
    candidates = (
        result.get("body", {}).get("audio_info", {}).get("duration"),
        result.get("body", {}).get("result", {}).get("duration"),
    )
    for value in candidates:
        if isinstance(value, (int, float)):
            return float(value) / 1000.0
    return None
