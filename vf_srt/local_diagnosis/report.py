from __future__ import annotations

from typing import Any

from ..core.json_utils import write_json


def build_segment_hints(hints: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for hint in hints:
        index = str(int(hint.get("index", 0)))
        output.setdefault(index, []).append({
            "type": hint.get("category"),
            "span": hint.get("span") or hint.get("candidate"),
            "suggestion": hint.get("suggestion"),
            "confidence": hint.get("confidence"),
            "source": hint.get("source"),
        })
    return output


def write_local_diagnosis(result: dict[str, Any], target: Any) -> None:
    write_json(target, result)
