from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any


def segment_to_dict(segment: Any) -> dict[str, Any]:
    if is_dataclass(segment):
        return asdict(segment)
    return dict(segment)


def make_hint(
    segment: dict[str, Any], *, span: str, suggestion: str | None,
    confidence: str, category: str, reason: str, source: str,
) -> dict[str, Any]:
    return {
        "index": int(segment.get("index", 0)),
        "raw_text": str(segment.get("raw_text", "")),
        "span": span,
        "suggestion": suggestion,
        "confidence": confidence,
        "category": category,
        "reason": reason,
        "source": source,
    }
