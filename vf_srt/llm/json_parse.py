"""Defensive structured-output parsing for LLM responses."""
from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _strip_fences(text: str) -> str:
    match = _FENCE.search(text)
    return match.group(1) if match else text


def _first_json_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` block, ignoring braces in strings."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for pos in range(start, len(text)):
        char = text[pos]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : pos + 1]
    return None


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from possibly noisy model output.

    Tolerates code fences and leading/trailing prose. Raises ``ValueError`` if
    no JSON object can be recovered (callers decide how to degrade).
    """
    candidate = _strip_fences(text or "").strip()
    for attempt in (candidate, _first_json_object(candidate) or ""):
        if not attempt:
            continue
        try:
            parsed = json.loads(attempt)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("No JSON object found in LLM response")
