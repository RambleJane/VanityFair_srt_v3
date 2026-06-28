from __future__ import annotations

import re
from typing import Any

from .models import make_hint, segment_to_dict
from .name_rules import NAME_RULES


_PATTERNS = (
    re.compile(r"阿[\u4e00-\u9fff]"),
    re.compile(r"[\u4e00-\u9fff]{1,3}(?:哥|姐|叔|生)"),
    re.compile(r"[\u4e00-\u9fff]{1,3}老板"),
)
_EXCLUDED = {"啊", "呀", "啦", "喇", "咩", "啫", "嗱", "喂", "唉", "哦", "嗯"}


def detect_unknown_name_candidates(
    segments: list[Any], knowledge: dict[str, Any], reference_profile: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    del config
    known = set(knowledge.get("characters", {}).get("official_names", []))
    known.update(knowledge.get("characters", {}).get("aliases", []))
    known.update(reference_profile.get("names", {}))
    known.update(knowledge.get("confirmed_terms", []))
    known.update(NAME_RULES)
    output: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for value in segments:
        segment = segment_to_dict(value)
        raw_text = str(segment.get("raw_text", ""))
        for pattern in _PATTERNS:
            for match in pattern.finditer(raw_text):
                candidate = match.group(0)
                key = (int(segment.get("index", 0)), candidate)
                if candidate in known or candidate in _EXCLUDED or key in seen:
                    continue
                seen.add(key)
                hint = make_hint(
                    segment, span=candidate, suggestion=None, confidence="low",
                    category="unknown_name_candidate",
                    reason="疑似短人名或称谓，但不在官方角色表、前8集参考字幕和confirmed glossary中。",
                    source="local_unknown_name_rule",
                )
                hint["candidate"] = candidate
                output.append(hint)
    return output
