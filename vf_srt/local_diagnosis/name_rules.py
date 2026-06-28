from __future__ import annotations

from collections import defaultdict
from typing import Any

from .models import make_hint, segment_to_dict


NAME_RULES: dict[str, tuple[str, str, str]] = {
    "徐少良": ("徐绍良", "high", "官方角色名为徐绍良；徐少良是常见同音误写。"),
    "石少良": ("徐绍良", "high", "官方角色名为徐绍良；石少良是近音误识。"),
    "阿梁": ("阿良", "high", "官方角色徐绍良别名为阿良，阿梁是常见近音误听。"),
    "阿亮": ("阿良", "high", "官方角色徐绍良别名为阿良，且角色表中无主要人物阿亮。"),
    "阿娘": ("阿良", "medium", "阿娘可能是阿良的近音误听，必须结合人物与片场语境确认。"),
    "梁啊": ("良啊", "high", "称呼语境中梁啊常为良啊的近音误听。"),
    "娘啊": ("良啊", "medium", "称呼语境中娘啊可能为良啊，需结合上下文确认。"),
    "洪沙": ("洪生", "medium", "洪沙可能是洪生的近音误听。"),
    "龚老板": ("孔老板", "medium", "剧情资料中有孔氏电影公司，老板姓氏需结合场景确认。"),
    "大清明": ("邵大明", "high", "官方角色名为邵大明，大清明是近音误识候选。"),
}
_ALIANG_CONTEXT = ("徐绍良", "阿良", "阿霞", "替身", "片场", "演员", "训练班", "契爷", "师母")


def _profile_count(profile: dict[str, Any], name: str) -> int:
    return int(profile.get("names", {}).get(name, 0))


def diagnose_character_names(
    segments: list[Any], knowledge: dict[str, Any], reference_profile: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    del config
    rows = [segment_to_dict(segment) for segment in segments]
    episode_text = "\n".join(str(row.get("raw_text", "")) for row in rows)
    official_names = set(knowledge.get("characters", {}).get("official_names", []))
    aliases = set(knowledge.get("characters", {}).get("aliases", []))
    output: list[dict[str, Any]] = []
    for position, segment in enumerate(rows):
        raw_text = str(segment.get("raw_text", ""))
        context = " ".join(
            str(rows[index].get("raw_text", ""))
            for index in range(max(0, position - 2), min(len(rows), position + 3))
        )
        occupied: list[tuple[int, int]] = []
        for span, (suggestion, default_confidence, base_reason) in NAME_RULES.items():
            start = raw_text.find(span)
            if start < 0 or any(start < end and start + len(span) > begin for begin, end in occupied):
                continue
            confidence = default_confidence
            if span in {"阿娘", "娘啊"} and any(marker in context for marker in _ALIANG_CONTEXT):
                confidence = "high"
            if span == "洪沙" and "洪生" in episode_text:
                confidence = "high"
            evidence: list[str] = [base_reason]
            count_key = "阿良" if suggestion in {"阿良", "良啊"} else suggestion
            count = _profile_count(reference_profile, count_key)
            if count:
                evidence.append(f"前8集人工字幕画像中“{count_key}”出现 {count} 次")
            if suggestion in official_names or suggestion in aliases:
                evidence.append("该建议写法见官方角色表或别名")
            output.append(make_hint(
                segment, span=span, suggestion=suggestion, confidence=confidence,
                category="character_name", reason="；".join(evidence),
                source="characters_official+reference_profile",
            ))
            occupied.append((start, start + len(span)))
    return output


def build_likely_characters(
    hints: list[dict[str, Any]], knowledge: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    alias_to_name = (
        (knowledge or {}).get("characters", {}).get("alias_to_name", {})
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for hint in hints:
        if hint.get("category") == "character_name" and hint.get("suggestion"):
            suggestion = str(hint["suggestion"])
            lookup = suggestion[:-1] if suggestion.endswith("啊") else suggestion
            canonical = str(
                alias_to_name.get(suggestion, alias_to_name.get(lookup, suggestion))
            )
            grouped[canonical].append(hint)
    return [
        {
            "name": name,
            "aliases_seen": sorted({str(item["span"]) for item in items}),
            "evidence_count": len(items),
            "confidence": "high" if any(item.get("confidence") == "high" for item in items) else "medium",
        }
        for name, items in sorted(grouped.items(), key=lambda pair: (-len(pair[1]), pair[0]))
    ]
