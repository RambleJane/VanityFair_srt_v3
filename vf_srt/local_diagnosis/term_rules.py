from __future__ import annotations

from typing import Any

from .models import make_hint, segment_to_dict


TERM_RULES: dict[str, dict[str, str]] = {
    "𫪈": {"suggestion": "喺", "confidence": "high", "category": "cantonese_word", "reason": "𫪈常见为“喺”的ASR异体或误识。"},
    "𫩥": {"suggestion": "嚿", "confidence": "medium", "category": "cantonese_word", "reason": "金额语境中常见为“嚿水”的“嚿”。"},
    "恁": {"suggestion": "谂", "confidence": "medium", "category": "cantonese_word", "reason": "表示“想”的语境中，恁常为谂的误写。"},
    "游浮": {"suggestion": "游埠", "confidence": "high", "category": "term", "reason": "粤语“游埠”指旅游，常被ASR误听为游浮。"},
    "五楼云吞": {"suggestion": "五柳云吞", "confidence": "high", "category": "food_term", "reason": "面档食物名应重点核对为五柳云吞。"},
    "大勇": {"suggestion": "大蓉", "confidence": "medium", "category": "food_term", "reason": "面档云吞面规格语境中可能为“大蓉”。"},
    "打滑稽": {"suggestion": "打麻雀", "confidence": "medium", "category": "mahjong_term", "reason": "麻将场景中可能为“打麻雀”。"},
    "抖音": {"suggestion": "抖阵", "confidence": "high", "category": "period_anachronism", "reason": "70年代语境不应出现“抖音”，片场语境可能为“抖阵”。"},
}


def diagnose_terms(
    segments: list[Any], knowledge: dict[str, Any], reference_profile: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    del config
    confirmed = set(knowledge.get("confirmed_terms", []))
    profile_terms = reference_profile.get("terms", {})
    output: list[dict[str, Any]] = []
    for value in segments:
        segment = segment_to_dict(value)
        raw_text = str(segment.get("raw_text", ""))
        for span, rule in TERM_RULES.items():
            if span not in raw_text:
                continue
            suggestion = rule["suggestion"]
            evidence = rule["reason"]
            count = int(profile_terms.get(suggestion, 0))
            if count:
                evidence += f" 前8集画像中“{suggestion}”出现 {count} 次。"
            source = "glossary_confirmed" if suggestion in confirmed or suggestion in knowledge.get("confirmed_glossary_text", "") else "local_term_rule"
            output.append(make_hint(
                segment, span=span, suggestion=suggestion,
                confidence=rule["confidence"], category=rule["category"],
                reason=evidence, source=source,
            ))
    return output
