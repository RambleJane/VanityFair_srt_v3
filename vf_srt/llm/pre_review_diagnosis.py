"""pre_review_diagnosis: LLM-assisted, hint-only review of local_review output.

Reads ``cache/local_review/{episode}_local_review.json`` and asks an injected
LLM client to produce human-review notes *only*. It never rewrites subtitles,
never translates, never drafts Yue lines, and never auto-applies anything.

The stage accepts an injected ``LLMClient`` for offline tests and can also be
run through the configured DeepSeek client at the CLI/pipeline boundary.
"""
from __future__ import annotations

import json
import re
from typing import Any

from ..core.cache import use_cache
from ..core.config import resolve_stage_llm
from ..core.json_utils import read_json, write_json
from ..knowledge.diagnosis_capsule import build_diagnosis_knowledge_capsule
from ..local_diagnosis.knowledge_loader import load_local_knowledge
from ..local_diagnosis.reference_profile import load_or_build_reference_profile
from ..local_review import run_local_review
from .batching import iter_batches
from .client import LLMClient, LLMNotConfiguredError
from .json_parse import parse_json_object

SYSTEM_PROMPT = (
    "你是 1977 年香港电视剧《大亨》的粤语字幕审阅诊断助手。\n"
    "你只提供人工审阅建议。\n"
    "不得改写字幕正文。\n"
    "不得生成最终字幕。\n"
    "不得生成 yue_draft 或任何字幕底稿。\n"
    "不得翻译成普通话或其他语言。\n"
    "不得凭空补对白；缺失内容只能标为不确定。\n"
    "local_review_hints 是本地规则提示，可能有误，需要结合上下文判断。\n"
    "uncertain glossary（uncertain_terms）不能自动采用。\n"
    "raw_text 和当前上下文优先。\n"
    "所有建议必须标明不确定性（confidence: high/medium/low）。\n"
    "你的所有建议都不会被自动应用（do_not_auto_apply）。\n"
    "只输出 index，不要输出或改写 raw_text；程序会从 local_review 回填事实字段。\n"
    "不要使用“同上”“同前”“同第N条”“same as above”等相对引用，每条理由必须独立完整。\n"
    "建议必须拆成 observed_span、action、suggested_span；不要把局部替换、整行改写和说明混在 suggestion。\n"
    "action 只能是 replace/listen/keep/uncertain。除非整行极高确定，否则 suggested_full_text 必须为 null。\n"
    "数字、尺码、金额、型号、码数（如34A/32/40/2/4）不得给 high confidence；不明确时 action=uncertain 且 confidence=low。\n"
    "罕见粤语字、异体字、OCR/ASR奇字只能给局部 observed_span -> suggested_span，并要求人工确认，不得直接整行改写。\n"
    "输出严格 JSON，且只输出一个 JSON 对象，不要附加解释文字。"
)

_OUTPUT_SCHEMA_HINT = {
    "summary": {
        "scene_overview": "string",
        "main_characters": ["string"],
        "relationships": ["string"],
        "important_events": ["string"],
        "tone_style": "string",
    },
    "proper_nouns": [
        {"canonical_name": "string", "aliases_seen": ["string"],
         "type": "person/place/company/term",
         "source": "official/confirmed_glossary/reference_profile/model_infer",
         "status": "confirmed/inferred/uncertain", "evidence_indices": [0],
         "evidence": "string", "confidence": "high/medium/low"}
    ],
    "possible_asr_errors": [
        {"index": 0, "observed_span": "string",
         "action": "replace/listen/keep/uncertain", "suggested_span": "string/null",
         "suggested_full_text": None, "problem": "string",
         "reason": "string", "confidence": "high/medium/low",
         "do_not_auto_apply": True}
    ],
    "line_hints": [
        {"index": 0, "category": "name/term/asr/context/segmentation/other",
         "observed_span": "string/null", "action": "replace/listen/keep/uncertain",
         "suggested_span": "string/null", "hint": "string", "reason": "string",
         "risk_level": "high/medium/low", "do_not_auto_apply": True}
    ],
    "uncertain_points": ["string"],
}

_CONFIDENCE_VALUES = frozenset({"high", "medium", "low"})
_ACTION_VALUES = frozenset({"replace", "listen", "keep", "uncertain"})
_HINT_CATEGORIES = frozenset({"name", "term", "asr", "context", "segmentation", "other"})
_PROPER_NOUN_TYPES = frozenset({"person", "place", "company", "term"})
_RELATIVE_REFERENCE = re.compile(
    r"同\s*(?:上(?:一条)?|前(?:一条)?|(?:第\s*)?(?:index\s*)?\d+\s*(?:条)?)|"
    r"与\s*(?:第\s*)?(?:index\s*)?\d+\s*(?:条)?\s*相同|"
    r"如上|见上|参考上条|"
    r"same\s+as\s+(?:above|before|previous|index\s*\d+)",
    re.IGNORECASE,
)
_LISTEN_MARKERS = ("听辨", "听原", "人工确认", "需确认", "不确定", "无法确定")
_KEEP_MARKERS = ("保留原文", "无需修改", "不需修改", "误报", "可忽略")
_NUMBER_CONTEXT_MARKERS = ("数字", "尺码", "金额", "型号", "码数")
_COMMON_PROPER_NOUNS = frozenset({
    "导演", "老板", "先生", "小姐", "师傅", "训练班", "片场",
})
_RAW_MODEL_PREVIEW_CHARS = 3000


def _validation_state() -> dict[str, int]:
    return {
        "invalid_index_count": 0,
        "invalid_schema_count": 0,
        "relative_reference_count": 0,
        "raw_text_repaired_count": 0,
        "confidence_adjusted_count": 0,
        "proper_noun_filtered_count": 0,
    }


def _contains_relative_reference(value: Any) -> bool:
    if isinstance(value, str):
        return bool(_RELATIVE_REFERENCE.search(value))
    if isinstance(value, dict):
        return any(_contains_relative_reference(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_relative_reference(item) for item in value)
    return False


def _as_index(value: Any) -> int | None:
    try:
        index = int(value)
    except (TypeError, ValueError):
        return None
    return index if index > 0 else None


def _clean_optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _normalize_level(value: Any, validation: dict[str, int]) -> str:
    level = str(value or "").lower().strip()
    if level in _CONFIDENCE_VALUES:
        return level
    validation["invalid_schema_count"] += 1
    validation["confidence_adjusted_count"] += 1
    return "low"


def _normalize_action(
    value: Any,
    *,
    suggested_span: str | None,
    suggested_full_text: str | None,
    narrative: str,
    validation: dict[str, int],
) -> str:
    action = str(value or "").lower().strip()
    if action in _ACTION_VALUES:
        return action
    if action:
        validation["invalid_schema_count"] += 1
        return "uncertain"
    if any(marker in narrative for marker in _KEEP_MARKERS):
        return "keep"
    if any(marker in narrative for marker in _LISTEN_MARKERS):
        return "listen"
    if suggested_span or suggested_full_text:
        return "replace"
    return "uncertain"


def _is_number_sensitive(
    observed_span: Any,
    suggested_span: Any,
    problem: Any,
    source_text: str,
    *,
    window_chars: int = 5,
) -> bool:
    """Detect numeric/size edits without treating evidence indices as content."""
    observed = str(observed_span or "")
    suggested = str(suggested_span or "")
    problem_text = str(problem or "")
    if re.search(r"\d", observed) or re.search(r"\d", suggested):
        return True
    if any(marker in problem_text for marker in _NUMBER_CONTEXT_MARKERS):
        return True
    if observed and observed in source_text:
        start = source_text.find(observed)
        window = source_text[
            max(0, start - window_chars): start + len(observed) + window_chars
        ]
        return bool(re.search(r"\d", window))
    return False


def _has_rare_character(value: Any) -> bool:
    text = str(value or "")
    return any(ord(char) > 0x9FFF and char.isalpha() for char in text)


def attach_source_record_fields(
    item: dict[str, Any], source_record: dict[str, Any], episode: str,
) -> dict[str, Any]:
    """Attach immutable facts from local_review, never from model output."""
    return {
        **item,
        "episode": str(source_record.get("episode") or episode).zfill(2),
        "index": int(source_record.get("index", 0)),
        "start": source_record.get("start"),
        "end": source_record.get("end"),
        "raw_text": str(source_record.get("raw_text", "")),
        "segmentation_flags": list(source_record.get("segmentation_flags", []) or []),
        "local_review_flags": list(source_record.get("local_review_flags", []) or []),
        "do_not_auto_apply": True,
    }


def validate_diagnosis_indices(
    items: Any, records_by_index: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[Any]]:
    valid: list[dict[str, Any]] = []
    invalid: list[Any] = []
    for item in items if isinstance(items, list) else []:
        index = _as_index(item.get("index")) if isinstance(item, dict) else None
        if index is None or index not in records_by_index:
            invalid.append(item)
        else:
            valid.append(item)
    return valid, invalid


def _compact_hint(hint: dict[str, Any]) -> dict[str, Any]:
    return {
        "category": hint.get("category"),
        "observed": hint.get("span") or hint.get("observed") or hint.get("candidate"),
        "suggestion": hint.get("suggestion"),
        "confidence": hint.get("confidence"),
        "reason": hint.get("reason"),
        "source": hint.get("source"),
    }


def compact_local_review_hints(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Trim local_review records to the small subset the prompt should carry.

    Drops the large ``debug``/``flags`` blobs; keeps index/time/raw_text and the
    two flag families plus compacted hints.
    """
    compact: list[dict[str, Any]] = []
    for record in records:
        compact.append({
            "index": int(record.get("index", 0)),
            "start": record.get("start"),
            "end": record.get("end"),
            "raw_text": str(record.get("raw_text", "")),
            "segmentation_flags": list(record.get("segmentation_flags", []) or []),
            "local_review_flags": list(record.get("local_review_flags", []) or []),
            "local_review_hints": [
                _compact_hint(hint) for hint in record.get("local_review_hints", []) or []
            ],
        })
    return compact


def build_pre_review_diagnosis_prompt(
    episode: str,
    batch_index: int,
    batch_total: int,
    compact_records: list[dict[str, Any]],
    capsule: dict[str, Any],
) -> dict[str, str]:
    """Return the ``{system, user}`` prompt pair for one batch."""
    user = json.dumps(
        {
            "task": "pre_review_diagnosis",
            "instruction": (
                "粗读这一批 local_review 之后的字幕，只输出人工审阅诊断。"
                "只诊断，不改写字幕，不翻译，不生成 yue_draft，不凭空补对白。"
            ),
            "episode": episode,
            "batch_index": batch_index,
            "batch_total": batch_total,
            "knowledge_capsule": capsule,
            "records": compact_records,
            "output_schema": _OUTPUT_SCHEMA_HINT,
        },
        ensure_ascii=False,
        indent=2,
    )
    return {"system": SYSTEM_PROMPT, "user": user}


def _reject_model_item(
    invalid_model_items: list[dict[str, Any]],
    validation: dict[str, int],
    *,
    section: str,
    batch_id: int,
    reason: str,
    item: Any,
) -> None:
    invalid_model_items.append({
        "section": section,
        "batch_id": batch_id,
        "reason": reason,
        "model_item": item,
    })
    if reason == "invalid_index":
        validation["invalid_index_count"] += 1
    elif reason == "relative_reference":
        validation["relative_reference_count"] += 1
    elif reason == "common_noun":
        validation["proper_noun_filtered_count"] += 1
    else:
        validation["invalid_schema_count"] += 1


def _legacy_suggestion_fields(
    item: dict[str, Any], source_text: str,
) -> tuple[str | None, str | None, str | None]:
    observed_span = _clean_optional_text(
        item.get("observed_span") or item.get("observed") or item.get("span")
    )
    model_raw = _clean_optional_text(item.get("raw_text"))
    if observed_span is None and model_raw and model_raw != source_text and model_raw in source_text:
        observed_span = model_raw

    suggested_span = _clean_optional_text(item.get("suggested_span"))
    suggested_full_text = _clean_optional_text(item.get("suggested_full_text"))
    legacy = _clean_optional_text(item.get("suggestion"))
    if suggested_span is None and suggested_full_text is None and legacy:
        if any(marker in legacy for marker in _LISTEN_MARKERS):
            pass
        elif observed_span and len(legacy) <= max(12, len(observed_span) * 2):
            suggested_span = legacy
        elif model_raw and model_raw == source_text:
            suggested_full_text = legacy
        else:
            suggested_span = legacy
    return observed_span, suggested_span, suggested_full_text


def normalize_model_diagnosis_item(
    item: Any,
    records_by_index: dict[int, dict[str, Any]],
    episode: str,
    validation: dict[str, int],
    invalid_model_items: list[dict[str, Any]],
    *,
    batch_id: int,
) -> dict[str, Any] | None:
    """Normalize one possible_asr_errors item and attach source facts."""
    if not isinstance(item, dict):
        _reject_model_item(
            invalid_model_items, validation, section="possible_asr_errors",
            batch_id=batch_id, reason="invalid_schema", item=item,
        )
        return None
    index = _as_index(item.get("index"))
    if index is None or index not in records_by_index:
        _reject_model_item(
            invalid_model_items, validation, section="possible_asr_errors",
            batch_id=batch_id, reason="invalid_index", item=item,
        )
        return None
    if _contains_relative_reference(item):
        _reject_model_item(
            invalid_model_items, validation, section="possible_asr_errors",
            batch_id=batch_id, reason="relative_reference", item=item,
        )
        return None

    source = records_by_index[index]
    source_text = str(source.get("raw_text", ""))
    if "raw_text" in item and str(item.get("raw_text", "")) != source_text:
        validation["raw_text_repaired_count"] += 1
    observed_span, suggested_span, suggested_full_text = _legacy_suggestion_fields(
        item, source_text
    )
    problem = str(item.get("problem", "") or "").strip()
    reason = str(item.get("reason", "") or "").strip()
    narrative = " ".join((problem, reason, str(item.get("suggestion", "") or "")))
    action = _normalize_action(
        item.get("action"), suggested_span=suggested_span,
        suggested_full_text=suggested_full_text, narrative=narrative,
        validation=validation,
    )
    confidence = _normalize_level(item.get("confidence"), validation)

    if _is_number_sensitive(
        observed_span, suggested_span, problem, source_text
    ):
        if confidence != "low":
            validation["confidence_adjusted_count"] += 1
        confidence = "low"
        if action == "replace":
            action = "uncertain"
        suggested_full_text = None
    if _has_rare_character(observed_span):
        if confidence == "high":
            validation["confidence_adjusted_count"] += 1
            confidence = "medium"
        if action == "replace":
            action = "listen"
        suggested_full_text = None
    if action == "keep":
        suggested_span = None
        suggested_full_text = None
    if action == "replace" and not (observed_span and (suggested_span or suggested_full_text)):
        validation["invalid_schema_count"] += 1
        action = "uncertain"

    normalized = {
        "observed_span": observed_span,
        "action": action,
        "suggested_span": suggested_span,
        "suggested_full_text": suggested_full_text,
        "problem": problem,
        "reason": reason,
        "confidence": confidence,
        # Compatibility only; structured consumers use suggested_span/full_text.
        "suggestion": suggested_span or suggested_full_text,
    }
    return attach_source_record_fields(normalized, source, episode)


def _infer_hint_category(item: dict[str, Any], source: dict[str, Any]) -> str:
    category = str(item.get("category", "") or "").lower().strip()
    if category in _HINT_CATEGORIES:
        return category
    text = " ".join(str(item.get(key, "") or "") for key in ("hint", "reason"))
    local_flags = set(source.get("local_review_flags", []) or [])
    if "name" in text.lower() or any("name" in flag for flag in local_flags):
        return "name"
    if "term" in text.lower() or any("term" in flag or "cantonese" in flag for flag in local_flags):
        return "term"
    if source.get("segmentation_flags"):
        return "segmentation"
    return "other"


def normalize_model_line_hint(
    item: Any,
    records_by_index: dict[int, dict[str, Any]],
    episode: str,
    validation: dict[str, int],
    invalid_model_items: list[dict[str, Any]],
    *,
    batch_id: int,
) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        _reject_model_item(
            invalid_model_items, validation, section="line_hints",
            batch_id=batch_id, reason="invalid_schema", item=item,
        )
        return None
    index = _as_index(item.get("index"))
    if index is None or index not in records_by_index:
        _reject_model_item(
            invalid_model_items, validation, section="line_hints",
            batch_id=batch_id, reason="invalid_index", item=item,
        )
        return None
    if _contains_relative_reference(item):
        _reject_model_item(
            invalid_model_items, validation, section="line_hints",
            batch_id=batch_id, reason="relative_reference", item=item,
        )
        return None

    source = records_by_index[index]
    source_text = str(source.get("raw_text", ""))
    if "raw_text" in item and str(item.get("raw_text", "")) != source_text:
        validation["raw_text_repaired_count"] += 1
    observed_span, suggested_span, _ = _legacy_suggestion_fields(item, source_text)
    hint = str(item.get("hint", "") or "").strip()
    reason = str(item.get("reason", "") or "").strip()
    narrative = " ".join((hint, reason, str(item.get("suggestion", "") or "")))
    action = _normalize_action(
        item.get("action"), suggested_span=suggested_span,
        suggested_full_text=None, narrative=narrative, validation=validation,
    )
    risk_level = _normalize_level(item.get("risk_level"), validation)
    if _is_number_sensitive(
        observed_span, suggested_span, " ".join((hint, reason)), source_text
    ):
        if risk_level != "low":
            validation["confidence_adjusted_count"] += 1
        risk_level = "low"
        if action == "replace":
            action = "uncertain"
    if _has_rare_character(observed_span) and action == "replace":
        action = "listen"
    if action == "keep":
        suggested_span = None
    raw_category = str(item.get("category", "") or "").lower().strip()
    if raw_category and raw_category not in _HINT_CATEGORIES:
        validation["invalid_schema_count"] += 1
    normalized = {
        "category": _infer_hint_category(item, source),
        "observed_span": observed_span,
        "action": action,
        "suggested_span": suggested_span,
        "hint": hint,
        "reason": reason,
        "risk_level": risk_level,
    }
    return attach_source_record_fields(normalized, source, episode)


def _canonicalize_name(name: str, knowledge: dict[str, Any]) -> tuple[str, str | None]:
    characters = knowledge.get("characters", {}) if isinstance(knowledge, dict) else {}
    aliases = characters.get("alias_to_name", {}) or {}
    cleaned = name.strip()
    canonical = str(aliases.get(cleaned, "") or "").strip()
    if canonical:
        return canonical, cleaned if cleaned != canonical else None
    base = re.split(r"[（(]", cleaned, maxsplit=1)[0].strip()
    canonical = str(aliases.get(base, "") or "").strip()
    if canonical:
        return canonical, cleaned if cleaned != canonical else None
    return cleaned, None


def _evidence_indices(item: dict[str, Any], records_by_index: dict[int, dict[str, Any]]) -> list[int]:
    values: list[Any] = list(item.get("evidence_indices", []) or [])
    evidence = str(item.get("evidence", "") or "")
    values.extend(
        match.group(1)
        for match in re.finditer(r"(?:index\s*|第\s*)(\d+)(?:\s*条|\s*行)?", evidence, re.IGNORECASE)
    )
    output: list[int] = []
    for value in values:
        index = _as_index(value)
        if index in records_by_index and index not in output:
            output.append(index)
    return output


def normalize_model_proper_noun(
    item: Any,
    records_by_index: dict[int, dict[str, Any]],
    knowledge: dict[str, Any],
    validation: dict[str, int],
    invalid_model_items: list[dict[str, Any]],
    *,
    batch_id: int,
) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        _reject_model_item(
            invalid_model_items, validation, section="proper_nouns",
            batch_id=batch_id, reason="invalid_schema", item=item,
        )
        return None
    if _contains_relative_reference(item):
        _reject_model_item(
            invalid_model_items, validation, section="proper_nouns",
            batch_id=batch_id, reason="relative_reference", item=item,
        )
        return None
    original_name = str(item.get("canonical_name") or item.get("name") or "").strip()
    if not original_name:
        _reject_model_item(
            invalid_model_items, validation, section="proper_nouns",
            batch_id=batch_id, reason="invalid_schema", item=item,
        )
        return None

    canonical_name, inferred_alias = _canonicalize_name(original_name, knowledge)
    aliases_seen = [
        str(value).strip() for value in item.get("aliases_seen", []) or []
        if str(value).strip() and str(value).strip() != canonical_name
    ]
    if inferred_alias and inferred_alias not in aliases_seen:
        aliases_seen.append(inferred_alias)
    noun_type = str(item.get("type", "term") or "term").lower().strip()
    if noun_type not in _PROPER_NOUN_TYPES:
        validation["invalid_schema_count"] += 1
        noun_type = "term"
    confidence = _normalize_level(item.get("confidence"), validation)

    characters = knowledge.get("characters", {}) if isinstance(knowledge, dict) else {}
    official = set(characters.get("official_names", []) or [])
    known_aliases = set(characters.get("aliases", []) or [])
    confirmed_terms = set(knowledge.get("confirmed_terms", []) or [])
    if (
        canonical_name in _COMMON_PROPER_NOUNS
        and canonical_name not in official
        and original_name not in official
        and original_name not in known_aliases
    ):
        _reject_model_item(
            invalid_model_items, validation, section="proper_nouns",
            batch_id=batch_id, reason="common_noun", item=item,
        )
        return None
    if canonical_name in official or original_name in official or original_name in known_aliases:
        source = "official"
        status = "confirmed"
    elif canonical_name in confirmed_terms or original_name in confirmed_terms:
        source = "confirmed_glossary"
        status = "confirmed"
    else:
        source = "model_infer"
        requested_status = str(item.get("status", "") or "").lower().strip()
        status = requested_status if requested_status in {"inferred", "uncertain"} else "inferred"
        if confidence == "low":
            status = "uncertain"

    return {
        "canonical_name": canonical_name,
        "aliases_seen": list(dict.fromkeys(aliases_seen)),
        "type": noun_type,
        "source": source,
        "status": status,
        "evidence_indices": _evidence_indices(item, records_by_index),
        "evidence": str(item.get("evidence", "") or "").strip(),
        "confidence": confidence,
        "do_not_auto_apply": True,
    }


def _merge_proper_nouns(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    source_rank = {"official": 3, "confirmed_glossary": 2, "reference_profile": 1, "model_infer": 0}
    status_rank = {"confirmed": 2, "inferred": 1, "uncertain": 0}
    confidence_rank = {"high": 2, "medium": 1, "low": 0}
    for item in items:
        key = (item["canonical_name"], item["type"])
        if key not in merged:
            merged[key] = {**item}
            continue
        current = merged[key]
        current["aliases_seen"] = list(dict.fromkeys(
            [*current.get("aliases_seen", []), *item.get("aliases_seen", [])]
        ))
        current["evidence_indices"] = sorted(set(
            [*current.get("evidence_indices", []), *item.get("evidence_indices", [])]
        ))
        evidence = [value for value in (current.get("evidence"), item.get("evidence")) if value]
        current["evidence"] = "；".join(dict.fromkeys(evidence))
        if source_rank.get(item["source"], 0) > source_rank.get(current["source"], 0):
            current["source"] = item["source"]
        if status_rank.get(item["status"], 0) > status_rank.get(current["status"], 0):
            current["status"] = item["status"]
        if confidence_rank.get(item["confidence"], 0) > confidence_rank.get(current["confidence"], 0):
            current["confidence"] = item["confidence"]
    return list(merged.values())


def _normalize_batch_summary(
    summary: Any,
    *,
    batch_id: int,
    compact_records: list[dict[str, Any]],
    knowledge: dict[str, Any],
    validation: dict[str, int],
    invalid_model_items: list[dict[str, Any]],
) -> dict[str, Any]:
    value = summary if isinstance(summary, dict) else {}
    def safe_text(raw: Any, field: str) -> str:
        text = str(raw or "").strip()
        if text and _contains_relative_reference(text):
            _reject_model_item(
                invalid_model_items, validation, section=f"summary.{field}",
                batch_id=batch_id, reason="relative_reference", item=raw,
            )
            return ""
        return text

    def safe_list(raw: Any, field: str) -> list[str]:
        output: list[str] = []
        for item in raw if isinstance(raw, list) else []:
            text = safe_text(item, field)
            if text:
                output.append(text)
        return output

    characters: list[str] = []
    for name in safe_list(value.get("main_characters"), "main_characters"):
        canonical, _ = _canonicalize_name(str(name), knowledge)
        if canonical and canonical not in characters:
            characters.append(canonical)
    indices = [int(record["index"]) for record in compact_records]
    return {
        "batch_id": batch_id,
        "start_index": min(indices) if indices else None,
        "end_index": max(indices) if indices else None,
        "scene_overview": safe_text(value.get("scene_overview"), "scene_overview"),
        "main_characters": characters,
        "relationships": safe_list(value.get("relationships"), "relationships"),
        "important_events": safe_list(value.get("important_events"), "important_events"),
        "tone_style": safe_text(value.get("tone_style"), "tone_style"),
        "relationships_are_unverified": True,
        "do_not_auto_apply": True,
    }


def _summary_without_reduce(batch_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    characters: list[str] = []
    for summary in batch_summaries:
        for name in summary.get("main_characters", []):
            if name not in characters:
                characters.append(name)
    return {
        "scene_overview": "",
        "main_characters": characters,
        "relationships": [],
        "important_events": [],
        "tone_style": "",
        "aggregation": "batch_summaries_only_no_reduce",
    }


def _dedupe(items: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    out: list[dict[str, Any]] = []
    for item in items:
        signature = tuple(item.get(key) for key in keys)
        if signature in seen:
            continue
        seen.add(signature)
        out.append(item)
    return out


def _load_local_review(
    episode: str,
    paths: Any,
    config: dict[str, Any],
    local_review: dict[str, Any] | None,
    overwrite: bool,
) -> dict[str, Any]:
    if local_review is not None:
        return local_review
    source = paths.local_review_cache_dir / f"{episode}_local_review.json"
    if source.is_file():
        loaded = read_json(source)
        if isinstance(loaded, dict):
            return loaded
    # Fall back to building local_review (which reads the segments cache).
    return run_local_review(episode, paths, config, overwrite=overwrite)


def _empty_summary() -> dict[str, Any]:
    return {
        "scene_overview": "", "main_characters": [], "relationships": [],
        "important_events": [], "tone_style": "",
    }


def _model_content_preview(raw: str) -> tuple[str, bool]:
    text = str(raw or "")
    return text[:_RAW_MODEL_PREVIEW_CHARS], len(text) > _RAW_MODEL_PREVIEW_CHARS


def _request_and_parse_batch(
    client: LLMClient,
    prompt: dict[str, str],
    parse_retry_attempts: int,
) -> tuple[dict[str, Any], str, int, bool]:
    retry_instruction = (
        "\n\n上一次输出无法解析为 JSON。请只输出严格 JSON，"
        "不要解释，不要 markdown，不要代码围栏。"
    )
    raw = ""
    attempts = 0
    for attempt in range(max(0, int(parse_retry_attempts)) + 1):
        attempts += 1
        user = prompt["user"] if attempt == 0 else prompt["user"] + retry_instruction
        raw = client(prompt["system"], user)
        try:
            return parse_json_object(raw), raw, attempts, True
        except ValueError:
            continue
    return {}, raw, attempts, False


def _load_batch_result(path: Any) -> dict[str, Any] | None:
    try:
        loaded = read_json(path)
    except (OSError, ValueError):
        return None
    if not isinstance(loaded, dict) or not isinstance(loaded.get("parsed"), dict):
        return None
    return loaded


def _infer_failed_batch_ids(
    episode: str, paths: Any, batch_total: int,
) -> list[int]:
    failed: list[int] = []
    for batch_id in range(1, batch_total + 1):
        path = paths.pre_review_diagnosis_cache_dir / f"{episode}_batch_{batch_id:04d}.json"
        loaded = _load_batch_result(path)
        if loaded is None or not loaded.get("parsed"):
            failed.append(batch_id)
    return failed


def build_pre_review_diagnosis(
    episode: str,
    paths: Any,
    config: dict[str, Any],
    *,
    client: LLMClient | None = None,
    local_review: dict[str, Any] | None = None,
    overwrite: bool = False,
    rerun_failed_batches: bool = False,
) -> dict[str, Any]:
    """Run the diagnosis stage for one episode. Hint-only, never auto-applied."""
    episode = str(episode).zfill(2)
    target = paths.pre_review_diagnosis_cache_dir / f"{episode}_pre_review_diagnosis.json"
    existing_result = read_json(target) if rerun_failed_batches and target.is_file() else None
    if not rerun_failed_batches and use_cache(target, overwrite):
        loaded = read_json(target)
        return loaded if isinstance(loaded, dict) else {}

    settings = config.get("pre_review_diagnosis", {}) if isinstance(config, dict) else {}
    stage_llm = resolve_stage_llm(config, "pre_review_diagnosis")
    batch_size = int(stage_llm.get("batch_size", settings.get("batch_size", 50)))
    parse_retry_attempts = int(stage_llm.get("parse_retry_attempts", 2))
    write_batch_files = bool(settings.get("write_batch_files", True))

    review = _load_local_review(episode, paths, config, local_review, overwrite)
    records = [r for r in review.get("records", []) if isinstance(r, dict)]
    records_by_index = {
        int(record["index"]): record for record in records
        if _as_index(record.get("index")) is not None
    }
    batches = list(iter_batches(records, batch_size)) if records else []
    rerun_ids: set[int] = set()
    if rerun_failed_batches and isinstance(existing_result, dict):
        rerun_ids.update(
            index for value in existing_result.get("failed_batch_ids", []) or []
            if (index := _as_index(value)) is not None
        )
        if not rerun_ids and int(existing_result.get("stats", {}).get("parse_errors", 0)):
            rerun_ids.update(_infer_failed_batch_ids(episode, paths, len(batches)))
        if not rerun_ids:
            return existing_result

    proper_nouns: list[dict[str, Any]] = []
    asr_errors: list[dict[str, Any]] = []
    line_hints: list[dict[str, Any]] = []
    uncertain_points: list[str] = []
    batch_summaries: list[dict[str, Any]] = []
    invalid_model_items: list[dict[str, Any]] = []
    validation = _validation_state()
    failed_batch_ids: list[int] = []
    parse_retry_count = 0
    knowledge: dict[str, Any] = {}

    if records:
        if client is None:
            raise LLMNotConfiguredError(
                "pre_review_diagnosis requires an LLM client; configure the API-key "
                "environment variable or inject a mock client"
            )
        # Load shared knowledge once; capsule selection is then per-batch.
        knowledge = load_local_knowledge(paths, config)
        reference_profile = load_or_build_reference_profile(paths, config, overwrite=False)
        for batch_index, batch in enumerate(batches, start=1):
            compact_records = compact_local_review_hints(batch)
            batch_path = (
                paths.pre_review_diagnosis_cache_dir
                / f"{episode}_batch_{batch_index:04d}.json"
            )
            cached_batch = (
                _load_batch_result(batch_path)
                if rerun_failed_batches and batch_index not in rerun_ids
                else None
            )
            reused_cached_batch = cached_batch is not None and bool(cached_batch.get("parsed"))
            if reused_cached_batch:
                parsed = cached_batch["parsed"]
                raw = str(cached_batch.get("raw_model_content_preview", "") or "")
                attempt_count = int(cached_batch.get("parse_attempt_count", 0) or 0)
                parse_success = True
            else:
                capsule = build_diagnosis_knowledge_capsule(
                    batch, paths, config,
                    knowledge=knowledge, reference_profile=reference_profile,
                )
                prompt = build_pre_review_diagnosis_prompt(
                    episode, batch_index, len(batches), compact_records, capsule
                )
                parsed, raw, attempt_count, parse_success = _request_and_parse_batch(
                    client, prompt, parse_retry_attempts
                )
                parse_retry_count += max(0, attempt_count - 1)
            if not parse_success:
                failed_batch_ids.append(batch_index)
            batch_summary = _normalize_batch_summary(
                parsed.get("summary"), batch_id=batch_index,
                compact_records=compact_records, knowledge=knowledge,
                validation=validation, invalid_model_items=invalid_model_items,
            )
            batch_summaries.append(batch_summary)
            for item in parsed.get("proper_nouns", []) or []:
                normalized = normalize_model_proper_noun(
                    item, records_by_index, knowledge, validation,
                    invalid_model_items, batch_id=batch_index,
                )
                if normalized is not None:
                    proper_nouns.append(normalized)
            for item in parsed.get("possible_asr_errors", []) or []:
                normalized = normalize_model_diagnosis_item(
                    item, records_by_index, episode, validation,
                    invalid_model_items, batch_id=batch_index,
                )
                if normalized is not None:
                    asr_errors.append(normalized)
            for item in parsed.get("line_hints", []) or []:
                normalized = normalize_model_line_hint(
                    item, records_by_index, episode, validation,
                    invalid_model_items, batch_id=batch_index,
                )
                if normalized is not None:
                    line_hints.append(normalized)
            for value in parsed.get("uncertain_points", []) or []:
                text = str(value or "").strip()
                if not text:
                    continue
                if _contains_relative_reference(text):
                    _reject_model_item(
                        invalid_model_items, validation, section="uncertain_points",
                        batch_id=batch_index, reason="relative_reference", item=value,
                    )
                    continue
                uncertain_points.append(text)

            if write_batch_files and not reused_cached_batch:
                preview, truncated = _model_content_preview(raw)
                write_json(
                    batch_path,
                    {
                        "episode": episode,
                        "batch_index": batch_index,
                        "batch_total": len(batches),
                        "record_indices": [r["index"] for r in compact_records],
                        "status": "complete" if parse_success else "failed",
                        "parse_attempt_count": attempt_count,
                        "raw_model_content_preview": preview,
                        "raw_model_content_truncated": truncated,
                        "parsed": parsed,
                    },
                )

    parse_errors = len(failed_batch_ids)
    validation.update({
        "parse_errors": parse_errors,
        "failed_batch_count": parse_errors,
        "failed_batch_ids": failed_batch_ids,
    })
    result = {
        "episode": episode,
        "stage": "pre_review_diagnosis",
        "source_stage": "local_review",
        "status": "incomplete" if failed_batch_ids else "complete",
        "failed_batch_ids": failed_batch_ids,
        "do_not_auto_apply": True,
        "summary": _summary_without_reduce(batch_summaries) if batch_summaries else _empty_summary(),
        "batch_summaries": batch_summaries,
        "proper_nouns": _merge_proper_nouns(proper_nouns),
        "possible_asr_errors": _dedupe(
            asr_errors, ("index", "observed_span", "suggested_span", "action")
        ),
        "line_hints": _dedupe(
            line_hints, ("index", "category", "observed_span", "hint")
        ),
        "uncertain_points": list(dict.fromkeys(uncertain_points)),
        "invalid_model_items": invalid_model_items,
        "validation": validation,
        "stats": {
            "total_records": len(records),
            "batch_size": batch_size,
            "batches": len(batches),
            "parse_errors": parse_errors,
            "parse_retries": parse_retry_count,
            "invalid_model_items": len(invalid_model_items),
        },
    }
    write_json(target, result)
    return result


def run_pre_review_diagnosis(
    episode: str,
    paths: Any,
    config: dict[str, Any],
    *,
    client: LLMClient | None = None,
    local_review: dict[str, Any] | None = None,
    overwrite: bool = False,
    rerun_failed_batches: bool = False,
) -> dict[str, Any]:
    return build_pre_review_diagnosis(
        episode, paths, config,
        client=client, local_review=local_review, overwrite=overwrite,
        rerun_failed_batches=rerun_failed_batches,
    )
