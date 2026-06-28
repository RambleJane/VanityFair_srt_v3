"""pre_review_diagnosis: LLM-assisted, hint-only review of local_review output.

Reads ``cache/local_review/{episode}_local_review.json`` and asks an injected
LLM client to produce human-review notes *only*. It never rewrites subtitles,
never translates, never drafts Yue lines, and never auto-applies anything.

No network client is shipped in this milestone: the stage takes an
``LLMClient`` callable so tests run with a mock and real DeepSeek wiring can be
added later without changing this logic.
"""
from __future__ import annotations

import json
from typing import Any

from ..core.cache import use_cache
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
        {"name": "string", "type": "person/place/company/term",
         "evidence": "string", "confidence": "high/medium/low"}
    ],
    "possible_asr_errors": [
        {"index": 0, "raw_text": "string", "problem": "string",
         "suggestion": "string", "reason": "string", "confidence": "high/medium/low"}
    ],
    "line_hints": [
        {"index": 0, "hint": "string", "risk_level": "high/medium/low"}
    ],
    "uncertain_points": ["string"],
}

_SUMMARY_LIST_KEYS = ("main_characters", "relationships", "important_events")
_SUMMARY_TEXT_KEYS = ("scene_overview", "tone_style")


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


def _stamp(items: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict):
            out.append({**item, "do_not_auto_apply": True})
    return out


def _merge_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {key: [] for key in _SUMMARY_LIST_KEYS}
    texts: dict[str, list[str]] = {key: [] for key in _SUMMARY_TEXT_KEYS}
    for summary in summaries:
        if not isinstance(summary, dict):
            continue
        for key in _SUMMARY_LIST_KEYS:
            for value in summary.get(key, []) or []:
                if value not in merged[key]:
                    merged[key].append(value)
        for key in _SUMMARY_TEXT_KEYS:
            value = str(summary.get(key, "") or "").strip()
            if value and value not in texts[key]:
                texts[key].append(value)
    for key in _SUMMARY_TEXT_KEYS:
        merged[key] = " ".join(texts[key])
    return merged


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


def build_pre_review_diagnosis(
    episode: str,
    paths: Any,
    config: dict[str, Any],
    *,
    client: LLMClient | None = None,
    local_review: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run the diagnosis stage for one episode. Hint-only, never auto-applied."""
    episode = str(episode).zfill(2)
    target = paths.pre_review_diagnosis_cache_dir / f"{episode}_pre_review_diagnosis.json"
    if use_cache(target, overwrite):
        loaded = read_json(target)
        return loaded if isinstance(loaded, dict) else {}

    settings = config.get("pre_review_diagnosis", {}) if isinstance(config, dict) else {}
    batch_size = int(settings.get("batch_size", 50))
    write_batch_files = bool(settings.get("write_batch_files", True))

    review = _load_local_review(episode, paths, config, local_review, overwrite)
    records = [r for r in review.get("records", []) if isinstance(r, dict)]
    batches = list(iter_batches(records, batch_size)) if records else []

    proper_nouns: list[dict[str, Any]] = []
    asr_errors: list[dict[str, Any]] = []
    line_hints: list[dict[str, Any]] = []
    uncertain_points: list[Any] = []
    summaries: list[dict[str, Any]] = []
    parse_errors = 0

    if records:
        if client is None:
            raise LLMNotConfiguredError(
                "pre_review_diagnosis requires an LLM client; none configured "
                "(this milestone ships no network client — inject one or use a mock)"
            )
        # Load shared knowledge once; capsule selection is then per-batch.
        knowledge = load_local_knowledge(paths, config)
        reference_profile = load_or_build_reference_profile(paths, config, overwrite=False)
        for batch_index, batch in enumerate(batches, start=1):
            compact_records = compact_local_review_hints(batch)
            capsule = build_diagnosis_knowledge_capsule(
                batch, paths, config,
                knowledge=knowledge, reference_profile=reference_profile,
            )
            prompt = build_pre_review_diagnosis_prompt(
                episode, batch_index, len(batches), compact_records, capsule
            )
            raw = client(prompt["system"], prompt["user"])
            try:
                parsed = parse_json_object(raw)
            except ValueError:
                parse_errors += 1
                parsed = {}
            batch_summary = parsed.get("summary") if isinstance(parsed.get("summary"), dict) else {}
            summaries.append(batch_summary)
            proper_nouns.extend(_stamp(parsed.get("proper_nouns")))
            asr_errors.extend(_stamp(parsed.get("possible_asr_errors")))
            line_hints.extend(_stamp(parsed.get("line_hints")))
            uncertain_points.extend(v for v in (parsed.get("uncertain_points") or []) if v)

            if write_batch_files:
                write_json(
                    paths.pre_review_diagnosis_cache_dir
                    / f"{episode}_batch_{batch_index:04d}.json",
                    {
                        "episode": episode,
                        "batch_index": batch_index,
                        "batch_total": len(batches),
                        "record_indices": [r["index"] for r in compact_records],
                        "parsed": parsed,
                    },
                )

    result = {
        "episode": episode,
        "stage": "pre_review_diagnosis",
        "source_stage": "local_review",
        "do_not_auto_apply": True,
        "summary": _merge_summaries(summaries) if summaries else _empty_summary(),
        "proper_nouns": _dedupe(proper_nouns, ("name", "type")),
        "possible_asr_errors": _dedupe(asr_errors, ("index", "suggestion")),
        "line_hints": _dedupe(line_hints, ("index", "hint")),
        "uncertain_points": list(dict.fromkeys(uncertain_points)),
        "stats": {
            "total_records": len(records),
            "batch_size": batch_size,
            "batches": len(batches),
            "parse_errors": parse_errors,
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
) -> dict[str, Any]:
    return build_pre_review_diagnosis(
        episode, paths, config,
        client=client, local_review=local_review, overwrite=overwrite,
    )
