"""Build conservative Cantonese master-draft lines for later human review.

Only ``local_review`` supplies facts. Only formal ``pre_review_diagnosis``
lists supply advice; model output is normalized before it reaches the draft.
"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from ..core.cache import use_cache
from ..core.config import resolve_stage_llm
from ..core.json_utils import read_json, write_json
from .batching import iter_batches
from .client import LLMClient, LLMNotConfiguredError
from .json_parse import parse_json_object

SYSTEM_PROMPT = (
    "你是 1977 年香港电视剧《大亨》的粤语字幕母本草稿助手。\n"
    "你只根据 raw_text、pre_review_diagnosis 正式建议和上下文，生成供人工审核的繁体粤语/港式中文底稿。\n"
    "你不是翻译助手。不得翻译成普通话，不得润色成现代标准书面语。\n"
    "只修 ASR 明显错字、人名、专名、粤语字词和明显断裂；不得凭空补对白或大幅改写句式。\n"
    "不得生成最终字幕；不确定时保留 raw_text，并标记 needs_listen 或 uncertain。\n"
    "不得消费 invalid_model_items；主题曲行保持原样。reviewer_yue_master 必须留空。\n"
    "输出严格 JSON，且只输出一个 JSON 对象。"
)

_ACTIONS = frozenset({"replace", "listen", "keep", "uncertain"})
_LEVELS = frozenset({"high", "medium", "low"})
_STATUSES = ("changed", "unchanged", "needs_listen", "uncertain")
_PREVIEW_CHARS = 3000


def yue_draft_path(paths: Any, episode: str) -> Path:
    return paths.yue_draft_cache_dir / f"{str(episode).zfill(2)}_segments_yue_draft.json"


def _as_index(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _text(value: Any) -> str:
    return str(value or "").strip()


def _action(value: Any) -> str:
    value = _text(value).lower()
    return value if value in _ACTIONS else "uncertain"


def _confidence(value: Any) -> str:
    value = _text(value).lower()
    return value if value in _LEVELS else "low"


def _compact_hint(hint: dict[str, Any], source: str) -> dict[str, Any]:
    explicit_confidence = hint.get("confidence")
    if explicit_confidence is None and hint.get("risk_level") is not None:
        # line_hints exposes review risk, not confidence: low risk means the
        # recommendation itself is comparatively safe.
        risk = _text(hint.get("risk_level")).lower()
        explicit_confidence = {"low": "high", "medium": "medium", "high": "low"}.get(risk)
    result = {
        "source": source,
        "index": _as_index(hint.get("index")),
        "observed_span": _text(hint.get("observed_span")),
        "suggested_span": _text(hint.get("suggested_span")),
        "action": _action(hint.get("action")),
        "confidence": _confidence(explicit_confidence),
    }
    note = _text(hint.get("reason", hint.get("hint", hint.get("problem"))))
    if note:
        result["note"] = note
    return result


def collect_diagnosis_hints_by_index(
    diagnosis_data: dict[str, Any],
) -> dict[int, list[dict[str, Any]]]:
    """Collect only the two formal per-line lists (never debug/invalid data)."""
    result: dict[int, list[dict[str, Any]]] = {}
    for section in ("possible_asr_errors", "line_hints"):
        values = diagnosis_data.get(section, []) if isinstance(diagnosis_data, dict) else []
        for value in values if isinstance(values, list) else []:
            if not isinstance(value, dict):
                continue
            index = _as_index(value.get("index"))
            if index is not None:
                result.setdefault(index, []).append(_compact_hint(value, section))
    return result


def _proper_noun_capsule(diagnosis: dict[str, Any], indices: set[int]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    values = diagnosis.get("proper_nouns", []) if isinstance(diagnosis, dict) else []
    for item in values if isinstance(values, list) else []:
        if not isinstance(item, dict):
            continue
        evidence = {_as_index(value) for value in item.get("evidence_indices", []) or []}
        if evidence and not (indices & evidence):
            continue
        result.append({key: item.get(key) for key in (
            "canonical_name", "aliases_seen", "type", "status", "confidence"
        ) if key in item})
    return result[:40]


def compact_yue_draft_records(
    batch: list[dict[str, Any]], all_records: list[dict[str, Any]],
    hints_by_index: dict[int, list[dict[str, Any]]], *, window_before: int = 3,
    window_after: int = 4,
) -> list[dict[str, Any]]:
    positions = {int(record["index"]): pos for pos, record in enumerate(all_records)}
    compact: list[dict[str, Any]] = []
    for record in batch:
        index = int(record["index"])
        pos = positions[index]
        before = all_records[max(0, pos - window_before):pos]
        after = all_records[pos + 1:pos + 1 + window_after]
        compact.append({
            "index": index, "start": record.get("start"), "end": record.get("end"),
            "raw_text": _text(record.get("raw_text")),
            "nearby_context_before": [{"index": r.get("index"), "raw_text": _text(r.get("raw_text"))} for r in before],
            "nearby_context_after": [{"index": r.get("index"), "raw_text": _text(r.get("raw_text"))} for r in after],
            "segmentation_flags": list(record.get("segmentation_flags", []) or []),
            "local_review_flags": list(record.get("local_review_flags", []) or []),
            "formal_diagnosis_hints": hints_by_index.get(index, []),
        })
    return compact


def build_yue_draft_prompt(
    episode: str, batch_id: int, batch_total: int, records: list[dict[str, Any]],
    proper_nouns: list[dict[str, Any]], *, batch_summaries: list[Any] | None = None,
) -> dict[str, str]:
    payload = {
        "task": "yue_draft_auto_lines", "episode": str(episode).zfill(2),
        "batch": {"id": batch_id, "total": batch_total},
        "rules": {
            "replace": "仅在上下文明确时局部替换",
            "listen": "保留 raw_text，标 needs_listen 或 uncertain",
            "keep": "保留 raw_text", "uncertain": "优先保留 raw_text",
            "low_confidence": "默认不应用", "numbers_sizes_money_rare_chars": "不得自动强改",
        },
        "proper_noun_capsule": proper_nouns,
        "batch_summaries": [{
            key: summary.get(key) for key in (
                "batch_id", "start_index", "end_index", "scene_overview",
                "main_characters", "relationships", "important_events", "tone_style",
            ) if key in summary
        } for summary in (batch_summaries or [])[:4] if isinstance(summary, dict)],
        "lines": records,
        "output_schema": {"records": [{
            "index": 1, "yue_draft": "string",
            "draft_status": "changed/unchanged/needs_listen/uncertain",
            "draft_confidence": "high/medium/low", "applied_hints": [],
            "unapplied_hints": [], "reason": "string",
        }]},
    }
    return {"system": SYSTEM_PROMPT, "user": json.dumps(payload, ensure_ascii=False)}


def _sensitive_hint(hint: dict[str, Any]) -> bool:
    value = _text(hint.get("observed_span")) + _text(hint.get("suggested_span"))
    if re.search(r"\d", value):
        return True
    return any(ord(char) > 0x9FFF and char.isalpha() for char in value)


def _theme_line(source: dict[str, Any]) -> bool:
    flags = list(source.get("segmentation_flags", []) or []) + list(source.get("local_review_flags", []) or [])
    return any("theme" in str(flag).lower() or "song" in str(flag).lower() for flag in flags)


def _public_hint(hint: dict[str, Any]) -> dict[str, Any]:
    return {key: hint.get(key) for key in (
        "source", "observed_span", "suggested_span", "action", "confidence", "note"
    ) if hint.get(key) not in (None, "")}


def _safe_model_edit(raw: str, proposed: str) -> bool:
    """Reject broad rewrites and changes to protected numeric/rare tokens."""
    if re.findall(r"\d+(?:[A-Za-z]+)?", raw) != re.findall(r"\d+(?:[A-Za-z]+)?", proposed):
        return False
    raw_rare = [char for char in raw if ord(char) > 0x9FFF and char.isalpha()]
    proposed_rare = [char for char in proposed if ord(char) > 0x9FFF and char.isalpha()]
    if raw_rare != proposed_rare:
        return False
    if abs(len(proposed) - len(raw)) > max(4, len(raw) // 4):
        return False
    threshold = 0.30 if max(len(raw), len(proposed)) <= 4 else 0.60
    return SequenceMatcher(None, raw, proposed).ratio() >= threshold


def normalize_yue_draft_item(
    item: dict[str, Any], source: dict[str, Any], formal_hints: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply the conservative action/confidence policy independently of the model."""
    raw = _text(source.get("raw_text"))
    proposed = _text(item.get("yue_draft")) or raw
    blocking = [h for h in formal_hints if h["action"] in {"listen", "keep", "uncertain"}]
    low = [h for h in formal_hints if h["confidence"] == "low"]
    sensitive = [h for h in formal_hints if _sensitive_hint(h)]
    replace = [h for h in formal_hints if h["action"] == "replace" and h["confidence"] in {"high", "medium"}]

    if _theme_line(source):
        yue, status, confidence, applied = raw, "unchanged", "high", []
    elif any(h["action"] == "listen" for h in blocking):
        yue, status, confidence, applied = raw, "needs_listen", "low", []
    elif blocking or low or sensitive:
        yue, applied = raw, []
        only_keep = blocking and all(h["action"] == "keep" for h in blocking)
        status = "unchanged" if only_keep and not low and not sensitive else "uncertain"
        confidence = "medium" if status == "unchanged" else "low"
    elif proposed != raw:
        expected = raw
        for hint in replace:
            old, new = hint["observed_span"], hint["suggested_span"]
            if old and new and old in expected:
                expected = expected.replace(old, new, 1)
        applied = [
            hint for hint in replace
            if hint["observed_span"] in raw and hint["suggested_span"] in proposed
        ]
        licensed_local_edit = not replace or proposed == expected or len(applied) == len(replace)
        if licensed_local_edit and _safe_model_edit(raw, proposed):
            yue, status = proposed, "changed"
            model_confidence = _confidence(item.get("draft_confidence", "medium"))
            confidence = (
                "high" if applied and all(h["confidence"] == "high" for h in applied)
                and model_confidence == "high" else model_confidence
            )
        else:
            yue, status, confidence, applied = raw, "uncertain", "low", []
    elif proposed == raw:
        yue, status, applied = raw, "unchanged", []
        confidence = _confidence(item.get("draft_confidence", "high"))
    applied_ids = {id(h) for h in applied}
    return {
        "yue_draft": yue, "draft_status": status, "draft_confidence": confidence,
        "applied_hints": [_public_hint(h) for h in applied],
        "unapplied_hints": [_public_hint(h) for h in formal_hints if id(h) not in applied_ids],
        "reason": _text(item.get("reason", item.get("note"))),
    }


def attach_source_fields_to_yue_draft(
    item: dict[str, Any], source: dict[str, Any], episode: str,
) -> dict[str, Any]:
    """Overwrite immutable facts with local_review values."""
    return {
        **item, "episode": str(source.get("episode") or episode).zfill(2),
        "index": int(source["index"]), "start": source.get("start"), "end": source.get("end"),
        "raw_text": _text(source.get("raw_text")),
        "segmentation_flags": list(source.get("segmentation_flags", []) or []),
        "local_review_flags": list(source.get("local_review_flags", []) or []),
        "reviewer_yue_master": "", "reviewer_note": "",
        "do_not_auto_apply_to_master": True,
    }


def validate_yue_draft_output(result: dict[str, Any]) -> dict[str, Any]:
    records = result.get("records", [])
    counts = {status: 0 for status in _STATUSES}
    for record in records:
        if record.get("draft_status") in counts:
            counts[record["draft_status"]] += 1
    result["summary"] = {"total": len(records), **counts}
    result.setdefault("validation", {})["record_count_matches_source"] = (
        len(records) == int(result.pop("source_record_count", len(records)))
    )
    return result


def _request_and_parse(
    client: LLMClient, prompt: dict[str, str], retries: int,
) -> tuple[dict[str, Any], str, int, bool]:
    raw = ""
    instruction = "\n上次输出无法解析。请只返回一个符合 schema 的严格 JSON 对象。"
    for attempt in range(max(0, retries) + 1):
        raw = client(prompt["system"], prompt["user"] + (instruction if attempt else ""))
        try:
            return parse_json_object(raw), raw, attempt + 1, True
        except ValueError:
            pass
    return {}, raw, max(0, retries) + 1, False


def _load_batch(path: Path) -> dict[str, Any] | None:
    try:
        value = read_json(path)
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) and isinstance(value.get("parsed"), dict) else None


def _load_inputs(
    episode: str, paths: Any, local_review_data: dict[str, Any] | None,
    diagnosis_data: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    local = local_review_data
    diagnosis = diagnosis_data
    if local is None:
        local = read_json(paths.local_review_cache_dir / f"{episode}_local_review.json")
    if diagnosis is None:
        diagnosis = read_json(paths.pre_review_diagnosis_cache_dir / f"{episode}_pre_review_diagnosis.json")
    if not isinstance(local, dict) or local.get("stage") != "local_review":
        raise ValueError("yue_draft_auto_lines requires formal local_review input")
    if not isinstance(diagnosis, dict) or diagnosis.get("stage") != "pre_review_diagnosis":
        raise ValueError("yue_draft_auto_lines requires formal pre_review_diagnosis input")
    return local, diagnosis


def build_yue_draft_auto_lines(
    episode: str, local_review_data: dict[str, Any] | None,
    diagnosis_data: dict[str, Any] | None, paths: Any, config: dict[str, Any],
    client: LLMClient | None = None, *, overwrite: bool = False,
    rerun_failed_batches: bool = False,
) -> dict[str, Any]:
    episode = str(episode).zfill(2)
    target = yue_draft_path(paths, episode)
    if not rerun_failed_batches and use_cache(target, overwrite):
        value = read_json(target)
        return value if isinstance(value, dict) else {}

    local, diagnosis = _load_inputs(episode, paths, local_review_data, diagnosis_data)
    records = [
        record for record in local.get("records", [])
        if isinstance(record, dict) and _as_index(record.get("index")) is not None
    ]
    by_index = {int(record["index"]): record for record in records}
    hints_by_index = collect_diagnosis_hints_by_index(diagnosis)
    stage = resolve_stage_llm(config, "yue_draft_auto_lines")
    settings = config.get("yue_draft_auto_lines", {})
    batch_size = int(stage.get("batch_size", settings.get("batch_size", 30)))
    before = int(stage.get("window_before", settings.get("window_before", 3)))
    after = int(stage.get("window_after", settings.get("window_after", 4)))
    retries = int(stage.get("parse_retry_attempts", settings.get("parse_retry_attempts", 2)))
    write_batches = bool(settings.get("write_batch_files", True))
    batches = list(iter_batches(records, batch_size)) if records else []

    existing = read_json(target) if rerun_failed_batches and target.is_file() else {}
    rerun_ids = {
        index for value in existing.get("failed_batch_ids", []) or []
        if (index := _as_index(value)) is not None
    }
    if rerun_failed_batches and target.is_file() and not rerun_ids:
        return existing
    if records and client is None:
        raise LLMNotConfiguredError(
            "yue_draft_auto_lines requires an injected or configured LLM client"
        )

    output: dict[int, dict[str, Any]] = {}
    invalid: list[dict[str, Any]] = []
    failed: list[int] = []
    retry_count = 0
    for batch_id, batch in enumerate(batches, 1):
        batch_path = paths.yue_draft_cache_dir / f"{episode}_batch_{batch_id:04d}.json"
        cached = _load_batch(batch_path) if rerun_failed_batches and batch_id not in rerun_ids else None
        compact = compact_yue_draft_records(
            batch, records, hints_by_index, window_before=before, window_after=after
        )
        if cached is not None and cached.get("status") == "complete":
            parsed = cached["parsed"]
            raw = _text(cached.get("raw_model_content_preview"))
            attempts, ok = int(cached.get("parse_attempt_count", 0)), True
        else:
            indices = {int(record["index"]) for record in batch}
            summaries = []
            for summary in diagnosis.get("batch_summaries", []) or []:
                if not isinstance(summary, dict):
                    continue
                start, end = _as_index(summary.get("start_index")), _as_index(summary.get("end_index"))
                if start is None or end is None or any(start <= index <= end for index in indices):
                    summaries.append(summary)
            prompt = build_yue_draft_prompt(
                episode, batch_id, len(batches), compact,
                _proper_noun_capsule(diagnosis, indices),
                batch_summaries=summaries,
            )
            parsed, raw, attempts, ok = _request_and_parse(client, prompt, retries)  # type: ignore[arg-type]
            retry_count += max(0, attempts - 1)
        if not ok:
            failed.append(batch_id)

        model_items = parsed.get("records", parsed.get("lines", [])) if isinstance(parsed, dict) else []
        seen: set[int] = set()
        batch_indices = {int(record["index"]) for record in batch}
        for item in model_items if isinstance(model_items, list) else []:
            index = _as_index(item.get("index")) if isinstance(item, dict) else None
            if index is None or index not in by_index or index not in batch_indices:
                invalid.append({"batch_id": batch_id, "reason": "invalid_index", "item": item})
                continue
            normalized = normalize_yue_draft_item(item, by_index[index], hints_by_index.get(index, []))
            output[index] = attach_source_fields_to_yue_draft(normalized, by_index[index], episode)
            seen.add(index)

        for source in batch:
            index = int(source["index"])
            if index in seen:
                continue
            fallback = normalize_yue_draft_item({
                "yue_draft": source.get("raw_text"), "draft_confidence": "low",
                "reason": "模型批次失败或未返回该 index，保留原文等待人工复核",
            }, source, hints_by_index.get(index, []))
            if fallback["draft_status"] == "unchanged":
                fallback["draft_status"], fallback["draft_confidence"] = "uncertain", "low"
            output[index] = attach_source_fields_to_yue_draft(fallback, source, episode)

        if write_batches and cached is None:
            write_json(batch_path, {
                "episode": episode, "batch_index": batch_id, "batch_total": len(batches),
                "record_indices": [int(record["index"]) for record in batch],
                "status": "complete" if ok else "failed", "parse_attempt_count": attempts,
                "raw_model_content_preview": raw[:_PREVIEW_CHARS],
                "raw_model_content_truncated": len(raw) > _PREVIEW_CHARS, "parsed": parsed,
            })

    result = {
        "episode": episode, "stage": "yue_draft_auto_lines",
        "source_stage": "pre_review_diagnosis",
        "status": "incomplete" if failed else "complete", "failed_batch_ids": failed,
        "do_not_auto_apply_to_master": True,
        "records": [output[int(record["index"])] for record in records],
        "invalid_model_items": invalid,
        "validation": {
            "invalid_index_count": sum(value["reason"] == "invalid_index" for value in invalid),
            "parse_errors": len(failed), "failed_batch_ids": failed,
        },
        "stats": {
            "total_records": len(records), "batch_size": batch_size, "batches": len(batches),
            "parse_errors": len(failed), "parse_retries": retry_count,
            "invalid_model_items": len(invalid),
        },
        "source_record_count": len(records),
    }
    validate_yue_draft_output(result)
    write_json(target, result)
    return result


def run_yue_draft_auto_lines(
    episode: str, paths: Any, config: dict[str, Any], *, client: LLMClient | None = None,
    local_review_data: dict[str, Any] | None = None,
    diagnosis_data: dict[str, Any] | None = None, overwrite: bool = False,
    rerun_failed_batches: bool = False,
) -> dict[str, Any]:
    return build_yue_draft_auto_lines(
        episode, local_review_data, diagnosis_data, paths, config, client,
        overwrite=overwrite, rerun_failed_batches=rerun_failed_batches,
    )
