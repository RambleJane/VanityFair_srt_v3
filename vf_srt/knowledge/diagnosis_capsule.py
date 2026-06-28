"""Compact knowledge capsule for the pre-review diagnosis prompt.

The goal is to give DeepSeek *just enough* grounding to read a batch of
subtitles without dumping the whole agent directory into the prompt. Every
selector here is driven by what actually appears in the current batch of
records (raw_text + local_review_hints), then trimmed to a small subset of the
official knowledge.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.json_utils import read_json
from ..local_diagnosis.knowledge_loader import load_local_knowledge
from ..local_diagnosis.reference_profile import load_or_build_reference_profile


def _record_text_blob(records: list[dict[str, Any]]) -> str:
    """Concatenate every surface form the model will actually see."""
    parts: list[str] = []
    for record in records:
        parts.append(str(record.get("raw_text", "")))
        for hint in record.get("local_review_hints", []) or []:
            for key in ("span", "observed", "suggestion", "candidate"):
                value = hint.get(key)
                if value:
                    parts.append(str(value))
    return "\n".join(parts)


def select_relevant_characters(
    records: list[dict[str, Any]],
    knowledge: dict[str, Any],
    limit: int = 24,
) -> list[dict[str, Any]]:
    """Return official character entries whose name/alias appears in the batch."""
    blob = _record_text_blob(records)
    entries = knowledge.get("characters", {}).get("raw_entries", []) or []
    selected: list[dict[str, Any]] = []
    for item in entries:
        name = str(item.get("role_simplified") or item.get("role_raw") or "").strip()
        aliases = [
            str(alias).strip()
            for alias in item.get("aliases_simplified", []) or []
            if str(alias).strip()
        ]
        forms = [form for form in (name, *aliases) if form]
        if not forms or not any(form in blob for form in forms):
            continue
        selected.append({
            "name": name,
            "aliases": aliases,
            "actor": str(item.get("actor_simplified") or "").strip() or None,
        })
        if len(selected) >= limit:
            break
    return selected


def select_relevant_terms(
    records: list[dict[str, Any]],
    knowledge: dict[str, Any],
    limit: int = 40,
) -> dict[str, list[str]]:
    """Return confirmed/uncertain glossary terms that appear in the batch.

    Uncertain terms are returned separately so the prompt can flag that they
    must never be auto-applied.
    """
    blob = _record_text_blob(records)
    confirmed = [
        term for term in knowledge.get("confirmed_terms", []) or [] if term and term in blob
    ]
    uncertain = [
        term for term in knowledge.get("uncertain_terms", []) or [] if term and term in blob
    ]
    return {
        "confirmed": confirmed[:limit],
        "uncertain": uncertain[:limit],
    }


def _reference_profile_names(
    records: list[dict[str, Any]], reference_profile: dict[str, Any], limit: int = 24
) -> list[str]:
    blob = _record_text_blob(records)
    names = reference_profile.get("names") or {}
    if isinstance(names, dict):
        forms = list(names.keys())
    elif isinstance(names, list):
        forms = [str(item.get("form") or item) for item in names]
    else:
        forms = []
    return [form for form in forms if form and form in blob][:limit]


def _story_background(paths: Any, max_chars: int) -> str:
    source = paths.agent_dir / "05_story_outline_authoritative.md"
    if not source.is_file() or max_chars <= 0:
        return ""
    text = source.read_text(encoding="utf-8-sig")
    # Skip headers/notes, keep the first substantive prose paragraph only.
    for block in text.split("\n\n"):
        line = block.strip()
        if not line or line.startswith("#") or line.startswith("本文件") or line.startswith("若本文件"):
            continue
        return line[:max_chars]
    return text.strip()[:max_chars]


def build_diagnosis_knowledge_capsule(
    records: list[dict[str, Any]],
    paths: Any,
    config: dict[str, Any],
    knowledge: dict[str, Any] | None = None,
    reference_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a compact, batch-scoped knowledge capsule.

    ``knowledge`` / ``reference_profile`` may be passed in to avoid re-reading
    files per batch; otherwise they are loaded here.
    """
    settings = config.get("pre_review_diagnosis", {}) if isinstance(config, dict) else {}
    if knowledge is None:
        knowledge = load_local_knowledge(paths, config)
    if reference_profile is None:
        reference_profile = load_or_build_reference_profile(paths, config, overwrite=False)

    characters = select_relevant_characters(
        records, knowledge, int(settings.get("max_characters_in_capsule", 24))
    )
    terms = select_relevant_terms(
        records, knowledge, int(settings.get("max_terms_in_capsule", 40))
    )
    return {
        "story_background": _story_background(
            paths, int(settings.get("story_background_max_chars", 280))
        ),
        "relevant_characters": characters,
        "confirmed_terms": terms["confirmed"],
        "uncertain_terms": terms["uncertain"],
        "reference_profile_names": _reference_profile_names(records, reference_profile),
        "notes": "uncertain_terms 不可自动采用；以 raw_text 与当前上下文为准。",
    }
