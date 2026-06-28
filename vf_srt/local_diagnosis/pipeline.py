from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from ..core.cache import use_cache
from ..core.json_utils import read_json, write_json
from ..knowledge.reference_profile import load_reference_profile
from .knowledge_loader import load_local_knowledge
from .models import segment_to_dict
from .name_rules import build_likely_characters, diagnose_character_names
from .reference_profile import load_or_build_reference_profile
from .report import build_segment_hints, write_local_diagnosis
from .term_rules import diagnose_terms
from .theme_rules import diagnose_theme_song
from .unknown_names import detect_unknown_name_candidates


_ALIANG_MISHEARS = ("阿梁", "阿亮", "阿凉", "阿楠")


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _profile_path(paths: Any, config: dict[str, Any]) -> Path:
    value = Path(config.get("reference_profile", {}).get(
        "json_path", "reference/profile/reference_srt_profile.json"
    ))
    return value if value.is_absolute() else paths.root / value


def _aliang_profile_count(profile: dict[str, Any]) -> int:
    for item in profile.get("high_frequency_names", []):
        if item.get("name") != "徐绍良":
            continue
        for form in item.get("matched_forms", []):
            if form.get("form") == "阿良":
                return int(form.get("count", 0))
    for item in profile.get("address_terms", []):
        if item.get("term") == "阿良":
            return int(item.get("count", 0))
    return 0


def _official_alias_has_aliang(characters: dict[str, Any]) -> bool:
    for item in characters.get("characters", []):
        if item.get("role_simplified") != "徐绍良":
            continue
        aliases = item.get("aliases_simplified", []) or []
        return "阿良" in aliases
    return False


def _segment_dict(segment: Any) -> dict[str, Any]:
    if is_dataclass(segment):
        return asdict(segment)
    return dict(segment)


def _legacy_run_local_pre_review_diagnosis(
    episode: str, paths: Any, config: dict[str, Any], segments: list[Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    episode = str(episode).zfill(2)
    target = paths.local_diagnosis_cache_dir / f"{episode}_local_pre_review_diagnosis.json"
    if use_cache(target, overwrite):
        loaded = read_json(target)
        return loaded if isinstance(loaded, dict) else {}

    profile_path = _profile_path(paths, config)
    characters_path = paths.agent_dir / "characters_official.json"
    sources = {
        "reference_profile": _relative(profile_path, paths.root),
        "characters_official": _relative(characters_path, paths.root),
    }
    missing_sources = [
        value for value, path in (
            (sources["reference_profile"], profile_path),
            (sources["characters_official"], characters_path),
        )
        if not path.is_file()
    ]
    profile = load_reference_profile(paths, config)
    characters = read_json(characters_path) if characters_path.is_file() else {"characters": []}
    if not isinstance(characters, dict):
        characters = {"characters": []}
    aliang_count = _aliang_profile_count(profile)
    official_alias = _official_alias_has_aliang(characters)

    if segments is None:
        segment_path = paths.segments_cache_dir / f"{episode}_segments_raw.json"
        segments = read_json(segment_path) if segment_path.is_file() else []
    name_diagnosis: list[dict[str, Any]] = []
    for segment_value in segments:
        segment = _segment_dict(segment_value)
        raw_text = str(segment.get("raw_text", ""))
        for variant in _ALIANG_MISHEARS:
            if variant not in raw_text:
                continue
            evidence: list[str] = []
            if aliang_count:
                evidence.append(f"前8集人工精校字幕画像中“阿良”出现 {aliang_count} 次")
            if official_alias:
                evidence.append("官方角色徐绍良别名包括“阿良”")
            evidence.append(f"“{variant}”是近音误听候选，需结合当前上下文人工确认")
            name_diagnosis.append({
                "index": int(segment.get("index", 0)),
                "start": segment.get("start"),
                "end": segment.get("end"),
                "raw_text": raw_text,
                "observed": variant,
                "suggestion": "阿良",
                "reason": "；".join(evidence) + "。",
                "action": "hint_only_no_asr_change",
            })

    result = {
        "episode": episode,
        "sources": sources,
        "missing_sources": missing_sources,
        "name_diagnosis": name_diagnosis,
        "segment_issues": name_diagnosis,
        "summary": {
            "name_hints": len(name_diagnosis),
            "reference_profile_loaded": bool(profile),
        },
    }
    write_json(target, result)
    return result


def build_local_pre_review_diagnosis(
    episode: str,
    paths: Any,
    config: dict[str, Any],
    segments: list[Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build a hint-only local diagnosis without changing segmentation or ASR text."""
    episode = str(episode).zfill(2)
    target = paths.local_diagnosis_cache_dir / f"{episode}_local_pre_review_diagnosis.json"
    if use_cache(target, overwrite):
        loaded = read_json(target)
        return loaded if isinstance(loaded, dict) else {}

    segment_path = paths.segments_cache_dir / f"{episode}_segments_raw.json"
    if segments is None:
        loaded_segments = read_json(segment_path) if segment_path.is_file() else []
        segments = loaded_segments if isinstance(loaded_segments, list) else []
    segment_rows = [segment_to_dict(segment) for segment in segments]

    knowledge = load_local_knowledge(paths, config)
    reference_profile = load_or_build_reference_profile(paths, config, overwrite=False)
    full_profile_path = _profile_path(paths, config)

    sources = {
        "segments": _relative(segment_path, paths.root),
        **knowledge["sources"],
        "reference_profile": _relative(full_profile_path, paths.root),
    }
    missing_sources = list(knowledge["missing_sources"])
    if not segment_path.is_file():
        missing_sources.append(sources["segments"])
    if not full_profile_path.is_file():
        missing_sources.append(sources["reference_profile"])
    missing_sources = sorted(set(missing_sources))

    character_hints = diagnose_character_names(
        segment_rows, knowledge, reference_profile, config
    )
    term_hints = diagnose_terms(segment_rows, knowledge, reference_profile, config)
    theme_hints = diagnose_theme_song(segment_rows, knowledge, config)
    unknown_name_candidates = detect_unknown_name_candidates(
        segment_rows, knowledge, reference_profile, config
    )
    possible_asr_errors = character_hints + term_hints + theme_hints
    all_hints = possible_asr_errors + unknown_name_candidates
    likely_characters = build_likely_characters(character_hints, knowledge)

    result = {
        "episode": episode,
        "do_not_auto_apply": True,
        "stage": "local_pre_review_diagnosis",
        "sources": sources,
        "missing_sources": missing_sources,
        "summary": {
            "total_segments": len(segment_rows),
            "possible_asr_errors": len(possible_asr_errors),
            "character_name_hints": len(character_hints),
            "term_hints": len(term_hints),
            "theme_song_hints": len(theme_hints),
            "unknown_name_candidates": len(unknown_name_candidates),
            "reference_profile_loaded": bool(
                reference_profile.get("source_episodes")
                or reference_profile.get("names")
            ),
        },
        "likely_characters": likely_characters,
        "possible_asr_errors": possible_asr_errors,
        "unknown_name_candidates": unknown_name_candidates,
        "segment_hints": build_segment_hints(all_hints),
        # Compatibility with the first local-diagnosis prototype.
        "name_diagnosis": character_hints,
        "segment_issues": possible_asr_errors,
    }
    write_local_diagnosis(result, target)
    return result


def run_local_pre_review_diagnosis(
    episode: str,
    paths: Any,
    config: dict[str, Any],
    segments: list[Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    return build_local_pre_review_diagnosis(
        episode=episode,
        paths=paths,
        config=config,
        segments=segments,
        overwrite=overwrite,
    )
