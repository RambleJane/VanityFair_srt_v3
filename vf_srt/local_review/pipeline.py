from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

from ..core.cache import use_cache
from ..core.json_utils import read_json, write_json
from ..local_diagnosis.knowledge_loader import load_local_knowledge
from ..local_diagnosis.models import segment_to_dict
from ..local_diagnosis.name_rules import build_likely_characters
from ..local_diagnosis.reference_profile import load_or_build_reference_profile
from .flags import extract_segmentation_flags, local_review_flags_for_hints
from .terms import collect_local_review_hints


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


def _cache_dir(paths: Any) -> Path:
    return Path(getattr(paths, "local_review_cache_dir", paths.root / "cache/local_review"))


def build_local_review(
    episode: str,
    paths: Any,
    config: dict[str, Any],
    segments: list[Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build local-only review flags without changing subtitle segments."""
    episode = str(episode).zfill(2)
    target = _cache_dir(paths) / f"{episode}_local_review.json"
    if use_cache(target, overwrite):
        loaded = read_json(target)
        return loaded if isinstance(loaded, dict) else {}

    segment_path = paths.segments_cache_dir / f"{episode}_segments_raw.json"
    segments_were_loaded = segments is None
    if segments is None:
        loaded_segments = read_json(segment_path) if segment_path.is_file() else []
        segments = loaded_segments if isinstance(loaded_segments, list) else []
    segment_rows = [deepcopy(segment_to_dict(segment)) for segment in segments]

    knowledge = load_local_knowledge(paths, config)
    reference_profile = load_or_build_reference_profile(paths, config, overwrite=False)
    hints = collect_local_review_hints(
        segment_rows, knowledge, reference_profile, config
    )
    hints_by_index: dict[int, list[dict[str, Any]]] = {}
    for hint in hints:
        hints_by_index.setdefault(int(hint.get("index", 0)), []).append(hint)

    records: list[dict[str, Any]] = []
    review_flag_counts: Counter[str] = Counter()
    segmentation_flag_counts: Counter[str] = Counter()
    for segment in segment_rows:
        index = int(segment.get("index", 0))
        segment_hints = deepcopy(hints_by_index.get(index, []))
        original_flags = list(segment.get("flags", []) or [])
        segmentation_flags = extract_segmentation_flags(original_flags)
        local_review_flags = local_review_flags_for_hints(segment_hints)
        record = deepcopy(segment)
        record.update({
            "episode": str(segment.get("episode") or episode).zfill(2),
            "index": index,
            "start": segment.get("start"),
            "end": segment.get("end"),
            "raw_text": str(segment.get("raw_text", "")),
            "flags": original_flags,
            "segmentation_flags": segmentation_flags,
            "local_review_flags": local_review_flags,
            "local_review_hints": segment_hints,
        })
        records.append(record)
        segmentation_flag_counts.update(segmentation_flags)
        review_flag_counts.update(local_review_flags)

    full_profile_path = _profile_path(paths, config)
    sources = {
        "segments": _relative(segment_path, paths.root),
        **knowledge["sources"],
        "reference_profile": _relative(full_profile_path, paths.root),
    }
    missing_sources = list(knowledge["missing_sources"])
    if segments_were_loaded and not segment_path.is_file():
        missing_sources.append(sources["segments"])
    if not full_profile_path.is_file():
        missing_sources.append(sources["reference_profile"])

    name_hints = [hint for hint in hints if hint.get("category") == "character_name"]
    result = {
        "episode": episode,
        "stage": "local_review",
        "do_not_auto_apply": True,
        "sources": sources,
        "missing_sources": sorted(set(missing_sources)),
        "summary": {
            "total_segments": len(records),
            "segments_with_local_review_flags": sum(
                bool(record["local_review_flags"]) for record in records
            ),
            "total_local_review_hints": len(hints),
            "segmentation_flag_counts": dict(sorted(segmentation_flag_counts.items())),
            "local_review_flag_counts": dict(sorted(review_flag_counts.items())),
        },
        "likely_characters": build_likely_characters(name_hints, knowledge),
        "records": records,
    }
    write_json(target, result)
    return result


def run_local_review(
    episode: str,
    paths: Any,
    config: dict[str, Any],
    segments: list[Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    return build_local_review(
        episode=episode,
        paths=paths,
        config=config,
        segments=segments,
        overwrite=overwrite,
    )
