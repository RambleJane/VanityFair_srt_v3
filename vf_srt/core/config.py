from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "project": {"run_until": "segmented"},
    "cache": {"overwrite_existing": False},
    "reference_profile": {
        "enabled": True,
        "json_path": "reference/profile/reference_srt_profile.json",
        "markdown_path": "reference/profile/reference_srt_profile.md",
        "source_srt_dir": "reference/simplified_human",
        "source_episodes": ["01", "02", "03", "04", "05", "06", "07", "08"],
        "rebuild_from_srt_if_missing": False,
    },
    "segmentation": {
        "target_min_chars": 10, "target_max_chars": 18,
        "soft_max_chars": 20, "hard_max_chars": 24,
        "target_min_duration": 1.2, "target_max_duration": 5.0,
        "soft_max_duration": 5.8, "hard_max_duration": 6.8,
        "hard_min_duration": 0.8, "weak_gap_default": 0.35,
        "soft_gap_floor": 0.50, "strong_gap_floor": 0.90,
        "speech_island_gap_seconds": 1.20, "max_subtitle_cps": 11.0,
        "comfortable_cps": 8.0, "extend_end_seconds": 0.35,
        "min_gap_between_subtitles": 0.08,
        "short_reaction_max_chars": 2,
        "short_reaction_max_duration": 0.8,
        "possible_over_split_gap": 0.50,
        "merge_short_segment_max_chars": 2,
        "merge_short_segment_max_duration": 0.8,
        "lookahead_words_after_soft_limit": 4,
        "forced_cut_min_natural_score": 2.0,
    },
    "pre_review_diagnosis": {
        "batch_size": 50,
        "write_batch_files": True,
        "story_background_max_chars": 280,
        "max_characters_in_capsule": 24,
        "max_terms_in_capsule": 40,
    },
    "theme_song": {
        "enabled": True,
        "json_path": "agent/theme_song.json",
        "opening": {
            "enabled": True,
            "search_start_seconds": 0.0,
            "search_end_seconds": 120.0,
            "min_line_score": 0.55,
            "min_first_line_score": 0.50,
            "min_matched_lines": 2,
        },
        "ending": {
            "enabled": True,
            "search_last_seconds": 180.0,
            "min_line_score": 0.55,
            "min_first_line_score": 0.50,
            "min_matched_lines": 3,
            "tail_min_line_score": 0.58,
            "tail_allow_repeated_lyrics": True,
            "tail_cover_to_end": True,
        },
        "max_gap_between_matched_lines_seconds": 12.0,
        "apply_fixed_lyrics": True,
        "theme_extend_end_seconds": 0.50,
        "theme_max_duration": 12.0,
    },
}


def _merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def _scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return {}
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none"}:
        return None
    if (value[:1], value[-1:]) in {(('"'), ('"')), (("'"), ("'"))}:
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return parsed
    try:
        return float(value) if "." in value else int(value)
    except ValueError:
        return value


def _load_simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for number, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        if ":" not in line:
            raise ValueError(f"Unsupported YAML at line {number}: {line}")
        key, value = line.split(":", 1)
        while stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        parsed = _scalar(value.split(" #", 1)[0])
        parent[key.strip()] = parsed
        if isinstance(parsed, dict):
            stack.append((indent, parsed))
    return root


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if not config_path:
        return config
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    updates = _load_simple_yaml(path.read_text(encoding="utf-8-sig"))
    theme_updates = updates.get("theme_song")
    if isinstance(theme_updates, dict) and "opening" not in theme_updates:
        legacy_keys = (
            "search_start_seconds", "search_end_seconds", "min_line_score",
            "min_first_line_score", "min_matched_lines",
        )
        legacy_opening = {
            key: theme_updates.pop(key) for key in legacy_keys if key in theme_updates
        }
        if legacy_opening:
            theme_updates["opening"] = legacy_opening
    return _merge(config, updates)


def parse_episodes(value: str) -> list[str]:
    episodes: list[str] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
        if match:
            first, last = map(int, match.groups())
            if last < first:
                raise ValueError(f"Episode range runs backwards: {part}")
            width = max(len(match.group(1)), len(match.group(2)), 2)
            episodes.extend(f"{item:0{width}d}" for item in range(first, last + 1))
        elif part.isdigit():
            episodes.append(part.zfill(max(2, len(part))))
        else:
            raise ValueError(f"Invalid episode expression: {part!r}")
    return list(dict.fromkeys(episodes))
