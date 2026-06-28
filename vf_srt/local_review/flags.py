from __future__ import annotations

from typing import Any, Iterable


SEGMENTATION_FLAGS = frozenset({
    "empty_text",
    "too_long_duration",
    "too_long_chars",
    "too_short_duration",
    "too_fast_reading",
    "particle_fragment",
    "cross_long_silence",
    "suspicious_timing",
    "short_reaction",
    "standalone_interjection",
    "possible_over_split",
    "forced_cut",
    "pressure_cut",
    "hard_forced_cut",
    "bad_forced_cut",
    "theme_long_duration",
})

LOCAL_REVIEW_FLAG_BY_CATEGORY = {
    "character_name": "possible_name_mishear",
    "unknown_name_candidate": "possible_unknown_name",
    "term": "possible_term_issue",
    "food_term": "possible_term_issue",
    "mahjong_term": "possible_term_issue",
    "cantonese_word": "possible_cantonese_word",
    "period_anachronism": "possible_period_term",
    "theme_song": "possible_theme_song_issue",
    "theme_song_unmatched": "possible_theme_song_issue",
}


def extract_segmentation_flags(flags: Iterable[str] | None) -> list[str]:
    """Return quality/cut flags without changing the segment's original flags."""
    return [
        flag for flag in dict.fromkeys(flags or [])
        if flag in SEGMENTATION_FLAGS
    ]


def local_review_flag_for_hint(hint: dict[str, Any]) -> str | None:
    return LOCAL_REVIEW_FLAG_BY_CATEGORY.get(str(hint.get("category", "")))


def local_review_flags_for_hints(hints: Iterable[dict[str, Any]]) -> list[str]:
    flags = (local_review_flag_for_hint(hint) for hint in hints)
    return list(dict.fromkeys(flag for flag in flags if flag is not None))
