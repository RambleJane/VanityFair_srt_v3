from __future__ import annotations

from typing import Any

from ..core.models import SubtitleSegment
from .candidates import char_count
from .particles import HEAD_INTERJECTIONS, bad_island_or_segment_text


SHORT_REACTIONS = {"嗯", "哦", "喂", "诶", "嗱", "唉", "好", "系", "咦", "吓", "啊"}
_QUALITY_PUNCTUATION = "，。！？；：、,.!?;:…—～~（）()【】[]《》〈〉“”‘’\"' \t\r\n"
_CUT_TYPE_FLAGS = {"forced_cut", "pressure_cut", "hard_forced_cut", "bad_forced_cut"}
_PRESSURE_CUT_REASONS = {
    "over_target_chars_before", "over_soft_chars_before", "over_hard_chars_before",
    "over_target_duration_before", "over_soft_duration_before", "over_hard_duration_before",
}
_HARD_CUT_REASONS = {"over_hard_chars_before", "over_hard_duration_before"}
_NATURAL_CUT_MARKERS = (
    "strong_gap", "soft_gap", "weak_gap", "gap_strong", "gap_soft", "gap_weak",
    "strong_punctuation", "soft_punctuation", "strong_punct", "soft_punct",
    "natural_comma_weak_gap_enough_chars", "natural_tail_particle_boundary",
    "weak_head_interjection", "weak_new_turn_hint",
)


def _clean_text(text: str) -> str:
    return "".join(char for char in text.strip() if char not in _QUALITY_PUNCTUATION)


def _boundary_debug(segment: SubtitleSegment) -> dict[str, Any]:
    debug = segment.debug
    merged = debug.get("merged_cut_debug")
    while isinstance(merged, dict):
        debug = merged
        merged = debug.get("merged_cut_debug")
    return debug


def classify_cut_type(segment: SubtitleSegment, config: dict[str, Any]) -> str | None:
    if "theme_song" in segment.flags or bool(segment.debug.get("theme_song")):
        return None
    settings = config["segmentation"]
    debug = _boundary_debug(segment)
    explicit_type = debug.get("cut_type")
    if explicit_type == "natural":
        return None
    if explicit_type in {"pressure_cut", "hard_forced_cut", "bad_forced_cut"}:
        return str(explicit_type)

    reasons = [str(item) for item in debug.get("cut_reasons", [])]
    reason_text = " ".join(reasons)
    pressure_reasons = [str(item) for item in debug.get("cut_pressure_reasons", [])]
    has_pressure = bool(pressure_reasons) or any(reason in reason_text for reason in _PRESSURE_CUT_REASONS)
    chars = char_count(segment.raw_text)
    duration = max(0.0, segment.end - segment.start)
    hard_limit_reached = bool(debug.get("cut_hard_limit_reached")) or any(
        reason in reason_text for reason in _HARD_CUT_REASONS
    ) or chars >= int(settings["hard_max_chars"]) or duration >= float(settings["hard_max_duration"])
    if not has_pressure:
        has_pressure = (
            chars >= int(settings["soft_max_chars"])
            or duration >= float(settings["target_max_duration"])
        )
    natural_marker = any(marker in reason_text for marker in _NATURAL_CUT_MARKERS)
    explicit_natural = debug.get("cut_has_natural_reason")
    if isinstance(explicit_natural, bool):
        has_natural_reason: bool | None = explicit_natural
    elif natural_marker:
        score = debug.get("cut_score")
        has_natural_reason = score is None or float(score) >= float(
            settings.get("forced_cut_min_natural_score", 2.0)
        )
    else:
        has_natural_reason = None

    if hard_limit_reached:
        if has_natural_reason is True:
            return "hard_forced_cut"
        if has_natural_reason is False:
            return "bad_forced_cut"
        return "pressure_cut"
    if has_pressure:
        return "pressure_cut"
    return None


def _sync_boundary_cut_debug(segment: SubtitleSegment) -> None:
    boundary = _boundary_debug(segment)
    if boundary is segment.debug:
        return
    for key in (
        "cut_type", "cut_score", "cut_reasons", "cut_pressure_reasons",
        "cut_has_natural_reason", "cut_hard_limit_reached", "forced_cut",
    ):
        if key in boundary:
            segment.debug[key] = boundary[key]


def flag_segments(segments: list[SubtitleSegment], config: dict[str, Any]) -> list[SubtitleSegment]:
    settings = config["segmentation"]
    theme_settings = config.get("theme_song", {})
    for index, segment in enumerate(segments):
        _sync_boundary_cut_debug(segment)
        flags: list[str] = [flag for flag in dict.fromkeys(segment.flags) if flag not in _CUT_TYPE_FLAGS]
        chars = char_count(segment.raw_text)
        duration = segment.end - segment.start
        cps = chars / duration if duration > 0 else float("inf")
        cleaned = _clean_text(segment.raw_text)
        is_theme_song = "theme_song" in flags or bool(segment.debug.get("theme_song"))
        if is_theme_song and "theme_song" not in flags:
            flags.append("theme_song")
        if not segment.raw_text.strip():
            flags.append("empty_text")
        if chars > int(settings["hard_max_chars"]):
            flags.append("too_long_chars")
        if not is_theme_song and duration > float(settings["hard_max_duration"]) + 1e-9:
            flags.append("too_long_duration")
        if is_theme_song and duration > float(theme_settings.get("theme_max_duration", 12.0)):
            flags.append("theme_long_duration")
            segment.debug["theme_long_duration"] = True
        if duration < float(settings["hard_min_duration"]):
            flags.append("too_short_duration")
        if cps > float(settings["max_subtitle_cps"]):
            flags.append("too_fast_reading")
        if bad_island_or_segment_text(segment.raw_text):
            flags.append("particle_fragment")
        if float(segment.debug.get("max_internal_gap", 0.0)) >= float(settings["speech_island_gap_seconds"]):
            flags.append("cross_long_silence")
        if segment.start < 0 or segment.end <= segment.start:
            flags.append("suspicious_timing")
        if (
            chars <= int(settings.get("short_reaction_max_chars", 2))
            and duration <= float(settings.get("short_reaction_max_duration", 0.8))
            and cleaned in SHORT_REACTIONS
        ):
            flags.append("short_reaction")
        interjection_vocabulary = HEAD_INTERJECTIONS | SHORT_REACTIONS
        single_character_interjections = {char for item in interjection_vocabulary for char in item if len(item) == 1}
        if cleaned and (cleaned in interjection_vocabulary or all(char in single_character_interjections for char in cleaned)):
            flags.append("standalone_interjection")
        if not is_theme_song and (chars <= 2 or duration < 0.8):
            gap_limit = float(settings.get("possible_over_split_gap", 0.5))
            previous_gap = segment.start - segments[index - 1].end if index > 0 else float("inf")
            next_gap = segments[index + 1].start - segment.end if index + 1 < len(segments) else float("inf")
            if previous_gap < gap_limit or next_gap < gap_limit:
                flags.append("possible_over_split")
        cut_type = None if is_theme_song else classify_cut_type(segment, config)
        if cut_type is not None:
            flags.extend(["forced_cut", cut_type])
            segment.debug["cut_type"] = cut_type
        segment.flags = list(dict.fromkeys(flags))
        segment.debug.update({"chars": chars, "duration": round(duration, 3), "cps": round(cps, 3)})
    return segments
