from __future__ import annotations

from typing import Any

from ..core.models import SubtitleSegment
from .candidates import char_count
from .particles import HEAD_INTERJECTIONS, bad_island_or_segment_text


SHORT_REACTIONS = {"嗯", "哦", "喂", "诶", "嗱", "唉", "好", "系", "咦", "吓", "啊"}
_QUALITY_PUNCTUATION = "，。！？；：、,.!?;:…—～~（）()【】[]《》〈〉“”‘’\"' \t\r\n"
_FORCED_REASONS = {
    "over_soft_chars_before", "over_hard_chars_before",
    "over_target_duration_before", "over_hard_duration_before",
}
_NATURAL_CUT_REASONS = {"strong_gap", "soft_gap", "weak_gap", "strong_punctuation", "soft_punctuation"}


def _clean_text(text: str) -> str:
    return "".join(char for char in text.strip() if char not in _QUALITY_PUNCTUATION)


def _has_forced_cut_debug(segment: SubtitleSegment) -> bool:
    if bool(segment.debug.get("forced_cut")):
        return True
    reasons = " ".join(str(item) for item in segment.debug.get("cut_reasons", []))
    has_forced_reason = any(reason in reasons for reason in _FORCED_REASONS)
    has_natural_reason = any(reason in reasons for reason in _NATURAL_CUT_REASONS)
    return has_forced_reason and not has_natural_reason


def flag_segments(segments: list[SubtitleSegment], config: dict[str, Any]) -> list[SubtitleSegment]:
    settings = config["segmentation"]
    theme_settings = config.get("theme_song", {})
    for index, segment in enumerate(segments):
        flags: list[str] = list(dict.fromkeys(segment.flags))
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
        if not is_theme_song and duration > float(settings["hard_max_duration"]):
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
        if _has_forced_cut_debug(segment):
            flags.append("forced_cut")
        segment.flags = list(dict.fromkeys(flags))
        segment.debug.update({"chars": chars, "duration": round(duration, 3), "cps": round(cps, 3)})
    return segments
