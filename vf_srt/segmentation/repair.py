from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..core.models import SubtitleSegment
from .candidates import char_count
from .particles import bad_island_or_segment_text, is_head_interjection, is_tail_particle


def _is_theme_song(segment: SubtitleSegment) -> bool:
    return "theme_song" in segment.flags or bool(segment.debug.get("theme_song"))


def _needs_merge(segment: SubtitleSegment, settings: dict[str, Any]) -> bool:
    if _is_theme_song(segment):
        return False
    duration = segment.end - segment.start
    chars = char_count(segment.raw_text)
    return (
        (
            duration < float(settings.get("merge_short_segment_max_duration", 0.8))
            and chars <= int(settings.get("merge_short_segment_max_chars", 2))
        )
        or
        bad_island_or_segment_text(segment.raw_text)
        or chars < 3
        or (duration < float(settings["hard_min_duration"]) and chars <= 6)
    )


def _can_merge(
    left: SubtitleSegment, right: SubtitleSegment, short: SubtitleSegment,
    settings: dict[str, Any],
) -> bool:
    if _is_theme_song(left) or _is_theme_song(right):
        return False
    gap = right.start - left.end
    combined_chars = char_count(left.raw_text + right.raw_text)
    combined_duration = right.end - left.start
    one_character_exception = (
        char_count(short.raw_text) == 1
        and (
            is_tail_particle(short.raw_text)
            or is_head_interjection(short.raw_text)
            or short.raw_text.strip("，。！？；：、,.!?;:…—～~ ") in {"好", "系", "咦"}
        )
    )
    return (
        gap < float(settings.get("possible_over_split_gap", 0.5))
        and (
            one_character_exception
            or (
                combined_chars <= int(settings["hard_max_chars"]) + 4
                and combined_duration <= float(settings["hard_max_duration"]) + 2.0
            )
        )
    )


def _merge(left: SubtitleSegment, right: SubtitleSegment, direction: str) -> SubtitleSegment:
    merged = deepcopy(left)
    merged.start = min(left.start, right.start)
    merged.end = max(left.end, right.end)
    merged.raw_text = left.raw_text + right.raw_text
    merged.flags = list(dict.fromkeys(left.flags + right.flags))
    merged.debug = {
        **left.debug,
        "repair": list(left.debug.get("repair", [])) + [direction],
        "merged_short_segment": True,
        "merged_short_segment_direction": direction,
        "merged_cut_debug": right.debug,
        "natural_end": right.debug.get("natural_end", right.end),
        "max_internal_gap": max(
            float(left.debug.get("max_internal_gap", 0.0)),
            max(0.0, right.start - left.end),
            float(right.debug.get("max_internal_gap", 0.0)),
        ),
    }
    return merged


def repair_segments(segments: list[SubtitleSegment], config: dict[str, Any]) -> list[SubtitleSegment]:
    settings = config["segmentation"]
    repaired = [deepcopy(segment) for segment in sorted(segments, key=lambda item: (item.start, item.end))]
    index = 0
    while index < len(repaired):
        if not _needs_merge(repaired[index], settings):
            index += 1
            continue
        current = repaired[index]
        gap_limit = float(settings.get("possible_over_split_gap", 0.5))
        gap_previous = current.start - repaired[index - 1].end if index > 0 else float("inf")
        gap_next = repaired[index + 1].start - current.end if index + 1 < len(repaired) else float("inf")
        if gap_previous > gap_limit and gap_next > gap_limit:
            index += 1
            continue

        can_merge_previous = (
            index > 0 and _can_merge(repaired[index - 1], current, current, settings)
        )
        can_merge_next = (
            index + 1 < len(repaired)
            and _can_merge(current, repaired[index + 1], current, settings)
        )
        prefer_previous = is_tail_particle(current.raw_text) and can_merge_previous
        prefer_next = (
            is_head_interjection(current.raw_text)
            and gap_next < gap_previous
            and can_merge_next
        )
        if prefer_previous or (can_merge_previous and not prefer_next and (not can_merge_next or gap_previous <= gap_next)):
            repaired[index - 1] = _merge(repaired[index - 1], repaired[index], "merge_short_back")
            repaired.pop(index)
            index = max(0, index - 1)
        elif can_merge_next:
            repaired[index] = _merge(repaired[index], repaired[index + 1], "merge_short_forward")
            repaired.pop(index + 1)
        else:
            index += 1

    extend = float(settings["extend_end_seconds"])
    minimum_gap = float(settings["min_gap_between_subtitles"])
    for index, segment in enumerate(repaired):
        natural_end = float(segment.debug.get("natural_end", segment.end))
        proposed = min(
            natural_end + extend,
            segment.start + float(settings["hard_max_duration"]),
        )
        if index + 1 < len(repaired):
            proposed = min(proposed, repaired[index + 1].start - minimum_gap)
        segment.end = max(segment.start, proposed)
        if index > 0 and segment.start < repaired[index - 1].end + minimum_gap:
            repaired[index - 1].end = max(
                repaired[index - 1].start,
                segment.start - minimum_gap,
            )
        segment.index = index + 1
    return repaired
