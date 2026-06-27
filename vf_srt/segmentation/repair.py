from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..core.models import SubtitleSegment
from .candidates import char_count
from .particles import bad_island_or_segment_text


def _needs_merge(segment: SubtitleSegment, settings: dict[str, Any]) -> bool:
    duration = segment.end - segment.start
    chars = char_count(segment.raw_text)
    return (
        bad_island_or_segment_text(segment.raw_text)
        or chars < 3
        or (duration < float(settings["hard_min_duration"]) and chars <= 6)
    )


def _can_merge(left: SubtitleSegment, right: SubtitleSegment, settings: dict[str, Any]) -> bool:
    gap = right.start - left.end
    combined_chars = char_count(left.raw_text + right.raw_text)
    combined_duration = right.end - left.start
    return (
        left.source_utterance_index == right.source_utterance_index
        and gap < float(settings["speech_island_gap_seconds"])
        and combined_chars <= int(settings["hard_max_chars"])
        and combined_duration <= float(settings["hard_max_duration"])
    )


def _merge(left: SubtitleSegment, right: SubtitleSegment, direction: str) -> SubtitleSegment:
    merged = deepcopy(left)
    merged.end = right.end
    merged.raw_text = left.raw_text + right.raw_text
    merged.flags = list(dict.fromkeys(left.flags + right.flags))
    merged.debug = {
        **left.debug,
        "repair": list(left.debug.get("repair", [])) + [direction],
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
        if index > 0 and _can_merge(repaired[index - 1], repaired[index], settings):
            repaired[index - 1] = _merge(repaired[index - 1], repaired[index], "merge_short_back")
            repaired.pop(index)
            index = max(0, index - 1)
        elif index + 1 < len(repaired) and _can_merge(repaired[index], repaired[index + 1], settings):
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
