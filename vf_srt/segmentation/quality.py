from __future__ import annotations

from typing import Any

from ..core.models import SubtitleSegment
from .candidates import char_count
from .particles import bad_island_or_segment_text


def flag_segments(segments: list[SubtitleSegment], config: dict[str, Any]) -> list[SubtitleSegment]:
    settings = config["segmentation"]
    for segment in segments:
        flags: list[str] = []
        chars = char_count(segment.raw_text)
        duration = segment.end - segment.start
        cps = chars / duration if duration > 0 else float("inf")
        if not segment.raw_text.strip():
            flags.append("empty_text")
        if chars > int(settings["hard_max_chars"]):
            flags.append("too_long_chars")
        if duration > float(settings["hard_max_duration"]):
            flags.append("too_long_duration")
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
        segment.flags = flags
        segment.debug.update({"chars": chars, "duration": round(duration, 3), "cps": round(cps, 3)})
    return segments
