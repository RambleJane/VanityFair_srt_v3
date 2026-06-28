from __future__ import annotations

import csv
import os
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from ..core.json_utils import write_json
from ..core.models import CutCandidate, GapProfile, SpeechIsland, SubtitleSegment, Utterance
from .candidates import char_count


_DETAIL_FLAGS = (
    "short_reaction", "standalone_interjection", "possible_over_split",
    "forced_cut", "pressure_cut", "hard_forced_cut", "bad_forced_cut",
    "theme_song", "fixed_lyric", "theme_opening", "theme_ending",
    "theme_ending_unmatched",
)


def _segment_example(segment: SubtitleSegment) -> dict[str, Any]:
    return {
        "index": segment.index,
        "start": segment.start,
        "end": segment.end,
        "duration": round(max(0.0, segment.end - segment.start), 3),
        "raw_text": segment.raw_text,
        "flags": segment.flags,
        "debug": segment.debug,
    }


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_segmentation_outputs(
    episode: str, utterances: list[Utterance], islands: list[SpeechIsland],
    candidates: list[CutCandidate], segments: list[SubtitleSegment],
    gap_profile: GapProfile, paths: Any,
) -> dict[str, Any]:
    flag_counts = Counter(flag for segment in segments for flag in segment.flags)
    total_chars = sum(char_count(segment.raw_text) for segment in segments)
    total_duration = sum(max(0.0, segment.end - segment.start) for segment in segments)
    cps_values = [segment.debug.get("cps", 0.0) for segment in segments]
    opening_theme = [segment for segment in segments if "theme_opening" in segment.flags]
    ending_theme = [
        segment for segment in segments
        if "theme_ending" in segment.flags and "fixed_lyric" in segment.flags
    ]
    ending_unmatched = [segment for segment in segments if "theme_ending_unmatched" in segment.flags]
    opening_scores = [float(segment.debug["score"]) for segment in opening_theme if segment.debug.get("score") is not None]
    ending_scores = [float(segment.debug["score"]) for segment in ending_theme if segment.debug.get("score") is not None]
    ending_region_segments = ending_theme + ending_unmatched
    ending_starts = [float(segment.debug["ending_theme_start"]) for segment in ending_region_segments if segment.debug.get("ending_theme_start") is not None]
    ending_ends = [float(segment.debug["ending_theme_end"]) for segment in ending_region_segments if segment.debug.get("ending_theme_end") is not None]
    report = {
        "episode": episode,
        "total_utterances": len(utterances),
        "total_words": sum(len(utterance.words) for utterance in utterances),
        "total_islands": len(islands),
        "total_segments": len(segments),
        "gap_profile": asdict(gap_profile),
        "flag_counts": dict(sorted(flag_counts.items())),
        "forced_cut_summary": {
            "forced_cut": flag_counts.get("forced_cut", 0),
            "pressure_cut": flag_counts.get("pressure_cut", 0),
            "hard_forced_cut": flag_counts.get("hard_forced_cut", 0),
            "bad_forced_cut": flag_counts.get("bad_forced_cut", 0),
        },
        "theme_song_summary": {
            "opening_matched_lines": len(opening_theme),
            "ending_matched_lines": len(ending_theme),
            "ending_unmatched_segments": len(ending_unmatched),
            "ending_theme_start": min(ending_starts) if ending_starts else None,
            "ending_theme_end": max(ending_ends) if ending_ends else None,
            "opening_score_avg": round(sum(opening_scores) / len(opening_scores), 6) if opening_scores else None,
            "ending_score_avg": round(sum(ending_scores) / len(ending_scores), 6) if ending_scores else None,
        },
        "long_segments": [asdict(item) for item in segments if "too_long_chars" in item.flags or "too_long_duration" in item.flags][:20],
        "particle_fragment_examples": [asdict(item) for item in segments if "particle_fragment" in item.flags][:20],
        "averages": {
            "chars": round(total_chars / len(segments), 3) if segments else 0.0,
            "duration": round(total_duration / len(segments), 3) if segments else 0.0,
            "cps": round(sum(cps_values) / len(cps_values), 3) if cps_values else 0.0,
        },
    }
    for flag in _DETAIL_FLAGS:
        report[f"{flag}_count"] = flag_counts.get(flag, 0)
        report[f"{flag}_examples"] = [
            _segment_example(item) for item in segments if flag in item.flags
        ][:20]
    write_json(paths.reports_cache_dir / f"{episode}_segmentation_report.json", report)
    candidate_rows = []
    for candidate in candidates:
        row = asdict(candidate)
        row["reasons"] = " | ".join(candidate.reasons)
        candidate_rows.append(row)
    _write_csv(
        paths.lab_dir / f"{episode}_cut_candidates.csv",
        ["island_index", "word_pos", "time", "gap_after", "prev_text", "next_text", "trailing_punct", "chars_before", "chars_after", "duration_before", "duration_after", "score", "reasons"],
        candidate_rows,
    )
    preview_rows = []
    for segment in segments:
        preview_rows.append({
            "index": segment.index, "start": segment.start, "end": segment.end,
            "duration": segment.debug.get("duration"), "chars": segment.debug.get("chars"),
            "cps": segment.debug.get("cps"), "raw_text": segment.raw_text,
            "flags": " | ".join(segment.flags),
            "source_utterance_index": segment.source_utterance_index,
            "cut_score": segment.debug.get("cut_score"),
            "cut_type": segment.debug.get("cut_type"),
            "cut_reasons": " | ".join(segment.debug.get("cut_reasons", [])),
            "cut_pressure_reasons": " | ".join(segment.debug.get("cut_pressure_reasons", [])),
            "theme_song": segment.debug.get("theme_song", False),
            "theme_region": segment.debug.get("theme_region"),
            "lyric_index": segment.debug.get("lyric_index"),
            "theme_score": segment.debug.get("score"),
            "theme_unmatched": segment.debug.get("theme_unmatched", False),
            "theme_asr_text": segment.debug.get("asr_text"),
            "asr_text": segment.debug.get("asr_text"),
            "forced_cut": "forced_cut" in segment.flags,
        })
    _write_csv(
        paths.lab_dir / f"{episode}_segments_preview.csv",
        ["index", "start", "end", "duration", "chars", "cps", "raw_text", "flags", "source_utterance_index", "cut_type", "cut_score", "cut_reasons", "cut_pressure_reasons", "theme_song", "theme_region", "lyric_index", "theme_score", "theme_unmatched", "asr_text", "theme_asr_text", "forced_cut"],
        preview_rows,
    )
    return report
