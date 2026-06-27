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
    report = {
        "episode": episode,
        "total_utterances": len(utterances),
        "total_words": sum(len(utterance.words) for utterance in utterances),
        "total_islands": len(islands),
        "total_segments": len(segments),
        "gap_profile": asdict(gap_profile),
        "flag_counts": dict(sorted(flag_counts.items())),
        "long_segments": [asdict(item) for item in segments if "too_long_chars" in item.flags or "too_long_duration" in item.flags][:20],
        "particle_fragment_examples": [asdict(item) for item in segments if "particle_fragment" in item.flags][:20],
        "averages": {
            "chars": round(total_chars / len(segments), 3) if segments else 0.0,
            "duration": round(total_duration / len(segments), 3) if segments else 0.0,
            "cps": round(sum(cps_values) / len(cps_values), 3) if cps_values else 0.0,
        },
    }
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
            "cut_reasons": " | ".join(segment.debug.get("cut_reasons", [])),
        })
    _write_csv(
        paths.lab_dir / f"{episode}_segments_preview.csv",
        ["index", "start", "end", "duration", "chars", "cps", "raw_text", "flags", "source_utterance_index", "cut_score", "cut_reasons"],
        preview_rows,
    )
    return report
