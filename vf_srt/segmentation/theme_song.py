from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from ..core.models import SubtitleSegment, Utterance, WordToken


_THEME_EQUIVALENTS = str.maketrans({
    "拚": "拼",
    "妳": "你",
    "裏": "里",
    "裡": "里",
    "嘅": "的",
})
_PUNCTUATION_RE = re.compile(r"[^\w\u3400-\u4dbf\u4e00-\u9fff]+", re.UNICODE)


def _settings(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("theme_song", config)


def load_theme_song(path: str | Path) -> dict[str, Any] | None:
    source = Path(path)
    if not source.is_file():
        return None
    try:
        with source.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("lyrics"), list):
        return None
    for lyric in data["lyrics"]:
        if not isinstance(lyric, dict):
            return None
        simplified = lyric.get("simplified")
        traditional = lyric.get("traditional")
        if not any(isinstance(value, str) and value.strip() for value in (simplified, traditional)):
            return None
    return data


def normalize_theme_text(text: str) -> str:
    normalized = str(text).lower().translate(_THEME_EQUIVALENTS)
    return _PUNCTUATION_RE.sub("", normalized)


def line_similarity(a: str, b: str) -> float:
    left = normalize_theme_text(a)
    right = normalize_theme_text(b)
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _search_line_window(
    words: list[WordToken], cursor: int, lyric_text: str,
    previous_end: float | None, max_line_gap: float,
) -> tuple[float, int, int] | None:
    lyric_length = len(normalize_theme_text(lyric_text))
    if lyric_length == 0:
        return None
    min_window = max(3, math.ceil(lyric_length * 0.5))
    max_window = max(min_window, math.floor(lyric_length * 1.8 + 6))
    best: tuple[float, int, int, float, int] | None = None
    for start_index in range(cursor, len(words)):
        if previous_end is not None and words[start_index].start - previous_end > max_line_gap:
            break
        window_text = ""
        for end_index in range(start_index, len(words)):
            window_text += words[end_index].text
            window_length = len(normalize_theme_text(window_text))
            if window_length > max_window:
                break
            if window_length < min_window:
                continue
            score = line_similarity(window_text, lyric_text)
            rank = (score, -abs(window_length - lyric_length), -start_index)
            if best is None or rank > (best[0], best[3], best[4]):
                best = (score, start_index, end_index, rank[1], rank[2])
    if best is None:
        return None
    return best[0], best[1], best[2]


def detect_theme_song_matches(
    utterances: list[Utterance], theme_song: dict[str, Any], config: dict[str, Any],
) -> list[dict[str, Any]]:
    settings = _settings(config)
    start_seconds = float(settings.get("search_start_seconds", 0.0))
    end_seconds = float(settings.get("search_end_seconds", 120.0))
    words = sorted(
        (
            word for utterance in utterances for word in utterance.words
            if word.end >= start_seconds and word.start <= end_seconds
        ),
        key=lambda word: (word.start, word.end, word.source_utterance_index, word.word_index),
    )
    lyrics = theme_song.get("lyrics")
    if not words or not isinstance(lyrics, list):
        return []

    cursor = 0
    previous_end: float | None = None
    matches: list[dict[str, Any]] = []
    max_line_gap = float(settings.get("max_gap_between_matched_lines_seconds", 12.0))
    for lyric_number, lyric in enumerate(lyrics):
        if not isinstance(lyric, dict):
            break
        lyric_text = str(lyric.get("simplified") or lyric.get("traditional") or "")
        best = _search_line_window(words, cursor, lyric_text, previous_end, max_line_gap)
        if best is None:
            break
        score, start_index, end_index = best
        threshold_key = "min_first_line_score" if lyric_number == 0 else "min_line_score"
        if score < float(settings.get(threshold_key, 0.5 if lyric_number == 0 else 0.55)):
            break
        matched_words = words[start_index:end_index + 1]
        matches.append({
            "lyric_index": lyric.get("index", lyric_number + 1),
            "simplified": str(lyric.get("simplified") or lyric.get("traditional") or ""),
            "traditional": str(lyric.get("traditional") or lyric.get("simplified") or ""),
            "start": matched_words[0].start,
            "end": matched_words[-1].end,
            "score": round(score, 6),
            "asr_text": "".join(word.text + word.trailing_punct for word in matched_words),
            "word_start_index": start_index,
            "word_end_index": end_index,
        })
        cursor = end_index + 1
        previous_end = matched_words[-1].end

    if len(matches) < int(settings.get("min_matched_lines", 2)):
        return []
    return matches


def build_theme_song_segments(
    episode: str, matches: list[dict[str, Any]], config: dict[str, Any],
) -> list[SubtitleSegment]:
    settings = _settings(config)
    extend = float(settings.get("theme_extend_end_seconds", 0.5))
    maximum_duration = float(settings.get("theme_max_duration", 12.0))
    output: list[SubtitleSegment] = []
    for index, match in enumerate(matches, start=1):
        start = float(match["start"])
        end = float(match["end"]) + extend
        debug = {
            "theme_song": True,
            "lyric_index": match["lyric_index"],
            "score": match["score"],
            "asr_text": match["asr_text"],
        }
        if end - start > maximum_duration:
            debug["theme_long_duration"] = True
        output.append(SubtitleSegment(
            index=index,
            episode=str(episode).zfill(2),
            source_utterance_index=0,
            start=start,
            end=end,
            raw_text=str(match.get("simplified") or match.get("traditional") or ""),
            flags=["theme_song", "fixed_lyric"],
            debug=debug,
        ))
    return output


def _trim_ends_to_prevent_overlap(
    segments: list[SubtitleSegment], minimum_gap: float,
) -> list[SubtitleSegment]:
    ordered = sorted(segments, key=lambda item: (item.start, item.end, item.index))
    for index, segment in enumerate(ordered[:-1]):
        following = ordered[index + 1]
        if segment.end > following.start - minimum_gap:
            segment.end = max(segment.start, following.start - minimum_gap)
    for index, segment in enumerate(ordered, start=1):
        segment.index = index
    return ordered


def apply_theme_song_override(
    episode: str, segments: list[SubtitleSegment], utterances: list[Utterance],
    paths: Any, config: dict[str, Any],
) -> list[SubtitleSegment]:
    settings = _settings(config)
    if not bool(settings.get("enabled", False)) or not bool(settings.get("apply_fixed_lyrics", True)):
        return segments
    configured_path = Path(str(settings.get("json_path", "agent/theme_song.json")))
    source = configured_path if configured_path.is_absolute() else Path(paths.root) / configured_path
    theme_song = load_theme_song(source)
    if theme_song is None:
        return segments
    matches = detect_theme_song_matches(utterances, theme_song, settings)
    if not matches:
        return segments
    theme_segments = build_theme_song_segments(episode, matches, settings)
    theme_start = theme_segments[0].start
    theme_end = theme_segments[-1].end
    remaining: list[SubtitleSegment] = []
    for original in segments:
        segment = deepcopy(original)
        duration = max(0.0, segment.end - segment.start)
        overlap = max(0.0, min(segment.end, theme_end) - max(segment.start, theme_start))
        if duration > 0 and overlap > duration * 0.5:
            continue
        remaining.append(segment)
    minimum_gap = float(config.get("segmentation", {}).get("min_gap_between_subtitles", 0.08))
    return _trim_ends_to_prevent_overlap(theme_segments + remaining, minimum_gap)
