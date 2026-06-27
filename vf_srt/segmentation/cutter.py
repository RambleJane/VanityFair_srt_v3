from __future__ import annotations

from typing import Any

from ..core.models import CutCandidate, GapProfile, SpeechIsland, SubtitleSegment
from .candidates import char_count
from .islands import words_text
from .particles import bad_cut_before_next_word


def _window_metrics(island: SpeechIsland, start: int, end: int) -> tuple[int, float]:
    words = island.words[start:end + 1]
    return char_count(words_text(words)), max(0.0, words[-1].end - words[0].start)


def _candidate_key(candidate: CutCandidate, chars: int, duration: float, settings: dict[str, Any]) -> tuple[float, float, float]:
    target_chars = (float(settings["target_min_chars"]) + float(settings["target_max_chars"])) / 2
    target_duration = (float(settings["target_min_duration"]) + float(settings["target_max_duration"])) / 2
    adjusted = candidate.score - abs(chars - target_chars) * 0.12 - abs(duration - target_duration) * 0.15
    return adjusted, candidate.gap_after, -abs(chars - target_chars)


def cut_island_to_segments(
    island: SpeechIsland, candidates: list[CutCandidate], config: dict[str, Any],
    gap_profile: GapProfile, episode: str = "",
) -> list[SubtitleSegment]:
    settings = config["segmentation"]
    by_position = {candidate.word_pos: candidate for candidate in candidates}
    output: list[SubtitleSegment] = []
    start = 0
    while start < len(island.words):
        forced_cut = False
        if start == len(island.words) - 1:
            chosen_end = start
            chosen = None
        else:
            possible: list[tuple[CutCandidate, int, float]] = []
            forced_end: int | None = None
            for end in range(start, len(island.words) - 1):
                chars, duration = _window_metrics(island, start, end)
                candidate = by_position[end]
                if chars >= int(settings["target_min_chars"]) or duration >= float(settings["target_min_duration"]):
                    possible.append((candidate, chars, duration))
                if chars >= int(settings["hard_max_chars"]) or duration >= float(settings["hard_max_duration"]):
                    forced_end = end
                    break
            if forced_end is None:
                tail_chars, tail_duration = _window_metrics(island, start, len(island.words) - 1)
                if tail_chars <= int(settings["soft_max_chars"]) and tail_duration <= float(settings["soft_max_duration"]):
                    chosen_end, chosen = len(island.words) - 1, None
                else:
                    eligible = possible or [(by_position[end], *_window_metrics(island, start, end)) for end in range(start, len(island.words) - 1)]
                    chosen, _, _ = max(eligible, key=lambda item: _candidate_key(*item, settings))
                    chosen_end = chosen.word_pos
            else:
                forced_cut = True
                window_start = max(start, forced_end - 6)
                eligible = [
                    item for item in possible
                    if window_start <= item[0].word_pos <= forced_end
                    and item[1] <= int(settings["hard_max_chars"])
                    and item[2] <= float(settings["hard_max_duration"])
                ]
                if eligible:
                    chosen, _, _ = max(eligible, key=lambda item: _candidate_key(*item, settings))
                    chosen_end = chosen.word_pos
                else:
                    chosen_end, chosen = forced_end, by_position.get(forced_end)
                while chosen_end > start and bad_cut_before_next_word(island.words[chosen_end + 1].text):
                    chosen_end -= 1
                    chosen = by_position.get(chosen_end)
        words = island.words[start:chosen_end + 1]
        internal_gaps = [max(0.0, b.start - a.end) for a, b in zip(words, words[1:])]
        cut_reasons = chosen.reasons if chosen else ["island_end"]
        natural_cut_markers = ("strong_gap", "soft_gap", "weak_gap", "punctuation")
        forced_without_natural_boundary = forced_cut and not any(
            marker in " ".join(cut_reasons) for marker in natural_cut_markers
        )
        output.append(SubtitleSegment(
            index=0, episode=episode, source_utterance_index=island.source_utterance_index,
            start=words[0].start, end=words[-1].end, raw_text=words_text(words), flags=[],
            debug={
                "source": "speech_island", "island_reason": island.reason,
                "word_start_pos": start, "word_end_pos": chosen_end,
                "cut_score": chosen.score if chosen else None,
                "cut_reasons": cut_reasons,
                "forced_cut": forced_without_natural_boundary,
                "forced_limit_reached": forced_cut,
                "gap_after": chosen.gap_after if chosen else None,
                "max_internal_gap": max(internal_gaps, default=0.0),
                "natural_end": words[-1].end,
            },
        ))
        start = chosen_end + 1
    return output
