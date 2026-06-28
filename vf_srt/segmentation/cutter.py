from __future__ import annotations

from typing import Any

from ..core.models import CutCandidate, GapProfile, SpeechIsland, SubtitleSegment
from .candidates import char_count
from .islands import words_text
from .particles import bad_cut_before_next_word


_PRESSURE_REASON_NAMES = {
    "over_target_chars_before",
    "over_soft_chars_before",
    "over_hard_chars_before",
    "over_target_duration_before",
    "over_soft_duration_before",
    "over_hard_duration_before",
}
_NATURAL_REASON_MARKERS = (
    "strong_gap", "soft_gap", "weak_gap",
    "gap_strong", "gap_soft", "gap_weak",
    "strong_punctuation", "soft_punctuation",
    "strong_punct", "soft_punct",
    "natural_comma_weak_gap_enough_chars",
    "natural_tail_particle_boundary",
    "weak_head_interjection", "weak_new_turn_hint",
)


def _window_metrics(island: SpeechIsland, start: int, end: int) -> tuple[int, float]:
    words = island.words[start:end + 1]
    return char_count(words_text(words)), max(0.0, words[-1].end - words[0].start)


def _reason_name(reason: str) -> str:
    return str(reason).rsplit(" ", 1)[-1]


def _reason_points(reason: str) -> float:
    try:
        return float(str(reason).split(" ", 1)[0])
    except (TypeError, ValueError):
        return 0.0


def _candidate_cut_score(candidate: CutCandidate) -> float:
    """Return candidate score without island-relative pressure bonuses."""
    return sum(
        _reason_points(reason)
        for reason in candidate.reasons
        if _reason_name(reason) not in _PRESSURE_REASON_NAMES
    )


def _candidate_key(
    candidate: CutCandidate, chars: int, duration: float, settings: dict[str, Any],
) -> tuple[float, float, float]:
    del duration
    return (
        _candidate_cut_score(candidate),
        candidate.gap_after,
        -abs(chars - float(settings["target_max_chars"])),
    )


def _is_eligible(chars: int, duration: float, settings: dict[str, Any]) -> bool:
    return chars >= int(settings["target_min_chars"]) or duration >= float(settings["target_min_duration"])


def _in_soft_pressure(chars: int, duration: float, settings: dict[str, Any]) -> bool:
    return (
        chars >= int(settings["soft_max_chars"])
        or duration >= float(settings["soft_max_duration"])
        or (
            chars >= int(settings["target_max_chars"])
            and duration >= float(settings["target_max_duration"])
        )
    )


def _at_hard_limit(chars: int, duration: float, settings: dict[str, Any]) -> bool:
    return chars >= int(settings["hard_max_chars"]) or duration >= float(settings["hard_max_duration"])


def _pressure_reasons(chars: int, duration: float, settings: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if chars >= int(settings["hard_max_chars"]):
        reasons.append("over_hard_chars_before")
    elif chars >= int(settings["soft_max_chars"]):
        reasons.append("over_soft_chars_before")
    if duration >= float(settings["hard_max_duration"]):
        reasons.append("over_hard_duration_before")
    elif duration >= float(settings["soft_max_duration"]):
        reasons.append("over_soft_duration_before")
    elif duration >= float(settings["target_max_duration"]):
        reasons.append("over_target_duration_before")
    return reasons


def _has_natural_reason(reasons: list[str]) -> bool:
    text = " ".join(str(reason) for reason in reasons)
    return any(marker in text for marker in _NATURAL_REASON_MARKERS)


def _classify_cut(
    pressure_reasons: list[str], hard_limit_reached: bool, has_natural_reason: bool,
) -> str:
    if not pressure_reasons:
        return "natural"
    if hard_limit_reached:
        return "hard_forced_cut" if has_natural_reason else "bad_forced_cut"
    return "pressure_cut"


def cut_island_to_segments(
    island: SpeechIsland, candidates: list[CutCandidate], config: dict[str, Any],
    gap_profile: GapProfile, episode: str = "",
) -> list[SubtitleSegment]:
    del gap_profile
    settings = config["segmentation"]
    by_position = {candidate.word_pos: candidate for candidate in candidates}
    output: list[SubtitleSegment] = []
    start = 0
    last_candidate_position = len(island.words) - 2
    lookahead_words = int(settings.get("lookahead_words_after_soft_limit", 4))
    minimum_natural_score = float(settings.get("forced_cut_min_natural_score", 2.0))

    while start < len(island.words):
        chosen: CutCandidate | None = None
        chosen_end = start
        forced_limit_reached = False
        pressure_start_position: int | None = None
        decision_pressure_reasons: list[str] = []
        scan_end = start - 1

        if start < len(island.words) - 1:
            possible: list[tuple[CutCandidate, int, float]] = []
            for end in range(start, len(island.words) - 1):
                scan_end = end
                chars, duration = _window_metrics(island, start, end)
                candidate = by_position[end]
                if _is_eligible(chars, duration, settings):
                    possible.append((candidate, chars, duration))

                if pressure_start_position is None and _in_soft_pressure(chars, duration, settings):
                    pressure_start_position = end
                    decision_pressure_reasons = _pressure_reasons(chars, duration, settings)
                if _at_hard_limit(chars, duration, settings):
                    forced_limit_reached = True
                    if not decision_pressure_reasons:
                        decision_pressure_reasons = _pressure_reasons(chars, duration, settings)
                    break
                if pressure_start_position is not None and end >= pressure_start_position + lookahead_words:
                    break

            if pressure_start_position is None and not forced_limit_reached:
                tail_chars, tail_duration = _window_metrics(island, start, len(island.words) - 1)
                if not _in_soft_pressure(tail_chars, tail_duration, settings):
                    chosen_end = len(island.words) - 1
                else:
                    eligible = possible or [
                        (by_position[end], *_window_metrics(island, start, end))
                        for end in range(start, last_candidate_position + 1)
                    ]
                    chosen, _, _ = max(eligible, key=lambda item: _candidate_key(*item, settings))
                    chosen_end = chosen.word_pos
            else:
                eligible = possible or [
                    (by_position[end], *_window_metrics(island, start, end))
                    for end in range(start, scan_end + 1)
                ]
                chosen, _, _ = max(eligible, key=lambda item: _candidate_key(*item, settings))
                chosen_end = chosen.word_pos

            while (
                chosen is not None
                and chosen_end > start
                and chosen_end + 1 < len(island.words)
                and bad_cut_before_next_word(island.words[chosen_end + 1].text)
            ):
                chosen_end -= 1
                chosen = by_position.get(chosen_end)

        words = island.words[start:chosen_end + 1]
        chars, duration = _window_metrics(island, start, chosen_end)
        internal_gaps = [max(0.0, b.start - a.end) for a, b in zip(words, words[1:])]
        candidate_reasons = [] if chosen is None else [
            reason for reason in chosen.reasons
            if _reason_name(reason) not in _PRESSURE_REASON_NAMES
        ]
        cut_score = None if chosen is None else round(_candidate_cut_score(chosen), 3)
        has_natural_reason = bool(
            chosen is not None
            and _has_natural_reason(candidate_reasons)
            and cut_score is not None
            and cut_score >= minimum_natural_score
        )
        local_pressure_reasons = [] if chosen is None else _pressure_reasons(chars, duration, settings)
        pressure_reasons = local_pressure_reasons or (
            decision_pressure_reasons if chosen is not None and not has_natural_reason else []
        )
        hard_limit_reached = bool(chosen is not None and _at_hard_limit(chars, duration, settings))
        cut_type = _classify_cut(pressure_reasons, hard_limit_reached, has_natural_reason)
        cut_reasons = candidate_reasons + pressure_reasons if chosen is not None else ["island_end"]
        output.append(SubtitleSegment(
            index=0, episode=episode, source_utterance_index=island.source_utterance_index,
            start=words[0].start, end=words[-1].end, raw_text=words_text(words), flags=[],
            debug={
                "source": "speech_island", "island_reason": island.reason,
                "word_start_pos": start, "word_end_pos": chosen_end,
                "cut_score": cut_score,
                "cut_reasons": cut_reasons,
                "cut_pressure_reasons": pressure_reasons,
                "decision_pressure_reasons": decision_pressure_reasons,
                "cut_has_natural_reason": has_natural_reason,
                "cut_type": cut_type,
                "forced_cut": cut_type != "natural",
                "forced_limit_reached": forced_limit_reached,
                "cut_hard_limit_reached": hard_limit_reached,
                "soft_pressure_start_word_pos": pressure_start_position,
                "gap_after": chosen.gap_after if chosen else None,
                "max_internal_gap": max(internal_gaps, default=0.0),
                "natural_end": words[-1].end,
            },
        ))
        start = chosen_end + 1
    return output
