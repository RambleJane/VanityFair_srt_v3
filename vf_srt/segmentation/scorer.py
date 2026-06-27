from __future__ import annotations

from typing import Any

from ..core.models import CutCandidate, GapProfile
from .particles import bad_cut_before_next_word, is_head_interjection, is_particle_fragment, is_tail_particle
from .punctuation import is_soft_punct, is_strong_punct


def _add(candidate: CutCandidate, points: float, reason: str) -> None:
    candidate.score += points
    candidate.reasons.append(f"{points:+g} {reason}")


def score_candidate(
    candidate: CutCandidate, config: dict[str, Any], gap_profile: GapProfile,
) -> CutCandidate:
    settings = config["segmentation"]
    candidate.score = 0.0
    candidate.reasons = []
    if candidate.gap_after >= gap_profile.strong_gap:
        _add(candidate, 4, "strong_gap")
    elif candidate.gap_after >= gap_profile.soft_gap:
        _add(candidate, 2, "soft_gap")
    elif candidate.gap_after >= gap_profile.weak_gap:
        _add(candidate, 1, "weak_gap")
    if is_strong_punct(candidate.trailing_punct):
        _add(candidate, 2, "strong_punctuation")
    elif is_soft_punct(candidate.trailing_punct):
        _add(candidate, 1, "soft_punctuation")
    if 10 <= candidate.chars_before <= 18:
        _add(candidate, 1, "comfortable_chars_before")
    if candidate.chars_before > 22:
        _add(candidate, 4, "over_hard_chars_before")
    elif candidate.chars_before > 18:
        _add(candidate, 2, "over_soft_chars_before")
    if candidate.duration_before > float(settings["hard_max_duration"]):
        _add(candidate, 4, "over_hard_duration_before")
    elif candidate.duration_before > float(settings["target_max_duration"]):
        _add(candidate, 2, "over_target_duration_before")
    if bad_cut_before_next_word(candidate.next_text):
        _add(candidate, -5, "cut_before_tail_particle")
    if candidate.chars_before < 3:
        _add(candidate, -4, "short_before")
    if candidate.chars_after < 3:
        _add(candidate, -4, "short_after")
    if is_particle_fragment(candidate.next_text) or (candidate.chars_after <= 2 and is_tail_particle(candidate.next_text)):
        _add(candidate, -8, "isolated_particle")
    if is_head_interjection(candidate.next_text) or candidate.next_text.startswith("阿"):
        _add(candidate, 0.5, "weak_new_turn_hint")
    return candidate
