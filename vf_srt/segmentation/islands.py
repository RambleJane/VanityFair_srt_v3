from __future__ import annotations

from typing import Any

from ..core.models import GapProfile, SpeechIsland, Utterance, WordToken
from .particles import bad_island_or_segment_text, is_tail_particle


def words_text(words: list[WordToken]) -> str:
    return "".join(word.text + word.trailing_punct for word in words).strip()


def _make_island(words: list[WordToken], reason: str) -> SpeechIsland:
    return SpeechIsland(
        source_utterance_index=words[0].source_utterance_index,
        words=words, start=words[0].start, end=words[-1].end,
        text=words_text(words), reason=reason,
    )


def _repair_particle_islands(islands: list[SpeechIsland], silence_limit: float) -> list[SpeechIsland]:
    index = 0
    while index < len(islands):
        island = islands[index]
        if not bad_island_or_segment_text(island.text):
            index += 1
            continue
        if index > 0 and island.start - islands[index - 1].end < silence_limit:
            previous = islands[index - 1]
            previous.words.extend(island.words)
            previous.end = island.end
            previous.text = words_text(previous.words)
            previous.reason += "+particle_merged_back"
            islands.pop(index)
        elif index + 1 < len(islands) and islands[index + 1].start - island.end < silence_limit:
            following = islands[index + 1]
            following.words = island.words + following.words
            following.start = island.start
            following.text = words_text(following.words)
            following.reason += "+particle_merged_forward"
            islands.pop(index)
        else:
            index += 1
    return islands


def build_speech_islands(
    utterances: list[Utterance], gap_profile: GapProfile, config: dict[str, Any],
) -> list[SpeechIsland]:
    settings = config["segmentation"]
    configured_gap = float(settings["speech_island_gap_seconds"])
    split_gap = min(configured_gap, gap_profile.strong_gap)
    all_islands: list[SpeechIsland] = []
    for utterance in utterances:
        if not utterance.words:
            continue
        chunks: list[tuple[list[WordToken], str]] = []
        current = [utterance.words[0]]
        current_reason = "utterance_boundary"
        for word, following in zip(utterance.words, utterance.words[1:]):
            gap = max(0.0, following.start - word.end)
            reason = ""
            if gap >= 3.0:
                reason = "very_long_silence"
            elif gap >= configured_gap:
                reason = "speech_island_gap"
            elif gap >= split_gap:
                reason = "strong_gap"
            # Protect particles at ordinary pauses, but a multi-second silence
            # must remain a hard boundary; the repair pass can attach the token
            # to a following phrase when that is temporally plausible.
            if reason == "strong_gap" and is_tail_particle(following.text):
                reason = ""
            if reason:
                chunks.append((current, current_reason))
                current = []
                current_reason = reason
            current.append(following)
        chunks.append((current, current_reason))
        all_islands.extend(_make_island(words, reason) for words, reason in chunks if words)
    return _repair_particle_islands(all_islands, configured_gap)
