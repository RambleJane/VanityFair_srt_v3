from __future__ import annotations

from typing import Any

from ..core.models import CutCandidate, GapProfile, SpeechIsland


def char_count(text: str) -> int:
    return sum(1 for char in text if not char.isspace() and char not in "，。！？；：、,.!?;:…—～~（）()【】[]《》〈〉“”‘’\"'")


def build_cut_candidates(
    island: SpeechIsland, gap_profile: GapProfile, config: dict[str, Any], island_index: int = 0,
) -> list[CutCandidate]:
    candidates: list[CutCandidate] = []
    for position, (word, following) in enumerate(zip(island.words, island.words[1:])):
        before = "".join(item.text + item.trailing_punct for item in island.words[:position + 1])
        after = "".join(item.text + item.trailing_punct for item in island.words[position + 1:])
        candidates.append(CutCandidate(
            island_index=island_index, word_pos=position, time=word.end,
            gap_after=max(0.0, following.start - word.end), prev_text=word.text,
            next_text=following.text, trailing_punct=word.trailing_punct,
            chars_before=char_count(before), chars_after=char_count(after),
            duration_before=max(0.0, word.end - island.start),
            duration_after=max(0.0, island.end - following.start), score=0.0, reasons=[],
        ))
    return candidates
