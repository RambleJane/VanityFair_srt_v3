from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WordToken:
    text: str
    start: float
    end: float
    source_utterance_index: int
    word_index: int
    trailing_punct: str = ""


@dataclass
class Utterance:
    index: int
    start: float
    end: float
    text: str
    words: list[WordToken]


@dataclass
class GapProfile:
    weak_gap: float
    soft_gap: float
    strong_gap: float
    p90: float
    p95: float
    p98: float
    p99: float


@dataclass
class SpeechIsland:
    source_utterance_index: int
    words: list[WordToken]
    start: float
    end: float
    text: str
    reason: str


@dataclass
class CutCandidate:
    island_index: int
    word_pos: int
    time: float
    gap_after: float
    prev_text: str
    next_text: str
    trailing_punct: str
    chars_before: int
    chars_after: int
    duration_before: float
    duration_after: float
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)


@dataclass
class SubtitleSegment:
    index: int
    episode: str
    source_utterance_index: int
    start: float
    end: float
    raw_text: str
    flags: list[str] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)
