from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

from ..core.json_utils import write_json
from ..core.models import GapProfile, Utterance


def collect_word_gaps(utterances: list[Utterance]) -> list[float]:
    gaps: list[float] = []
    for utterance in utterances:
        for previous, following in zip(utterance.words, utterance.words[1:]):
            gap = following.start - previous.end
            if gap >= 0:
                gaps.append(gap)
    return gaps


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def build_gap_profile(
    utterances: list[Utterance], config: dict[str, Any],
    episode: str | None = None, paths: Any | None = None,
) -> GapProfile:
    settings = config["segmentation"]
    gaps = collect_word_gaps(utterances)
    p90, p95, p98, p99 = (percentile(gaps, q) for q in (0.90, 0.95, 0.98, 0.99))
    profile = GapProfile(
        weak_gap=float(settings["weak_gap_default"]),
        soft_gap=max(float(settings["soft_gap_floor"]), p95),
        strong_gap=max(float(settings["strong_gap_floor"]), p98),
        p90=p90, p95=p95, p98=p98, p99=p99,
    )
    if episode is not None and paths is not None:
        write_json(paths.reports_cache_dir / f"{episode}_gap_profile.json", asdict(profile))
    return profile
