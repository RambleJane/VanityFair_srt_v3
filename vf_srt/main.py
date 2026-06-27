from __future__ import annotations

from typing import Any

from .core.config import parse_episodes
from .segmentation.pipeline import build_segments_from_doubao_result


def run_pipeline(episodes: str | list[str], paths: Any, config: dict[str, Any]) -> dict[str, Any]:
    selected = parse_episodes(episodes) if isinstance(episodes, str) else episodes
    stage = config.get("project", {}).get("run_until", "segmented")
    if stage != "segmented":
        raise NotImplementedError(f"Stage {stage!r} is reserved for a later v3 milestone")
    return {episode: build_segments_from_doubao_result(episode, paths, config) for episode in selected}
