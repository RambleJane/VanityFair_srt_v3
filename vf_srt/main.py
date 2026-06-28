from __future__ import annotations

from typing import Any

from .core.config import parse_episodes
from .local_diagnosis import run_local_pre_review_diagnosis
from .segmentation.pipeline import build_segments_from_doubao_result


def run_pipeline(episodes: str | list[str], paths: Any, config: dict[str, Any]) -> dict[str, Any]:
    selected = parse_episodes(episodes) if isinstance(episodes, str) else episodes
    stage = config.get("project", {}).get("run_until", "segmented")
    if stage == "segmented":
        return {episode: build_segments_from_doubao_result(episode, paths, config) for episode in selected}
    if stage == "local-diagnosis":
        output: dict[str, Any] = {}
        for episode in selected:
            segments = build_segments_from_doubao_result(episode, paths, config)
            output[episode] = run_local_pre_review_diagnosis(
                episode, paths, config, segments=segments,
                overwrite=bool(config.get("cache", {}).get("overwrite_existing", False)),
            )
        return output
    raise NotImplementedError(f"Stage {stage!r} is not available through run_pipeline")
