from __future__ import annotations

from typing import Any

from .core.config import parse_episodes
from .llm.client import LLMClient, resolve_client
from .llm.pre_review_diagnosis import run_pre_review_diagnosis
from .local_diagnosis import run_local_pre_review_diagnosis
from .local_review import run_local_review
from .segmentation.pipeline import build_segments_from_doubao_result


def run_pipeline(
    episodes: str | list[str],
    paths: Any,
    config: dict[str, Any],
    client: LLMClient | None = None,
) -> dict[str, Any]:
    selected = parse_episodes(episodes) if isinstance(episodes, str) else episodes
    stage = config.get("project", {}).get("run_until", "segmented")
    overwrite = bool(config.get("cache", {}).get("overwrite_existing", False))
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
    if stage == "local-review":
        output = {}
        for episode in selected:
            segments = build_segments_from_doubao_result(episode, paths, config)
            output[episode] = run_local_review(
                episode, paths, config, segments=segments, overwrite=overwrite,
            )
        return output
    if stage == "pre-review-diagnosis":
        output = {}
        active_client = client if client is not None else resolve_client(config)
        for episode in selected:
            segments = build_segments_from_doubao_result(episode, paths, config)
            review = run_local_review(
                episode, paths, config, segments=segments, overwrite=overwrite,
            )
            output[episode] = run_pre_review_diagnosis(
                episode, paths, config,
                client=active_client, local_review=review, overwrite=overwrite,
            )
        return output
    raise NotImplementedError(f"Stage {stage!r} is not available through run_pipeline")
