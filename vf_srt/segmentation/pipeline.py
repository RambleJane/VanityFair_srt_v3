from __future__ import annotations

from dataclasses import asdict
from typing import Any

from ..core.cache import use_cache
from ..core.json_utils import read_json, write_json
from ..core.models import SubtitleSegment
from ..doubao.normalize import normalize_result
from .candidates import build_cut_candidates
from .cutter import cut_island_to_segments
from .gap_profile import build_gap_profile
from .islands import build_speech_islands
from .quality import flag_segments
from .repair import repair_segments
from .report import write_segmentation_outputs
from .scorer import score_candidate


def _segment_from_dict(item: dict[str, Any]) -> SubtitleSegment:
    return SubtitleSegment(**item)


def build_segments_from_doubao_result(
    episode: str, paths: Any, config: dict[str, Any], overwrite: bool | None = None,
) -> list[SubtitleSegment]:
    episode = str(episode).zfill(2)
    target = paths.segments_cache_dir / f"{episode}_segments_raw.json"
    if overwrite is not None:
        config = {**config, "cache": {**config.get("cache", {}), "overwrite_existing": overwrite}}
    overwrite_existing = bool(config.get("cache", {}).get("overwrite_existing", False))
    if use_cache(target, overwrite_existing):
        return [_segment_from_dict(item) for item in read_json(target)]
    utterances = normalize_result(episode, paths, config)
    gap_profile = build_gap_profile(utterances, config, episode, paths)
    islands = build_speech_islands(utterances, gap_profile, config)
    all_candidates = []
    draft_segments = []
    for island_index, island in enumerate(islands, start=1):
        candidates = build_cut_candidates(island, gap_profile, config, island_index)
        for candidate in candidates:
            score_candidate(candidate, config, gap_profile)
        all_candidates.extend(candidates)
        draft_segments.extend(cut_island_to_segments(island, candidates, config, gap_profile, episode))
    segments = flag_segments(repair_segments(draft_segments, config), config)
    write_json(target, [asdict(segment) for segment in segments])
    write_segmentation_outputs(episode, utterances, islands, all_candidates, segments, gap_profile, paths)
    return segments
