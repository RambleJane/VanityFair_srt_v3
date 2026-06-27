from __future__ import annotations

import argparse
from pathlib import Path

from ..core.config import load_config, parse_episodes
from ..core.paths import build_paths
from ..segmentation.pipeline import build_segments_from_doubao_result
from .export_debug_xlsx import export_debug_xlsx


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run offline segmentation lab")
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--config")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    config = load_config(args.config)
    config["cache"]["overwrite_existing"] = args.overwrite
    paths = build_paths(root, config)
    for episode in parse_episodes(args.episodes):
        segments = build_segments_from_doubao_result(episode, paths, config)
        workbook = export_debug_xlsx(episode, paths)
        suffix = f", xlsx: {workbook.name}" if workbook else ", xlsx exporter unavailable"
        print(f"[{episode}] {len(segments)} segments{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
