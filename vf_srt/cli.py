from __future__ import annotations

import argparse
from pathlib import Path

from .core.config import load_config, parse_episodes
from .core.paths import build_paths
from .segmentation.pipeline import build_segments_from_doubao_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VanityFair subtitle pipeline v3")
    parser.add_argument("--episodes", required=True, help="Comma list or range, e.g. 09,10 or 09-12")
    parser.add_argument("--config", help="YAML config path (defaults to built-in segmentation settings)")
    parser.add_argument("--run-until", default="segmented", help="Currently only: segmented")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild normalized and segmented caches")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.run_until != "segmented":
        raise SystemExit("Only --run-until segmented is implemented; later stage names are reserved TODOs")
    root = Path(__file__).resolve().parents[1]
    config = load_config(args.config)
    config["project"]["run_until"] = args.run_until
    config["cache"]["overwrite_existing"] = args.overwrite
    paths = build_paths(root, config)
    for episode in parse_episodes(args.episodes):
        segments = build_segments_from_doubao_result(episode, paths, config)
        print(f"[{episode}] segmented: {len(segments)} subtitles")
    return 0
