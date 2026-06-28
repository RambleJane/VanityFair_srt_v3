from __future__ import annotations

import argparse
from pathlib import Path

from .core.config import load_config, parse_episodes
from .core.paths import build_paths
from .knowledge.reference_profile import write_reference_profile
from .llm.client import LLMNotConfiguredError, resolve_client
from .llm.pre_review_diagnosis import run_pre_review_diagnosis
from .local_diagnosis import run_local_pre_review_diagnosis
from .local_review import run_local_review
from .segmentation.pipeline import build_segments_from_doubao_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VanityFair subtitle pipeline v3")
    parser.add_argument("--episodes", help="Comma list or range, e.g. 09,10 or 09-12")
    parser.add_argument("--config", help="YAML config path (defaults to built-in segmentation settings)")
    parser.add_argument(
        "--run-until", default="segmented",
        choices=(
            "segmented", "reference-profile", "local-diagnosis",
            "local-review", "pre-review-diagnosis",
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Rebuild normalized and segmented caches")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the reference profile from local 01-08 SRT")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    config = load_config(args.config)
    config["project"]["run_until"] = args.run_until
    config["cache"]["overwrite_existing"] = args.overwrite
    paths = build_paths(root, config)
    if args.run_until == "reference-profile":
        value = Path(config["reference_profile"]["json_path"])
        profile_path = value if value.is_absolute() else paths.root / value
        force = bool(args.overwrite or args.rebuild)
        if profile_path.is_file() and not force:
            print(f"Reference profile exists; reused: {profile_path}")
            return 0
        try:
            profile = write_reference_profile(paths, config, overwrite=force)
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from None
        print(f"Reference profile written: {profile_path} ({profile['source']['line_count']} lines)")
        return 0
    if not args.episodes:
        raise SystemExit(f"--episodes is required for --run-until {args.run_until}")
    for episode in parse_episodes(args.episodes):
        segments = build_segments_from_doubao_result(episode, paths, config)
        if args.run_until == "local-diagnosis":
            diagnosis = run_local_pre_review_diagnosis(
                episode, paths, config, segments=segments, overwrite=args.overwrite,
            )
            summary = diagnosis["summary"]
            print(
                f"[{episode}] local diagnosis: "
                f"{summary['possible_asr_errors']} ASR hints, "
                f"{summary['unknown_name_candidates']} unknown-name candidates"
            )
        elif args.run_until == "local-review":
            local_review = run_local_review(
                episode, paths, config, segments=segments, overwrite=args.overwrite,
            )
            summary = local_review["summary"]
            print(
                f"[{episode}] local review: "
                f"{summary['segments_with_local_review_flags']} flagged segments, "
                f"{summary['total_local_review_hints']} hints"
            )
        elif args.run_until == "pre-review-diagnosis":
            review = run_local_review(
                episode, paths, config, segments=segments, overwrite=args.overwrite,
            )
            try:
                diagnosis = run_pre_review_diagnosis(
                    episode, paths, config,
                    client=resolve_client(config), local_review=review,
                    overwrite=args.overwrite,
                )
            except LLMNotConfiguredError as exc:
                raise SystemExit(
                    f"[{episode}] pre-review-diagnosis 需要 LLM client，本里程碑未接入网络客户端：{exc}"
                ) from None
            stats = diagnosis["stats"]
            print(
                f"[{episode}] pre-review diagnosis: "
                f"{len(diagnosis['possible_asr_errors'])} ASR notes, "
                f"{len(diagnosis['line_hints'])} line hints "
                f"({stats['batches']} batches)"
            )
        else:
            print(f"[{episode}] segmented: {len(segments)} subtitles")
    return 0
