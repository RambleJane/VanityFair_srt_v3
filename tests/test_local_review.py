from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from vf_srt.cli import build_parser
from vf_srt.core.config import DEFAULT_CONFIG
from vf_srt.core.models import SubtitleSegment
from vf_srt.core.paths import build_paths
from vf_srt.local_review import build_local_review
from vf_srt.local_review.flags import extract_segmentation_flags


def _config() -> dict:
    return copy.deepcopy(DEFAULT_CONFIG)


def test_segmentation_flag_filter_excludes_content_flags() -> None:
    assert extract_segmentation_flags([
        "too_short_duration",
        "theme_song",
        "fixed_lyric",
        "pressure_cut",
    ]) == ["too_short_duration", "pressure_cut"]


def test_local_review_is_non_mutating_and_writes_formal_schema(tmp_path: Path) -> None:
    config = _config()
    paths = build_paths(tmp_path, config)
    segments = [
        SubtitleSegment(
            1, "09", 1, 1.25, 2.75, "佢叫徐少良",
            ["too_short_duration", "theme_song", "fixed_lyric"],
            {"theme_song": True},
        ),
        SubtitleSegment(
            2, "09", 1, 3.0, 4.5, "你𫪈训练班",
            ["pressure_cut", "forced_cut"], {},
        ),
        SubtitleSegment(3, "09", 1, 5.0, 6.0, "抖音啊", [], {}),
        SubtitleSegment(4, "09", 1, 6.5, 7.5, "嗱阿蕉嚟咗", [], {}),
        SubtitleSegment(
            5, "09", 1, 8.0, 9.0, "无法匹配的片尾残句",
            ["theme_song", "theme_ending_unmatched"], {},
        ),
    ]
    original = copy.deepcopy(segments)

    result = build_local_review(
        "09", paths, config, segments=segments, overwrite=True
    )

    assert segments == original
    assert result["stage"] == "local_review"
    assert result["do_not_auto_apply"] is True
    assert len(result["records"]) == len(segments)

    first = result["records"][0]
    assert first["raw_text"] == original[0].raw_text
    assert first["start"] == original[0].start
    assert first["end"] == original[0].end
    assert first["flags"] == original[0].flags
    assert first["debug"] == original[0].debug
    assert first["segmentation_flags"] == ["too_short_duration"]
    assert "theme_song" not in first["segmentation_flags"]
    assert "fixed_lyric" not in first["segmentation_flags"]
    assert "possible_name_mishear" in first["local_review_flags"]

    second = result["records"][1]
    assert second["segmentation_flags"] == ["pressure_cut", "forced_cut"]
    assert second["local_review_flags"] == ["possible_cantonese_word"]
    assert result["records"][2]["local_review_flags"] == ["possible_period_term"]
    assert "possible_unknown_name" in result["records"][3]["local_review_flags"]
    assert "possible_theme_song_issue" in result["records"][4]["local_review_flags"]

    required_hint_fields = {
        "confidence", "category", "reason", "source", "suggestion",
        "do_not_auto_apply",
    }
    for record in result["records"]:
        for hint in record["local_review_hints"]:
            assert required_hint_fields <= set(hint)
            assert hint["do_not_auto_apply"] is True

    written = json.loads(
        (tmp_path / "cache/local_review/09_local_review.json")
        .read_text(encoding="utf-8")
    )
    assert written["records"][0]["raw_text"] == original[0].raw_text


def test_cli_accepts_local_review_stage() -> None:
    args = build_parser().parse_args([
        "--episodes", "09-12", "--run-until", "local-review",
    ])
    assert args.run_until == "local-review"


@pytest.mark.parametrize("episode", ["09", "10", "11", "12"])
def test_existing_segment_caches_can_run_local_review(
    episode: str, tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "cache/segments" / f"{episode}_segments_raw.json"
    if not source.is_file():
        pytest.skip(f"local integration cache is not present: {source.name}")
    config = _config()
    paths = replace(
        build_paths(root, config),
        local_review_cache_dir=tmp_path / "local_review",
    )

    result = build_local_review(episode, paths, config, overwrite=True)

    source_rows = json.loads(source.read_text(encoding="utf-8-sig"))
    assert result["summary"]["total_segments"] == len(source_rows)
    assert [record["raw_text"] for record in result["records"]] == [
        row["raw_text"] for row in source_rows
    ]
    assert [record["start"] for record in result["records"]] == [
        row["start"] for row in source_rows
    ]
    assert [record["end"] for record in result["records"]] == [
        row["end"] for row in source_rows
    ]
