from __future__ import annotations

import copy
import json
from pathlib import Path

from vf_srt.core.config import DEFAULT_CONFIG
from vf_srt.core.models import SubtitleSegment
from vf_srt.core.paths import build_paths
from vf_srt.local_diagnosis import build_local_pre_review_diagnosis
from vf_srt.local_diagnosis.name_rules import build_likely_characters, diagnose_character_names
from vf_srt.local_diagnosis.term_rules import diagnose_terms
from vf_srt.local_diagnosis.theme_rules import diagnose_theme_song
from vf_srt.local_diagnosis.unknown_names import detect_unknown_name_candidates


def _config() -> dict:
    return copy.deepcopy(DEFAULT_CONFIG)


def _knowledge() -> dict:
    return {
        "characters": {
            "official_names": ["徐绍良"],
            "aliases": ["阿良"],
            "alias_to_name": {"阿良": "徐绍良"},
            "raw_entries": [],
        },
        "confirmed_terms": ["喺", "抖阵"],
        "confirmed_glossary_text": "喺\n抖阵",
        "theme_song": {},
    }


def _segment(index: int, text: str, flags: list[str] | None = None) -> dict:
    return {
        "index": index,
        "episode": "09",
        "start": float(index),
        "end": float(index) + 1.0,
        "raw_text": text,
        "flags": flags or [],
        "debug": {},
    }


def test_character_name_rules_and_context_confidence() -> None:
    segments = [
        _segment(1, "佢叫徐少良"),
        _segment(2, "喂，阿梁啊"),
        _segment(3, "哦，阿亮啊"),
        _segment(4, "阿娘啊，你成日做替身"),
    ]
    hints = diagnose_character_names(
        segments, _knowledge(), {"names": {"阿良": 88}}, _config()
    )
    by_span = {hint["span"]: hint for hint in hints}
    assert by_span["徐少良"]["suggestion"] == "徐绍良"
    assert by_span["徐少良"]["confidence"] == "high"
    assert by_span["阿梁"]["suggestion"] == "阿良"
    assert by_span["阿亮"]["suggestion"] == "阿良"
    assert by_span["阿娘"]["confidence"] == "high"
    likely = build_likely_characters(hints, _knowledge())
    assert likely[0]["name"] == "徐绍良"
    assert {"徐少良", "阿梁", "阿亮"}.issubset(likely[0]["aliases_seen"])


def test_term_rules_cover_cantonese_and_period_anachronism() -> None:
    hints = diagnose_terms(
        [_segment(1, "你𫪈训练班"), _segment(2, "抖音啊")],
        _knowledge(), {}, _config(),
    )
    by_span = {hint["span"]: hint for hint in hints}
    assert by_span["𫪈"]["suggestion"] == "喺"
    assert by_span["𫪈"]["confidence"] == "high"
    assert by_span["抖音"]["suggestion"] == "抖阵"
    assert by_span["抖音"]["category"] == "period_anachronism"


def test_unknown_name_and_theme_ending_unmatched_hints() -> None:
    unknown = detect_unknown_name_candidates(
        [_segment(1, "嗱阿蕉以后就要学下人哋")], _knowledge(), {}, _config()
    )
    assert unknown[0]["candidate"] == "阿蕉"
    assert unknown[0]["confidence"] == "low"

    theme = diagnose_theme_song(
        [_segment(2, "无法匹配的片尾残句", ["theme_song", "theme_ending_unmatched"])],
        _knowledge(), _config(),
    )
    assert theme[0]["category"] == "theme_song_unmatched"
    assert theme[0]["confidence"] == "medium"


def test_pipeline_output_is_hint_only_and_does_not_mutate_segments(tmp_path: Path) -> None:
    config = _config()
    paths = build_paths(tmp_path, config)
    segments = [
        SubtitleSegment(1, "09", 1, 0.0, 1.0, "佢叫徐少良", [], {}),
        SubtitleSegment(2, "09", 1, 1.1, 2.0, "你𫪈训练班", [], {}),
    ]
    original = copy.deepcopy(segments)
    result = build_local_pre_review_diagnosis(
        "09", paths, config, segments=segments, overwrite=True
    )

    assert segments == original
    assert result["do_not_auto_apply"] is True
    assert result["stage"] == "local_pre_review_diagnosis"
    assert result["summary"]["total_segments"] == 2
    assert result["summary"]["possible_asr_errors"] == 2
    assert set(result["segment_hints"]) == {"1", "2"}
    written = json.loads(
        (tmp_path / "cache/local_diagnosis/09_local_pre_review_diagnosis.json")
        .read_text(encoding="utf-8")
    )
    assert written["do_not_auto_apply"] is True
