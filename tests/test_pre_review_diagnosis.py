from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from vf_srt.cli import build_parser
from vf_srt.core.config import DEFAULT_CONFIG
from vf_srt.core.paths import build_paths
from vf_srt.llm.client import LLMNotConfiguredError
from vf_srt.llm.pre_review_diagnosis import (
    SYSTEM_PROMPT,
    build_pre_review_diagnosis,
    build_pre_review_diagnosis_prompt,
    compact_local_review_hints,
    run_pre_review_diagnosis,
)

ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return copy.deepcopy(DEFAULT_CONFIG)


class FakeClient:
    """Records every (system, user) call; never touches the network."""

    def __init__(self, response="{}", *, fenced: bool = False) -> None:
        self.calls: list[dict[str, str]] = []
        self._response = response
        self._fenced = fenced

    def __call__(self, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        body = self._response(system, user) if callable(self._response) else self._response
        if self._fenced:
            return f"这是模型的说明：\n```json\n{body}\n```\n以上。"
        return body


def _valid_response(system: str = "", user: str = "") -> str:
    return json.dumps(
        {
            "summary": {
                "scene_overview": "戏院门外的对话",
                "main_characters": ["李华强"],
                "relationships": ["强哥是兄长"],
                "important_events": ["筹备开戏院"],
                "tone_style": "市井口语",
            },
            "proper_nouns": [
                {"name": "李华强", "type": "person", "evidence": "raw_text", "confidence": "high"}
            ],
            "possible_asr_errors": [
                {"index": 1, "raw_text": "阿强啊", "problem": "近音误听",
                 "suggestion": "阿强", "reason": "官方别名", "confidence": "medium"}
            ],
            "line_hints": [
                {"index": 1, "hint": "注意语气词", "risk_level": "low"}
            ],
            "uncertain_points": ["需人工确认人名"],
        },
        ensure_ascii=False,
    )


def _record(index: int, raw_text: str, hints=None, seg_flags=None, lr_flags=None) -> dict:
    return {
        "index": index,
        "start": 1.0 * index,
        "end": 1.0 * index + 1.5,
        "raw_text": raw_text,
        "flags": ["forced_cut"],
        "debug": {"huge": "x" * 5000},  # must NOT leak into the prompt
        "segmentation_flags": seg_flags or [],
        "local_review_flags": lr_flags or [],
        "local_review_hints": hints or [],
    }


def _local_review(records: list[dict]) -> dict:
    return {
        "episode": "09",
        "stage": "local_review",
        "do_not_auto_apply": True,
        "records": records,
    }


def _paths(tmp_path: Path):
    """Real agent/reference knowledge from ROOT, but caches isolated in tmp."""
    config = _config()
    return replace(
        build_paths(ROOT, config),
        segments_cache_dir=tmp_path / "segments",
        local_review_cache_dir=tmp_path / "local_review",
        pre_review_diagnosis_cache_dir=tmp_path / "prd",
    )


def test_reads_local_review_cache_not_segments(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    config = _config()
    sentinel = "独有哨兵句子SENTINEL"
    review = _local_review([_record(1, sentinel)])
    (paths.local_review_cache_dir).mkdir(parents=True, exist_ok=True)
    (paths.local_review_cache_dir / "09_local_review.json").write_text(
        json.dumps(review, ensure_ascii=False), encoding="utf-8"
    )
    client = FakeClient(_valid_response)

    # No segments cache exists in tmp, so the only source is the local_review file.
    result = build_pre_review_diagnosis("09", paths, config, client=client, overwrite=True)

    assert result["source_stage"] == "local_review"
    assert sentinel in client.calls[0]["user"]


def test_prompt_carries_hints_and_compact_capsule_not_full_agent(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    config = _config()
    hint = {
        "category": "character_name", "span": "李华强", "suggestion": "李华强",
        "confidence": "high", "reason": "本地规则提示：核对人名", "source": "local_rule",
    }
    review = _local_review([_record(1, "阿强啊，李华强嚟咗", hints=[hint])])
    client = FakeClient(_valid_response)

    build_pre_review_diagnosis("09", paths, config, client=client, local_review=review, overwrite=True)
    user = client.calls[0]["user"]

    # local_review_hints reach the model.
    assert "本地规则提示：核对人名" in user
    # Compact knowledge capsule is present and character selection works.
    assert "knowledge_capsule" in user
    assert "relevant_characters" in user
    assert "李华强" in user
    # The huge debug blob must be stripped.
    assert "x" * 5000 not in user
    # Full agent text must NOT be dumped: a later story paragraph is excluded.
    later_paragraph = "康炳仁出身劳苦大众"
    assert later_paragraph not in user


def test_system_prompt_constrains_behaviour() -> None:
    prompt = build_pre_review_diagnosis_prompt("09", 1, 1, [], {})
    system = prompt["system"]
    assert system == SYSTEM_PROMPT
    for phrase in (
        "只提供人工审阅建议",
        "不得改写字幕正文",
        "不得翻译",
        "不得生成 yue_draft",
        "不得凭空补对白",
    ):
        assert phrase in system


def test_output_schema_has_do_not_auto_apply(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    config = _config()
    review = _local_review([_record(1, "阿强啊")])
    client = FakeClient(_valid_response, fenced=True)  # also exercises fenced JSON parsing

    result = run_pre_review_diagnosis(
        "09", paths, config, client=client, local_review=review, overwrite=True
    )

    for key in ("episode", "stage", "source_stage", "do_not_auto_apply",
                "summary", "proper_nouns", "possible_asr_errors",
                "line_hints", "uncertain_points"):
        assert key in result
    assert result["do_not_auto_apply"] is True
    assert result["stage"] == "pre_review_diagnosis"
    assert result["source_stage"] == "local_review"
    assert result["possible_asr_errors"]
    for item in result["possible_asr_errors"]:
        assert item["do_not_auto_apply"] is True
    for item in result["line_hints"]:
        assert item["do_not_auto_apply"] is True
    assert result["stats"]["parse_errors"] == 0


def test_cache_is_reused(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    config = _config()
    review = _local_review([_record(1, "阿强啊")])
    client = FakeClient(_valid_response)

    first = build_pre_review_diagnosis("09", paths, config, client=client, local_review=review, overwrite=True)
    assert len(client.calls) == 1

    # Second run without overwrite reuses the cache and does not call the client.
    second = build_pre_review_diagnosis("09", paths, config, client=client, local_review=review, overwrite=False)
    assert len(client.calls) == 1
    assert second == first


def test_batches_records_and_writes_batch_files(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    config = _config()
    config["pre_review_diagnosis"]["batch_size"] = 50
    review = _local_review([_record(i, f"第{i}句对白") for i in range(1, 121)])
    client = FakeClient(_valid_response)

    result = build_pre_review_diagnosis("09", paths, config, client=client, local_review=review, overwrite=True)

    assert len(client.calls) == 3
    assert result["stats"]["batches"] == 3
    batch_files = sorted(paths.pre_review_diagnosis_cache_dir.glob("09_batch_*.json"))
    assert len(batch_files) == 3


def test_requires_client_when_records_present(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    config = _config()
    review = _local_review([_record(1, "阿强啊")])
    with pytest.raises(LLMNotConfiguredError):
        build_pre_review_diagnosis("09", paths, config, client=None, local_review=review, overwrite=True)


def test_empty_records_needs_no_client(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    config = _config()
    result = build_pre_review_diagnosis(
        "09", paths, config, client=None, local_review=_local_review([]), overwrite=True
    )
    assert result["do_not_auto_apply"] is True
    assert result["possible_asr_errors"] == []
    assert result["stats"]["batches"] == 0


def test_compact_local_review_hints_drops_debug() -> None:
    compact = compact_local_review_hints([_record(1, "阿强啊", hints=[{"category": "x", "span": "y"}])])
    assert compact[0].keys() == {
        "index", "start", "end", "raw_text",
        "segmentation_flags", "local_review_flags", "local_review_hints",
    }
    assert "debug" not in compact[0]


def test_cli_accepts_pre_review_diagnosis_stage() -> None:
    args = build_parser().parse_args([
        "--episodes", "09", "--run-until", "pre-review-diagnosis",
    ])
    assert args.run_until == "pre-review-diagnosis"
