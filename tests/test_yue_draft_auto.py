from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

from vf_srt.cli import build_parser
from vf_srt.core.config import DEFAULT_CONFIG
from vf_srt.core.paths import build_paths
from vf_srt.llm.yue_draft_auto import (
    build_yue_draft_auto_lines,
    collect_diagnosis_hints_by_index,
)

ROOT = Path(__file__).resolve().parents[1]


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses) if isinstance(responses, list) else [responses]
        self.calls = []

    def __call__(self, system, user):
        self.calls.append((system, user))
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


def _config():
    return copy.deepcopy(DEFAULT_CONFIG)


def _paths(tmp_path):
    return replace(
        build_paths(ROOT, _config()),
        segments_cache_dir=tmp_path / "segments",
        local_review_cache_dir=tmp_path / "local_review",
        pre_review_diagnosis_cache_dir=tmp_path / "diagnosis",
        yue_draft_cache_dir=tmp_path / "yue_draft",
    )


def _record(index, text, *, seg=None, local=None):
    return {
        "episode": "09", "index": index, "start": float(index),
        "end": float(index) + 1.2, "raw_text": text,
        "segmentation_flags": seg or [], "local_review_flags": local or [],
        "local_review_hints": [],
    }


def _local(*records):
    return {"episode": "09", "stage": "local_review", "records": list(records)}


def _diagnosis(**values):
    result = {
        "episode": "09", "stage": "pre_review_diagnosis",
        "proper_nouns": [], "possible_asr_errors": [], "line_hints": [],
        "uncertain_points": [], "batch_summaries": [],
    }
    result.update(values)
    return result


def _response(*records):
    return json.dumps({"records": list(records)}, ensure_ascii=False)


def test_reads_both_formal_caches_and_never_segments_or_invalid_items(tmp_path):
    paths = _paths(tmp_path)
    paths.local_review_cache_dir.mkdir(parents=True)
    paths.pre_review_diagnosis_cache_dir.mkdir(parents=True)
    local = _local(_record(1, "独有事实SENTINEL"))
    diagnosis = _diagnosis(invalid_model_items=[{
        "index": 1, "action": "replace", "observed_span": "独有事实SENTINEL",
        "suggested_span": "不应出现",
    }])
    (paths.local_review_cache_dir / "09_local_review.json").write_text(json.dumps(local), encoding="utf-8")
    (paths.pre_review_diagnosis_cache_dir / "09_pre_review_diagnosis.json").write_text(json.dumps(diagnosis), encoding="utf-8")
    client = FakeClient(_response({"index": 1, "yue_draft": "独有事实SENTINEL"}))

    result = build_yue_draft_auto_lines("09", None, None, paths, _config(), client, overwrite=True)

    assert result["records"][0]["raw_text"] == "独有事实SENTINEL"
    assert "invalid_model_items" not in client.calls[0][1]
    assert "不应出现" not in client.calls[0][1]


def test_fact_fields_are_always_reattached_from_local_review(tmp_path):
    source = _record(1, "原句", seg=["forced_cut"], local=["possible_asr"])
    model = {"index": 1, "episode": "77", "start": 99, "end": 100,
             "raw_text": "伪造", "yue_draft": "原句"}
    result = build_yue_draft_auto_lines(
        "09", _local(source), _diagnosis(), _paths(tmp_path), _config(),
        FakeClient(_response(model)), overwrite=True,
    )
    record = result["records"][0]
    assert (record["episode"], record["start"], record["end"], record["raw_text"]) == ("09", 1.0, 2.2, "原句")
    assert record["segmentation_flags"] == ["forced_cut"]
    assert record["local_review_flags"] == ["possible_asr"]
    assert record["reviewer_yue_master"] == ""
    assert record["do_not_auto_apply_to_master"] is True


def test_action_and_confidence_policy(tmp_path):
    records = [_record(1, "阿良嚟咗"), _record(2, "听唔清"), _record(3, "照旧啦"),
               _record(4, "疑似错字"), _record(5, "34A拎32"), _record(6, "生僻𠺢字")]
    hints = [
        {"index": 1, "observed_span": "阿良", "suggested_span": "阿梁", "action": "replace", "confidence": "high"},
        {"index": 2, "observed_span": "听唔清", "action": "listen", "confidence": "medium"},
        {"index": 3, "observed_span": "照旧", "action": "keep", "confidence": "high"},
        {"index": 4, "observed_span": "错字", "suggested_span": "錯字", "action": "replace", "confidence": "low"},
        {"index": 5, "observed_span": "34A", "suggested_span": "38A", "action": "replace", "confidence": "high"},
        {"index": 6, "observed_span": "𠺢", "suggested_span": "啲", "action": "replace", "confidence": "high"},
    ]
    proposals = [
        {"index": 1, "yue_draft": "阿梁嚟咗"}, {"index": 2, "yue_draft": "擅自补句"},
        {"index": 3, "yue_draft": "改咗啦"}, {"index": 4, "yue_draft": "疑似錯字"},
        {"index": 5, "yue_draft": "38A拎32"}, {"index": 6, "yue_draft": "生僻啲字"},
    ]
    result = build_yue_draft_auto_lines(
        "09", _local(*records), _diagnosis(possible_asr_errors=hints),
        _paths(tmp_path), _config(), FakeClient(_response(*proposals)), overwrite=True,
    )
    by_index = {r["index"]: r for r in result["records"]}
    assert (by_index[1]["yue_draft"], by_index[1]["draft_status"]) == ("阿梁嚟咗", "changed")
    assert (by_index[2]["yue_draft"], by_index[2]["draft_status"]) == ("听唔清", "needs_listen")
    assert (by_index[3]["yue_draft"], by_index[3]["draft_status"]) == ("照旧啦", "unchanged")
    assert by_index[4]["yue_draft"] == "疑似错字" and by_index[4]["draft_status"] == "uncertain"
    assert by_index[5]["yue_draft"] == "34A拎32"
    assert by_index[6]["yue_draft"] == "生僻𠺢字"


def test_theme_line_is_not_changed(tmp_path):
    source = _record(1, "主题曲原句", local=["possible_theme_song_issue"])
    hint = {"index": 1, "observed_span": "原句", "suggested_span": "新句", "action": "replace", "confidence": "high"}
    result = build_yue_draft_auto_lines(
        "09", _local(source), _diagnosis(line_hints=[hint]), _paths(tmp_path),
        _config(), FakeClient(_response({"index": 1, "yue_draft": "主题曲新句"})), overwrite=True,
    )
    assert result["records"][0]["yue_draft"] == "主题曲原句"


def test_conservative_model_only_traditionalization_is_allowed(tmp_path):
    result = build_yue_draft_auto_lines(
        "09", _local(_record(1, "听日见")), _diagnosis(), _paths(tmp_path), _config(),
        FakeClient(_response({
            "index": 1, "yue_draft": "聽日見", "draft_confidence": "high",
        })), overwrite=True,
    )
    assert result["records"][0]["yue_draft"] == "聽日見"
    assert result["records"][0]["draft_status"] == "changed"


def test_invalid_model_index_is_isolated(tmp_path):
    result = build_yue_draft_auto_lines(
        "09", _local(_record(1, "原句")), _diagnosis(), _paths(tmp_path), _config(),
        FakeClient(_response({"index": 999, "yue_draft": "越界"}, {"index": 1, "yue_draft": "原句"})), overwrite=True,
    )
    assert [r["index"] for r in result["records"]] == [1]
    assert result["validation"]["invalid_index_count"] == 1
    assert result["invalid_model_items"][0]["item"]["index"] == 999


def test_parse_retry_failure_status_and_batch_cache(tmp_path):
    config = _config()
    config["deepseek"]["stages"]["yue_draft_auto_lines"].update({"batch_size": 1, "parse_retry_attempts": 1})
    client = FakeClient(["not json", _response({"index": 1, "yue_draft": "原句"}), "bad", "still bad"])
    result = build_yue_draft_auto_lines(
        "09", _local(_record(1, "原句"), _record(2, "另一句")), _diagnosis(),
        _paths(tmp_path), config, client, overwrite=True,
    )
    assert result["status"] == "incomplete"
    assert result["failed_batch_ids"] == [2]
    assert result["stats"]["parse_retries"] == 2
    assert result["records"][1]["draft_status"] == "uncertain"
    batch = json.loads((_paths(tmp_path).yue_draft_cache_dir / "09_batch_0002.json").read_text(encoding="utf-8"))
    assert batch["status"] == "failed" and "raw_model_content_preview" in batch


def test_rerun_failed_batches_reuses_successful_batch(tmp_path):
    paths = _paths(tmp_path)
    config = _config()
    config["deepseek"]["stages"]["yue_draft_auto_lines"].update({"batch_size": 1, "parse_retry_attempts": 0})
    local = _local(_record(1, "第一句"), _record(2, "第二句"))
    first_client = FakeClient([_response({"index": 1, "yue_draft": "第一句"}), "bad"])
    first = build_yue_draft_auto_lines(
        "09", local, _diagnosis(), paths, config, first_client, overwrite=True,
    )
    assert first["failed_batch_ids"] == [2]

    rerun_client = FakeClient(_response({"index": 2, "yue_draft": "第二句"}))
    second = build_yue_draft_auto_lines(
        "09", local, _diagnosis(), paths, config, rerun_client,
        rerun_failed_batches=True,
    )
    assert second["status"] == "complete" and second["failed_batch_ids"] == []
    assert len(rerun_client.calls) == 1
    assert [record["index"] for record in second["records"]] == [1, 2]


def test_collect_hints_accepts_only_formal_line_lists():
    diagnosis = _diagnosis(
        possible_asr_errors=[{"index": 1, "action": "listen", "confidence": "low"}],
        invalid_model_items=[{"index": 2, "action": "replace", "confidence": "high"}],
        raw_model_content_preview="secret",
    )
    assert set(collect_diagnosis_hints_by_index(diagnosis)) == {1}


def test_cli_accepts_yue_draft_auto():
    args = build_parser().parse_args(["--episodes", "09", "--run-until", "yue-draft-auto"])
    assert args.run_until == "yue-draft-auto"
