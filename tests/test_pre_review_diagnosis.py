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


class ScriptedClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, str]] = []

    def __call__(self, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]


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


def test_cli_accepts_rerun_failed_batches() -> None:
    args = build_parser().parse_args([
        "--episodes", "09", "--run-until", "pre-review-diagnosis",
        "--rerun-failed-batches",
    ])
    assert args.rerun_failed_batches is True


def _response(**overrides) -> str:
    value = {
        "summary": {},
        "proper_nouns": [],
        "possible_asr_errors": [],
        "line_hints": [],
        "uncertain_points": [],
    }
    value.update(overrides)
    return json.dumps(value, ensure_ascii=False)


def test_source_facts_override_model_raw_text_and_structured_suggestion(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    config = _config()
    source = _record(
        338, "哎呀，四万𫪈门口我食啊嗱。",
        seg_flags=["pressure_cut"], lr_flags=["possible_cantonese_word"],
    )
    model = {
        "index": 338, "raw_text": "𫪈", "observed_span": "𫪈",
        "action": "replace", "suggested_span": "喺", "suggested_full_text": None,
        "problem": "异体字", "reason": "需人工确认", "confidence": "high",
    }
    hint = {
        "index": 338, "raw_text": "错误模型文本", "category": "term",
        "observed_span": "𫪈", "action": "listen", "suggested_span": "喺",
        "hint": "核对粤语字", "reason": "罕见字", "risk_level": "medium",
    }

    result = build_pre_review_diagnosis(
        "09", paths, config, client=FakeClient(_response(
            possible_asr_errors=[model], line_hints=[hint],
        )), local_review=_local_review([source]), overwrite=True,
    )

    error = result["possible_asr_errors"][0]
    for key in ("episode", "index", "start", "end", "raw_text",
                "segmentation_flags", "local_review_flags"):
        expected = "09" if key == "episode" else source[key]
        assert error[key] == expected
    assert error["observed_span"] == "𫪈"
    assert error["suggested_span"] == "喺"
    assert error["suggested_full_text"] is None
    assert error["action"] == "listen"  # rare character requires human confirmation
    assert error["confidence"] == "medium"
    assert error["do_not_auto_apply"] is True
    assert result["line_hints"][0]["raw_text"] == source["raw_text"]
    assert result["validation"]["raw_text_repaired_count"] == 2


def test_invalid_indices_and_relative_references_are_quarantined(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    result = build_pre_review_diagnosis(
        "09", paths, _config(), client=FakeClient(_response(
            possible_asr_errors=[
                {"index": 999, "observed_span": "错", "action": "replace",
                 "suggested_span": "对", "reason": "不存在", "confidence": "high"},
                {"index": 1, "observed_span": "阿强", "action": "replace",
                 "suggested_span": "李华强", "reason": "同第104条", "confidence": "high"},
            ],
            line_hints=[
                {"index": 1, "category": "name", "action": "uncertain",
                 "hint": "同上", "reason": "同前", "risk_level": "medium"},
            ],
            uncertain_points=["same as above"],
        )), local_review=_local_review([_record(1, "阿强啊")]), overwrite=True,
    )

    assert result["possible_asr_errors"] == []
    assert result["line_hints"] == []
    assert result["uncertain_points"] == []
    assert len(result["invalid_model_items"]) == 4
    assert result["validation"]["invalid_index_count"] == 1
    assert result["validation"]["relative_reference_count"] == 3


def test_invalid_enums_and_number_suggestions_are_downgraded(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    result = build_pre_review_diagnosis(
        "09", paths, _config(), client=FakeClient(_response(
            possible_asr_errors=[
                {"index": 1, "observed_span": "34A/32", "action": "replace",
                 "suggested_span": "三十四A/三十二", "suggested_full_text": "整行改写",
                 "problem": "泳衣尺码数字", "reason": "可能是尺码", "confidence": "high"},
                {"index": 2, "observed_span": "词", "action": "rewrite",
                 "suggested_span": "字", "reason": "猜测", "confidence": "certain"},
            ],
        )), local_review=_local_review([
            _record(1, "老细照34A同我拎32啦"), _record(2, "普通词"),
        ]), overwrite=True,
    )

    number = result["possible_asr_errors"][0]
    assert number["confidence"] == "low"
    assert number["action"] == "uncertain"
    assert number["suggested_full_text"] is None
    invalid_enum = result["possible_asr_errors"][1]
    assert invalid_enum["confidence"] == "low"
    assert invalid_enum["action"] == "uncertain"
    assert result["validation"]["invalid_schema_count"] >= 2


def test_proper_nouns_are_canonicalized_and_model_inference_is_not_confirmed(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    result = build_pre_review_diagnosis(
        "09", paths, _config(), client=FakeClient(_response(
            proper_nouns=[
                {"canonical_name": "阿良", "aliases_seen": ["阿梁"], "type": "person",
                 "evidence_indices": [1], "evidence": "index 1", "confidence": "high"},
                {"name": "徐绍良", "type": "person", "evidence": "第1条", "confidence": "high"},
                {"name": "新人物甲", "type": "person", "status": "confirmed",
                 "evidence": "模型推断", "confidence": "medium"},
            ],
        )), local_review=_local_review([_record(1, "阿梁啊")]), overwrite=True,
    )

    by_name = {item["canonical_name"]: item for item in result["proper_nouns"]}
    assert set(by_name) == {"徐绍良", "新人物甲"}
    assert "阿良" in by_name["徐绍良"]["aliases_seen"]
    assert "阿梁" in by_name["徐绍良"]["aliases_seen"]
    assert by_name["徐绍良"]["source"] == "official"
    assert by_name["徐绍良"]["status"] == "confirmed"
    assert by_name["新人物甲"]["source"] == "model_infer"
    assert by_name["新人物甲"]["status"] == "inferred"


def test_batch_summaries_are_preserved_without_text_concatenation(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    config = _config()
    config["deepseek"]["stages"]["pre_review_diagnosis"]["batch_size"] = 2

    def response(_system: str, user: str) -> str:
        prompt = json.loads(user)
        batch = prompt["batch_index"]
        return _response(summary={
            "scene_overview": f"第{batch}批场景",
            "main_characters": ["阿良" if batch == 1 else "徐绍良"],
            "relationships": [f"第{batch}批推断关系"],
            "important_events": [f"第{batch}批事件"],
            "tone_style": f"第{batch}批语气",
        })

    result = build_pre_review_diagnosis(
        "09", paths, config, client=FakeClient(response),
        local_review=_local_review([_record(i, f"第{i}句") for i in range(1, 5)]),
        overwrite=True,
    )

    assert len(result["batch_summaries"]) == 2
    assert result["batch_summaries"][0]["start_index"] == 1
    assert result["batch_summaries"][0]["end_index"] == 2
    assert result["batch_summaries"][1]["start_index"] == 3
    assert result["batch_summaries"][1]["end_index"] == 4
    assert result["summary"]["scene_overview"] == ""
    assert result["summary"]["relationships"] == []
    assert result["summary"]["main_characters"] == ["徐绍良"]
    assert result["summary"]["aggregation"] == "batch_summaries_only_no_reduce"


def test_prompt_forbids_relative_references_and_mixed_suggestions() -> None:
    system = build_pre_review_diagnosis_prompt("09", 1, 1, [], {})["system"]
    for phrase in ("不要输出或改写 raw_text", "同上", "observed_span", "suggested_span", "数字、尺码"):
        assert phrase in system


def test_parse_failure_retries_then_succeeds(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    client = ScriptedClient(["not json", _response()])

    result = build_pre_review_diagnosis(
        "09", paths, _config(), client=client,
        local_review=_local_review([_record(1, "普通对白")]), overwrite=True,
    )

    assert len(client.calls) == 2
    assert "上一次输出无法解析为 JSON" in client.calls[1]["user"]
    assert result["status"] == "complete"
    assert result["failed_batch_ids"] == []
    assert result["stats"]["parse_errors"] == 0
    assert result["stats"]["parse_retries"] == 1
    batch = json.loads((paths.pre_review_diagnosis_cache_dir / "09_batch_0001.json").read_text(encoding="utf-8"))
    assert batch["status"] == "complete"
    assert batch["parse_attempt_count"] == 2
    assert batch["raw_model_content_preview"] == _response()
    assert batch["raw_model_content_truncated"] is False


def test_parse_retries_exhausted_marks_incomplete_and_saves_preview(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    config = _config()
    config["deepseek"]["stages"]["pre_review_diagnosis"]["parse_retry_attempts"] = 2
    invalid = "x" * 4000
    client = ScriptedClient([invalid])

    result = build_pre_review_diagnosis(
        "09", paths, config, client=client,
        local_review=_local_review([_record(1, "普通对白")]), overwrite=True,
    )

    assert len(client.calls) == 3
    assert result["status"] == "incomplete"
    assert result["failed_batch_ids"] == [1]
    assert result["stats"]["parse_errors"] == 1
    assert result["validation"]["parse_errors"] == 1
    assert result["validation"]["failed_batch_count"] == 1
    assert result["validation"]["failed_batch_ids"] == [1]
    batch = json.loads((paths.pre_review_diagnosis_cache_dir / "09_batch_0001.json").read_text(encoding="utf-8"))
    assert batch["status"] == "failed"
    assert len(batch["raw_model_content_preview"]) == 3000
    assert batch["raw_model_content_truncated"] is True


def test_rerun_failed_batches_only_requests_failed_batch(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    config = _config()
    config["deepseek"]["stages"]["pre_review_diagnosis"].update({
        "batch_size": 2, "parse_retry_attempts": 2,
    })
    review = _local_review([_record(i, f"第{i}句") for i in range(1, 5)])
    batch_one = _response(summary={"scene_overview": "第一批"})
    first_client = ScriptedClient([batch_one, "bad json"])
    first = build_pre_review_diagnosis(
        "09", paths, config, client=first_client,
        local_review=review, overwrite=True,
    )
    assert first["failed_batch_ids"] == [2]
    assert len(first_client.calls) == 4  # batch 1 once; batch 2 initial + 2 retries

    batch_two = _response(summary={"scene_overview": "第二批"})
    rerun_client = ScriptedClient([batch_two])
    second = build_pre_review_diagnosis(
        "09", paths, config, client=rerun_client,
        local_review=review, rerun_failed_batches=True,
    )

    assert len(rerun_client.calls) == 1
    assert second["status"] == "complete"
    assert second["failed_batch_ids"] == []
    assert second["stats"]["parse_errors"] == 0
    assert [item["scene_overview"] for item in second["batch_summaries"]] == [
        "第一批", "第二批",
    ]


@pytest.mark.parametrize("relative", [
    "同153", "同 index 252", "同index 252", "与 index 252 相同",
    "与index252相同", "同 153", "同第153", "同第153条",
    "如上", "见上", "参考上条", "same as above", "same as index 252",
])
def test_extended_relative_references_are_quarantined(
    relative: str, tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    result = build_pre_review_diagnosis(
        "09", paths, _config(), client=FakeClient(_response(
            possible_asr_errors=[{
                "index": 1, "observed_span": "错", "action": "replace",
                "suggested_span": "对", "problem": "错误", "reason": relative,
                "confidence": "high",
            }],
        )), local_review=_local_review([_record(1, "错字")]), overwrite=True,
    )
    assert result["possible_asr_errors"] == []
    assert result["validation"]["relative_reference_count"] == 1
    assert result["invalid_model_items"][0]["reason"] == "relative_reference"


def test_evidence_numbers_do_not_downgrade_name_correction(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    result = build_pre_review_diagnosis(
        "09", paths, _config(), client=FakeClient(_response(
            possible_asr_errors=[{
                "index": 1, "observed_span": "阿梁", "action": "replace",
                "suggested_span": "阿良", "problem": "人名误听",
                "reason": "前8集出现88次，第213行也有证据", "confidence": "high",
            }],
        )), local_review=_local_review([_record(1, "阿梁啊")]), overwrite=True,
    )
    item = result["possible_asr_errors"][0]
    assert item["confidence"] == "high"
    assert item["action"] == "replace"


def test_real_size_context_still_downgrades_confidence(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    result = build_pre_review_diagnosis(
        "09", paths, _config(), client=FakeClient(_response(
            possible_asr_errors=[{
                "index": 1, "observed_span": "4", "action": "replace",
                "suggested_span": "四", "problem": "泳衣尺码数字",
                "reason": "可能是尺码", "confidence": "high",
            }],
        )), local_review=_local_review([_record(1, "照34A拎32")]), overwrite=True,
    )
    item = result["possible_asr_errors"][0]
    assert item["confidence"] == "low"
    assert item["action"] == "uncertain"


def test_common_noun_is_filtered_from_proper_nouns(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    result = build_pre_review_diagnosis(
        "09", paths, _config(), client=FakeClient(_response(
            proper_nouns=[{
                "canonical_name": "导演", "type": "term",
                "source": "confirmed_glossary", "status": "confirmed",
                "evidence_indices": [1], "evidence": "普通职业", "confidence": "high",
            }],
        )), local_review=_local_review([_record(1, "导演嚟咗")]), overwrite=True,
    )
    assert result["proper_nouns"] == []
    assert result["validation"]["proper_noun_filtered_count"] == 1
    assert result["invalid_model_items"][0]["reason"] == "common_noun"
