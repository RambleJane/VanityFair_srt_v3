from __future__ import annotations

import json
from pathlib import Path

from vf_srt.core.config import DEFAULT_CONFIG
from vf_srt.core.models import SubtitleSegment
from vf_srt.core.paths import build_paths
from vf_srt.knowledge.reference_profile import load_reference_profile, write_reference_profile
from vf_srt.local_diagnosis import run_local_pre_review_diagnosis


def _config() -> dict:
    return json.loads(json.dumps(DEFAULT_CONFIG, ensure_ascii=False))


def _write_fake_reference_sources(root: Path) -> None:
    reference = root / "reference" / "simplified_human"
    reference.mkdir(parents=True, exist_ok=True)
    for episode in range(1, 9):
        text = "喂阿良\n\n不用客气\n\n不用客气" if episode == 1 else "阿良早晨"
        (reference / f"{episode:02d}.srt").write_text(
            f"1\n00:00:01,000 --> 00:00:02,000\n{text}\n",
            encoding="utf-8-sig",
        )
    agent = root / "agent"
    agent.mkdir(parents=True, exist_ok=True)
    (agent / "characters_official.json").write_text(json.dumps({
        "characters": [{
            "actor_simplified": "郑少秋",
            "role_simplified": "徐绍良",
            "aliases_simplified": ["阿良", "良仔"],
        }],
    }, ensure_ascii=False), encoding="utf-8")
    (agent / "05_story_outline_authoritative.md").write_text("徐绍良在戏院工作。", encoding="utf-8")
    (agent / "06_story_clues_verified_names_uncertain.md").write_text("阿良进入电影公司。", encoding="utf-8")


def test_write_and_load_reference_profile(tmp_path: Path) -> None:
    config = _config()
    paths = build_paths(tmp_path, config)
    _write_fake_reference_sources(tmp_path)

    profile = write_reference_profile(paths, config, overwrite=True)

    assert profile["usage"]
    assert profile["high_frequency_names"][0]["name"] == "徐绍良"
    assert any(item["term"] == "阿良" for item in profile["address_terms"])
    assert (tmp_path / "reference/profile/reference_srt_profile.json").is_file()
    assert (tmp_path / "reference/profile/reference_srt_profile.md").is_file()
    assert load_reference_profile(paths, config)["source"]["episodes"] == [
        "01", "02", "03", "04", "05", "06", "07", "08",
    ]


def test_missing_profile_does_not_break_local_diagnosis(tmp_path: Path) -> None:
    config = _config()
    paths = build_paths(tmp_path, config)
    diagnosis = run_local_pre_review_diagnosis(
        "09", paths, config,
        segments=[SubtitleSegment(1, "09", 1, 0.0, 1.0, "普通对白", [], {})],
        overwrite=True,
    )
    assert "reference/profile/reference_srt_profile.json" in diagnosis["missing_sources"]
    assert diagnosis["summary"]["reference_profile_loaded"] is False


def test_aliang_hint_uses_reference_profile_evidence(tmp_path: Path) -> None:
    config = _config()
    paths = build_paths(tmp_path, config)
    profile_path = tmp_path / "reference/profile/reference_srt_profile.json"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps({
        "usage": "只作证据",
        "high_frequency_names": [{
            "name": "徐绍良",
            "matched_forms": [{"form": "阿良", "count": 88}],
        }],
        "address_terms": [],
    }, ensure_ascii=False), encoding="utf-8")
    characters_path = tmp_path / "agent/characters_official.json"
    characters_path.parent.mkdir(parents=True, exist_ok=True)
    characters_path.write_text(json.dumps({
        "characters": [{
            "role_simplified": "徐绍良",
            "aliases_simplified": ["阿良"],
        }],
    }, ensure_ascii=False), encoding="utf-8")

    diagnosis = run_local_pre_review_diagnosis(
        "09", paths, config,
        segments=[SubtitleSegment(1, "09", 1, 0.0, 1.0, "阿梁啊", [], {})],
        overwrite=True,
    )
    hint = diagnosis["name_diagnosis"][0]
    assert hint["suggestion"] == "阿良"
    assert "阿良”出现 88 次" in hint["reason"]
    assert "官方角色徐绍良别名" in hint["reason"]
    assert diagnosis["sources"]["reference_profile"] == "reference/profile/reference_srt_profile.json"


def test_v3_python_has_no_v2_runtime_path() -> None:
    root = Path(__file__).resolve().parents[1]
    offenders = []
    forbidden = "VanityFair_srt_" + "v2"
    for path in (root / "vf_srt").rglob("*.py"):
        if forbidden in path.read_text(encoding="utf-8"):
            offenders.append(path.relative_to(root).as_posix())
    assert offenders == []
