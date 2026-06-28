from __future__ import annotations

from pathlib import Path

from vf_srt.core.config import load_config, resolve_stage_llm
from vf_srt.core.paths import build_paths

ROOT = Path(__file__).resolve().parents[1]


def test_example_config_loads_with_v3_structure() -> None:
    config = load_config(str(ROOT / "config.example.yaml"))
    # New stage cache paths are configured.
    for key in ("yue_draft_cache", "yue_master_cache", "traditional_context_cache",
                "traditional_viewer_cache", "simplified_context_cache",
                "simplified_viewer_cache", "viewer_master_cache"):
        assert key in config["paths"]
    # Doubao audio-upload params are retained (env-name based).
    doubao = config["doubao"]
    for key in ("audio_url_template", "app_id_env", "access_token_env",
                "api_key_env", "resource_id", "language", "poll_interval_seconds"):
        assert key in doubao
    # DeepSeek uses a stages structure.
    assert set(config["deepseek"]["stages"]) >= {
        "pre_review_diagnosis", "yue_draft_auto_lines",
        "traditional_viewer_lines", "simplified_viewer_lines",
    }


def test_example_config_has_no_real_keys() -> None:
    text = (ROOT / "config.example.yaml").read_text(encoding="utf-8-sig")
    # Only env-var names, never inline secrets.
    assert "api_key_env" in text
    assert "sk-" not in text  # no DeepSeek/OpenAI-style key
    assert "api_key:" not in text  # no inline api_key field


def test_build_paths_creates_new_stage_dirs(tmp_path: Path) -> None:
    config = load_config(str(ROOT / "config.example.yaml"))
    paths = build_paths(tmp_path, config)
    for attr in ("yue_draft_cache_dir", "yue_master_cache_dir",
                 "traditional_context_cache_dir", "traditional_viewer_cache_dir",
                 "simplified_context_cache_dir", "simplified_viewer_cache_dir",
                 "viewer_master_cache_dir"):
        directory = getattr(paths, attr)
        assert directory.is_dir()


def test_resolve_stage_llm_inherits_then_overrides() -> None:
    config = load_config(str(ROOT / "config.example.yaml"))
    resolved = resolve_stage_llm(config, "yue_draft_auto_lines")
    # Inherited defaults
    assert resolved["timeout"] == 120
    assert resolved["window_before"] == 3
    # Stage overrides
    assert resolved["batch_size"] == 30
    assert resolved["max_tokens"] == 6000
    assert resolved["thinking"] == "enabled"

    pre_review = resolve_stage_llm(config, "pre_review_diagnosis")
    assert pre_review["parse_retry_attempts"] == 2


def test_resolve_stage_llm_unknown_stage_returns_defaults_only() -> None:
    config = load_config(str(ROOT / "config.example.yaml"))
    resolved = resolve_stage_llm(config, "does_not_exist")
    assert resolved["model"] == "deepseek-chat"
    assert "batch_size" not in resolved  # no stage overrides applied


def _write(tmp_path: Path, body: str) -> str:
    path = tmp_path / "legacy.yaml"
    path.write_text(body, encoding="utf-8")
    return str(path)


def test_legacy_v2_deepseek_keys_normalize_into_stages(tmp_path: Path) -> None:
    config_path = _write(tmp_path, "\n".join([
        "deepseek:",
        '  api_key_env: "DEEPSEEK_API_KEY"',
        "  request_timeout_seconds: 90",
        '  diagnosis_model: "deepseek-v4-flash"',
        '  diagnosis_thinking: "enabled"',
        '  yue_draft_model: "deepseek-v4-flash"',
        "  yue_batch_size: 25",
        '  translation_model: "deepseek-v4-flash"',
        "  translation_batch_size: 20",
        "",
    ]))
    config = load_config(config_path)
    ds = config["deepseek"]

    # request_timeout_seconds -> timeout
    assert ds["timeout"] == 90
    stages = ds["stages"]
    assert stages["pre_review_diagnosis"]["model"] == "deepseek-v4-flash"
    assert stages["pre_review_diagnosis"]["thinking"] == "enabled"
    assert stages["yue_draft_auto_lines"]["model"] == "deepseek-v4-flash"
    assert stages["yue_draft_auto_lines"]["batch_size"] == 25
    # translation_* fans out to both viewer stages
    assert stages["traditional_viewer_lines"]["batch_size"] == 20
    assert stages["simplified_viewer_lines"]["batch_size"] == 20


def test_v3_structure_is_not_overwritten_by_legacy_mapping(tmp_path: Path) -> None:
    # When both an explicit stages override and a legacy flat key are present,
    # the explicit v3 value wins.
    config_path = _write(tmp_path, "\n".join([
        "deepseek:",
        "  yue_batch_size: 25",
        "  stages:",
        "    yue_draft_auto_lines:",
        "      batch_size: 99",
        "",
    ]))
    config = load_config(config_path)
    assert config["deepseek"]["stages"]["yue_draft_auto_lines"]["batch_size"] == 99


def test_deepseek_client_reads_top_level_without_stage(monkeypatch) -> None:
    from vf_srt.llm.deepseek import DeepSeekClient
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-xyz")
    config = load_config(str(ROOT / "config.example.yaml"))
    client = DeepSeekClient.from_config(config, transport=lambda *a: (200, "{}"))
    assert client.model == "deepseek-chat"
    assert client.base_url == "https://api.deepseek.com"
    assert client.timeout == 120


def test_deepseek_client_applies_pre_review_stage_overrides(monkeypatch) -> None:
    from vf_srt.llm.deepseek import DeepSeekClient
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-xyz")
    config = load_config(str(ROOT / "config.example.yaml"))
    client = DeepSeekClient.from_config(
        config, stage="pre_review_diagnosis", transport=lambda *a: (200, "{}")
    )
    resolved = resolve_stage_llm(config, "pre_review_diagnosis")
    assert client.model == resolved["model"]
    assert client.thinking == resolved["thinking"]
    assert client.max_tokens == resolved["max_tokens"]
