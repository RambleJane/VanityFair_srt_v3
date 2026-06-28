from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    input_dir: Path
    doubao_cache_dir: Path
    normalized_cache_dir: Path
    segments_cache_dir: Path
    reports_cache_dir: Path
    lab_dir: Path
    review_dir: Path
    output_dir: Path
    logs_dir: Path
    agent_dir: Path
    reference_dir: Path
    reference_profile_dir: Path
    reference_simplified_human_dir: Path
    local_diagnosis_cache_dir: Path
    local_review_cache_dir: Path
    pre_review_diagnosis_cache_dir: Path
    yue_draft_cache_dir: Path
    yue_master_cache_dir: Path
    traditional_context_cache_dir: Path
    traditional_viewer_cache_dir: Path
    simplified_context_cache_dir: Path
    simplified_viewer_cache_dir: Path
    viewer_master_cache_dir: Path
    reference_profile_cache_dir: Path

    def ensure_directories(self) -> None:
        for path in (
            self.input_dir, self.doubao_cache_dir, self.normalized_cache_dir,
            self.segments_cache_dir, self.reports_cache_dir, self.lab_dir,
            self.review_dir, self.output_dir, self.logs_dir,
            self.agent_dir, self.reference_dir, self.reference_profile_dir,
            self.reference_simplified_human_dir, self.local_diagnosis_cache_dir,
            self.local_review_cache_dir, self.pre_review_diagnosis_cache_dir,
            self.yue_draft_cache_dir, self.yue_master_cache_dir,
            self.traditional_context_cache_dir, self.traditional_viewer_cache_dir,
            self.simplified_context_cache_dir, self.simplified_viewer_cache_dir,
            self.viewer_master_cache_dir, self.reference_profile_cache_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def build_paths(project_root: str | Path, config: dict[str, Any] | None = None) -> ProjectPaths:
    root = Path(project_root).resolve()
    paths = (config or {}).get("paths", {})
    def pick(name: str, default: str) -> Path:
        value = Path(paths.get(name, default))
        return value if value.is_absolute() else root / value
    result = ProjectPaths(
        root=root,
        input_dir=pick("input", "input"),
        doubao_cache_dir=pick("doubao_cache", "cache/doubao"),
        normalized_cache_dir=pick("normalized_cache", "cache/normalized"),
        segments_cache_dir=pick("segments_cache", "cache/segments"),
        reports_cache_dir=pick("reports_cache", "cache/reports"),
        lab_dir=pick("lab", "lab"),
        review_dir=pick("review", "review"),
        output_dir=pick("output", "output"),
        logs_dir=pick("logs", "logs"),
        agent_dir=pick("agent", "agent"),
        reference_dir=pick("reference", "reference"),
        reference_profile_dir=pick("reference_profile", "reference/profile"),
        reference_simplified_human_dir=pick("reference_simplified_human", "reference/simplified_human"),
        local_diagnosis_cache_dir=pick("local_diagnosis_cache", "cache/local_diagnosis"),
        local_review_cache_dir=pick("local_review_cache", "cache/local_review"),
        pre_review_diagnosis_cache_dir=pick(
            "pre_review_diagnosis_cache", "cache/pre_review_diagnosis"
        ),
        yue_draft_cache_dir=pick("yue_draft_cache", "cache/yue_draft"),
        yue_master_cache_dir=pick("yue_master_cache", "cache/yue_master"),
        traditional_context_cache_dir=pick(
            "traditional_context_cache", "cache/traditional_context"
        ),
        traditional_viewer_cache_dir=pick(
            "traditional_viewer_cache", "cache/traditional_viewer"
        ),
        simplified_context_cache_dir=pick(
            "simplified_context_cache", "cache/simplified_context"
        ),
        simplified_viewer_cache_dir=pick(
            "simplified_viewer_cache", "cache/simplified_viewer"
        ),
        viewer_master_cache_dir=pick("viewer_master_cache", "cache/viewer_master"),
        reference_profile_cache_dir=pick("reference_profile_cache", "cache/reference_profile"),
    )
    result.ensure_directories()
    return result
