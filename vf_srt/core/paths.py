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

    def ensure_directories(self) -> None:
        for path in (
            self.input_dir, self.doubao_cache_dir, self.normalized_cache_dir,
            self.segments_cache_dir, self.reports_cache_dir, self.lab_dir,
            self.review_dir, self.output_dir, self.logs_dir,
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
    )
    result.ensure_directories()
    return result
