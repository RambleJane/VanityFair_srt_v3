from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any


def export_debug_xlsx(episode: str, paths: Any) -> Path | None:
    """Export debug CSV/JSON views to XLSX through the optional artifact-tool runtime."""
    node = os.environ.get("VF_SRT_NODE")
    node_modules = os.environ.get("VF_SRT_NODE_MODULES")
    if not node or not node_modules or not Path(node).is_file() or not Path(node_modules).is_dir():
        return None
    script = Path(__file__).with_name("export_debug_xlsx.mjs")
    output = paths.lab_dir / f"{episode}_segmentation_debug.xlsx"
    environment = {**os.environ, "NODE_PATH": node_modules}
    subprocess.run(
        [node, str(script), episode, str(paths.lab_dir), str(paths.reports_cache_dir), str(output)],
        check=True, env=environment, capture_output=True, text=True,
    )
    return output
