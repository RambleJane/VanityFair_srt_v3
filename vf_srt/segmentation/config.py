from __future__ import annotations

from typing import Any


def segmentation_config(config: dict[str, Any]) -> dict[str, Any]:
    try:
        return config["segmentation"]
    except KeyError as exc:
        raise ValueError("Missing segmentation configuration") from exc
