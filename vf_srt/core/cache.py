from pathlib import Path


def use_cache(path: str | Path, overwrite: bool = False) -> bool:
    candidate = Path(path)
    return candidate.is_file() and candidate.stat().st_size > 0 and not overwrite
