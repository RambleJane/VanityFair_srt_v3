from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..core.json_utils import read_json


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig") if path.is_file() else ""


def _markdown_terms(text: str) -> set[str]:
    terms = set(re.findall(r"\*\*([^*]{1,24})\*\*", text))
    for line in text.splitlines():
        match = re.match(r"\s*-\s+([^：:（(]{1,24})", line)
        if match:
            terms.update(part.strip() for part in re.split(r"\s*/\s*|\s+或\s+", match.group(1)) if part.strip())
    return terms


def _characters_from_json(data: dict[str, Any]) -> dict[str, Any]:
    official_names: set[str] = set()
    aliases: set[str] = set()
    alias_to_name: dict[str, str] = {}
    entries = data.get("characters", []) if isinstance(data, dict) else []
    for item in entries:
        canonical = str(item.get("role_simplified") or item.get("role_raw") or "").strip()
        if canonical:
            official_names.add(canonical)
        for key in ("role_raw", "role_traditional", "role_simplified"):
            value = str(item.get(key) or "").strip()
            if value:
                official_names.add(value)
                alias_to_name[value] = canonical or value
        for key in ("aliases_raw", "aliases_traditional", "aliases_simplified"):
            for value in item.get(key, []) or []:
                alias = str(value).strip()
                if alias:
                    aliases.add(alias)
                    alias_to_name[alias] = canonical or alias
    return {
        "official_names": sorted(official_names),
        "aliases": sorted(aliases),
        "alias_to_name": alias_to_name,
        "raw_entries": entries,
    }


def _characters_from_text(text: str) -> dict[str, Any]:
    names: set[str] = set()
    aliases: set[str] = set()
    alias_to_name: dict[str, str] = {}
    for line in text.splitlines():
        if "…" not in line:
            continue
        _, role_text = line.split("…", 1)
        parts = [part.strip() for part in role_text.split("/") if part.strip()]
        if not parts:
            continue
        names.add(parts[0])
        for alias in parts[1:]:
            aliases.add(alias)
            alias_to_name[alias] = parts[0]
    return {"official_names": sorted(names), "aliases": sorted(aliases), "alias_to_name": alias_to_name, "raw_entries": []}


def load_local_knowledge(paths: Any, config: dict[str, Any]) -> dict[str, Any]:
    del config
    files = {
        "characters_json": paths.agent_dir / "characters_official.json",
        "characters_txt": paths.agent_dir / "characters_official.txt",
        "characters_md": paths.agent_dir / "02_characters_official.md",
        "theme_song": paths.agent_dir / "theme_song.json",
        "glossary_confirmed": paths.agent_dir / "03_glossary_confirmed.md",
        "glossary_uncertain": paths.agent_dir / "04_glossary_uncertain.md",
    }
    missing = [_relative(path, paths.root) for path in files.values() if not path.is_file()]
    if files["characters_json"].is_file():
        data = read_json(files["characters_json"])
        characters = _characters_from_json(data if isinstance(data, dict) else {})
    else:
        characters = _characters_from_text(
            _read_text(files["characters_txt"]) or _read_text(files["characters_md"])
        )
    confirmed_text = _read_text(files["glossary_confirmed"])
    uncertain_text = _read_text(files["glossary_uncertain"])
    theme = read_json(files["theme_song"]) if files["theme_song"].is_file() else {}
    return {
        "characters": characters,
        "confirmed_glossary_text": confirmed_text,
        "confirmed_terms": sorted(_markdown_terms(confirmed_text)),
        "uncertain_glossary_text": uncertain_text,
        "uncertain_terms": sorted(_markdown_terms(uncertain_text)),
        "theme_song": theme if isinstance(theme, dict) else {},
        "sources": {
            "characters": [
                _relative(files["characters_json"], paths.root),
                _relative(files["characters_md"], paths.root),
            ],
            "theme_song": _relative(files["theme_song"], paths.root),
            "glossary_confirmed": _relative(files["glossary_confirmed"], paths.root),
            "glossary_uncertain": _relative(files["glossary_uncertain"], paths.root),
            "reference_srt": "reference/simplified_human/01-08.srt",
        },
        "missing_sources": missing,
    }
