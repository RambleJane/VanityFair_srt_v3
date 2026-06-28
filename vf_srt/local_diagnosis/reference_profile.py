from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..core.cache import use_cache
from ..core.json_utils import read_json, write_json
from ..knowledge.reference_profile import load_reference_profile, parse_srt_text_lines


_DEFAULT_TERMS = ("游埠", "五柳云吞", "打麻雀", "走片", "大蓉", "鹼水鮮", "蝦餃鮮")
_ADDRESS_TERMS = ("哥", "姐", "叔", "伯", "爷", "师傅", "先生", "小姐", "老板", "老细")


def _character_forms(paths: Any) -> set[str]:
    source = paths.agent_dir / "characters_official.json"
    if not source.is_file():
        return set()
    data = read_json(source)
    forms: set[str] = set()
    for item in data.get("characters", []) if isinstance(data, dict) else []:
        for key in ("role_raw", "role_traditional", "role_simplified"):
            value = str(item.get(key) or "").strip()
            if value:
                forms.add(value)
        for key in ("aliases_raw", "aliases_traditional", "aliases_simplified"):
            forms.update(str(value).strip() for value in item.get(key, []) or [] if str(value).strip())
    return forms


def _compact_from_full(profile: dict[str, Any]) -> dict[str, Any]:
    names: dict[str, int] = {}
    for item in profile.get("high_frequency_names", []):
        for form in item.get("matched_forms", []):
            names[str(form.get("form", ""))] = int(form.get("count", 0))
    addresses = {term: 0 for term in _ADDRESS_TERMS}
    for item in profile.get("address_terms", []):
        term = str(item.get("term", ""))
        for suffix in addresses:
            if term.endswith(suffix):
                addresses[suffix] += int(item.get("count", 0))
    return {
        "source_episodes": list(profile.get("source", {}).get("episodes", [])),
        "names": names,
        "address_terms": addresses,
        "terms": {},
        "usage": profile.get("usage", ""),
    }


def build_reference_profile(paths: Any, config: dict[str, Any]) -> dict[str, Any]:
    episodes = [str(value).zfill(2) for value in config.get("reference_profile", {}).get(
        "source_episodes", ["01", "02", "03", "04", "05", "06", "07", "08"]
    )]
    lines: list[str] = []
    found_episodes: list[str] = []
    for episode in episodes:
        source = paths.reference_simplified_human_dir / f"{episode}.srt"
        if not source.is_file():
            continue
        found_episodes.append(episode)
        lines.extend(str(item["text"]) for item in parse_srt_text_lines(source))
    if not lines:
        full = load_reference_profile(paths, config)
        compact = _compact_from_full(full) if full else {
            "source_episodes": [], "names": {}, "address_terms": {}, "terms": {},
        }
        write_json(paths.reference_profile_cache_dir / "01_08_reference_profile.json", compact)
        return compact

    text = "\n".join(lines)
    forms = _character_forms(paths)
    forms.update(re.findall(r"阿[\u4e00-\u9fff]", text))
    names = {form: text.count(form) for form in sorted(forms) if len(form) >= 2 and text.count(form)}
    profile = {
        "source_episodes": found_episodes,
        "names": dict(sorted(names.items(), key=lambda item: (-item[1], item[0]))),
        "address_terms": {term: text.count(term) for term in _ADDRESS_TERMS},
        "terms": {term: text.count(term) for term in _DEFAULT_TERMS if text.count(term)},
        "usage": "只保存前8集人工字幕统计证据，不包含完整字幕，不得用于补写当前集对白。",
    }
    write_json(paths.reference_profile_cache_dir / "01_08_reference_profile.json", profile)
    return profile


def load_or_build_reference_profile(
    paths: Any, config: dict[str, Any], overwrite: bool = False,
) -> dict[str, Any]:
    target = paths.reference_profile_cache_dir / "01_08_reference_profile.json"
    if use_cache(target, overwrite):
        data = read_json(target)
        return data if isinstance(data, dict) else {}
    return build_reference_profile(paths, config)
