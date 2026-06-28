from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..core.json_utils import read_json, write_json


ADDRESS_SUFFIXES = (
    "哥",
    "姐",
    "叔",
    "伯",
    "爷",
    "妈",
    "父",
    "sir",
    "Sir",
    "先生",
    "小姐",
    "老板",
    "老细",
    "师傅",
    "导演",
    "经理",
    "秘书",
)

STYLE_BUCKETS = {
    "greeting": ("喂", "早", "再见", "恭喜"),
    "request": ("请", "帮", "麻烦", "拜托", "快点"),
    "refusal": ("不用", "不要", "不行", "没办法", "不可以"),
    "question": ("吗", "呢", "什么", "怎么", "是不是", "为什么", "？"),
    "emotion": ("唉", "哎呀", "糟了", "够了", "开玩笑"),
    "period_domain": ("戏", "戏院", "后台", "台期", "电影", "公司", "茶", "麻将", "警察"),
}

STOP_PHRASES = {
    "知道了",
    "是啊",
    "好",
    "没有",
    "不是",
    "什么",
    "为什么",
    "怎么",
}

TRAILING_PARTICLES = "啊呀啦喇呢嘛吗哦喂"


def _profile_settings(config: dict[str, Any] | None) -> dict[str, Any]:
    return (config or {}).get("reference_profile", {})


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _configured_paths(paths: Any, config: dict[str, Any] | None) -> tuple[Path, Path, Path]:
    settings = _profile_settings(config)
    return (
        _resolve(paths.root, settings.get("json_path", "reference/profile/reference_srt_profile.json")),
        _resolve(paths.root, settings.get("markdown_path", "reference/profile/reference_srt_profile.md")),
        _resolve(paths.root, settings.get("source_srt_dir", "reference/simplified_human")),
    )


def _source_episodes(config: dict[str, Any] | None) -> list[str]:
    configured = _profile_settings(config).get(
        "source_episodes", ["01", "02", "03", "04", "05", "06", "07", "08"]
    )
    if isinstance(configured, str):
        configured = [item.strip() for item in configured.split(",") if item.strip()]
    return [str(item).zfill(2) for item in configured]


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def parse_srt_text_lines(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n"))
    rows: list[dict[str, Any]] = []
    for block in blocks:
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line or line.isdigit() or "-->" in line:
                continue
            rows.append({"index": len(rows) + 1, "text": line})
    return rows


def build_reference_profile(paths: Any, config: dict[str, Any] | None = None) -> dict[str, Any]:
    character_data = load_character_data(paths)
    known_names = build_known_name_map(character_data)
    lines = load_reference_lines(paths, config)
    _, _, source_dir = _configured_paths(paths, config)

    name_hits = collect_name_hits(lines, known_names)
    address_terms = collect_address_terms(lines, known_names)
    fixed_phrases = collect_fixed_phrases(lines)
    suspicious_terms = collect_suspicious_terms(lines, known_names)
    style_samples = collect_style_samples(lines)
    storyline_terms = collect_storyline_terms(paths, known_names)

    return {
        "source": {
            "reference_dir": _relative_to_root(source_dir, paths.root),
            "episodes": sorted({item["episode"] for item in lines}),
            "line_count": len(lines),
            "character_source": _relative_to_root(paths.agent_dir / "characters_official.json", paths.root),
            "story_sources": [
                _relative_to_root(paths.agent_dir / "05_story_outline_authoritative.md", paths.root),
                _relative_to_root(paths.agent_dir / "06_story_clues_verified_names_uncertain.md", paths.root),
            ],
        },
        "usage": (
            "离线从01-08人工精校简体SRT、官方演员表和剧情资料提取。"
            "用于专名/称谓/风格参考，不得用来补写当前集不存在的对白。"
        ),
        "high_frequency_names": name_hits,
        "address_terms": address_terms,
        "relationship_address_examples": collect_relationship_examples(lines, known_names),
        "fixed_translation_phrases": fixed_phrases,
        "suspicious_proper_terms": suspicious_terms,
        "storyline_terms": storyline_terms,
        "style_samples": style_samples,
    }


def write_reference_profile(
    paths: Any, config: dict[str, Any] | None = None, overwrite: bool = False,
) -> dict[str, Any]:
    json_path, md_path, source_dir = _configured_paths(paths, config)
    if json_path.is_file() and not overwrite:
        return read_json(json_path)
    missing = [source_dir / f"{episode}.srt" for episode in _source_episodes(config)]
    missing = [path for path in missing if not path.is_file()]
    if missing:
        names = ", ".join(path.name for path in missing)
        raise FileNotFoundError(
            f"缺少参考字幕：{names}。可以继续使用已迁移的 {json_path}; "
            f"如需重建，请把 01–08 SRT 放到 {source_dir}。"
        )
    profile = build_reference_profile(paths, config)
    write_json(json_path, profile)
    write_reference_profile_markdown(profile, md_path)
    return profile


def load_reference_profile(paths: Any, config: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = _profile_settings(config)
    if not bool(settings.get("enabled", True)):
        return {}
    json_path, _, _ = _configured_paths(paths, config)
    if json_path.is_file():
        loaded = read_json(json_path)
        return loaded if isinstance(loaded, dict) else {}
    if bool(settings.get("rebuild_from_srt_if_missing", False)):
        return write_reference_profile(paths, config, overwrite=True)
    return {}


def load_character_data(paths: Any) -> dict[str, Any]:
    path = paths.agent_dir / "characters_official.json"
    if not path.exists():
        return {"characters": []}
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def build_known_name_map(character_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    known: dict[str, dict[str, Any]] = {}
    for item in character_data.get("characters", []) or []:
        if not isinstance(item, dict):
            continue
        canonical = str(item.get("role_simplified") or item.get("role_raw") or "").strip()
        actor = str(item.get("actor_simplified") or item.get("actor_raw") or "").strip()
        aliases: list[str] = []
        for key in (
            "role_raw",
            "role_traditional",
            "role_simplified",
            "actor_raw",
            "actor_traditional",
            "actor_simplified",
        ):
            value = str(item.get(key) or "").strip()
            if value:
                aliases.append(value)
        for key in ("aliases_raw", "aliases_traditional", "aliases_simplified"):
            values = item.get(key) or []
            if isinstance(values, list):
                aliases.extend(str(value).strip() for value in values if str(value).strip())
        for name in sorted(set(aliases), key=len, reverse=True):
            if len(name) < 2:
                continue
            known[name] = {"canonical": canonical or name, "actor": actor, "matched_form": name}
    return known


def load_reference_lines(paths: Any, config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    _, _, source_dir = _configured_paths(paths, config)
    allowed = set(_source_episodes(config))
    for path in sorted(source_dir.glob("*.srt")):
        if path.stem not in allowed:
            continue
        episode = path.stem
        for item in parse_srt_text_lines(path):
            clean = normalize_space(str(item["text"]))
            if clean:
                rows.append({"episode": episode, "index": int(item["index"]), "text": clean})
    return rows


def collect_name_hits(lines: list[dict[str, Any]], known_names: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in lines:
        text = row["text"]
        for name, meta in known_names.items():
            if name not in text:
                continue
            key = meta["canonical"]
            item = grouped.setdefault(
                key,
                {
                    "name": key,
                    "actor": meta.get("actor", ""),
                    "count": 0,
                    "matched_forms": Counter(),
                    "examples": [],
                },
            )
            item["count"] += text.count(name)
            item["matched_forms"][name] += text.count(name)
            append_example(item["examples"], row)

    output = []
    for item in grouped.values():
        item["matched_forms"] = [
            {"form": form, "count": count} for form, count in item["matched_forms"].most_common(8)
        ]
        output.append(item)
    return sorted(output, key=lambda item: (-item["count"], item["name"]))[:60]


def collect_address_terms(lines: list[dict[str, Any]], known_names: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    known_forms = set(known_names)
    pattern = re.compile(r"[\u4e00-\u9fffA-Za-z]{1,4}(?:" + "|".join(map(re.escape, ADDRESS_SUFFIXES)) + r")")
    for row in lines:
        for term in pattern.findall(row["text"]):
            term = normalize_candidate_term(term)
            if len(term) < 2:
                continue
            if term in STOP_PHRASES:
                continue
            counter[term] += 1
            append_example(examples[term], row)
    for form in known_forms:
        if form.startswith("阿") or any(form.endswith(suffix) for suffix in ADDRESS_SUFFIXES):
            for row in lines:
                if form in row["text"]:
                    counter[form] += row["text"].count(form)
                    append_example(examples[form], row)

    return [
        {"term": term, "count": count, "known_character": term in known_forms, "examples": examples[term]}
        for term, count in counter.most_common(80)
        if count >= 1
    ]


def collect_relationship_examples(lines: list[dict[str, Any]], known_names: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    important = {name for name in known_names if len(name) >= 2}
    for row in lines:
        text = row["text"]
        hits = [name for name in important if name in text]
        if not hits:
            continue
        if not any(marker in text for marker in ("哥", "姐", "叔", "伯", "妈", "先生", "小姐", "阿")):
            continue
        output.append(
            {
                "episode": row["episode"],
                "index": row["index"],
                "mentions": sorted(set(hits), key=len, reverse=True)[:4],
                "text": text,
            }
        )
        if len(output) >= 80:
            break
    return output


def collect_fixed_phrases(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in lines:
        for phrase in line_phrases(row["text"]):
            if phrase in STOP_PHRASES:
                continue
            counter[phrase] += 1
            append_example(examples[phrase], row)
    return [
        {"phrase": phrase, "count": count, "examples": examples[phrase]}
        for phrase, count in counter.most_common(100)
        if count >= 2 and useful_phrase(phrase)
    ][:60]


def collect_suspicious_terms(lines: list[dict[str, Any]], known_names: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    known_forms = set(known_names)
    counter: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    suffixes = ("公司", "戏院", "茶楼", "银行", "学校", "训练班", "导演", "老板", "经理", "小姐", "先生")
    pattern = re.compile(r"[\u4e00-\u9fffA-Za-z]{2,8}")
    for row in lines:
        text = row["text"]
        for term in pattern.findall(text):
            term = normalize_candidate_term(term)
            if term in known_forms or term in STOP_PHRASES:
                continue
            if any(term.endswith(suffix) for suffix in suffixes) or looks_like_name(term):
                counter[term] += 1
                append_example(examples[term], row)
    return [
        {"term": term, "count": count, "reason": guess_term_reason(term), "examples": examples[term]}
        for term, count in counter.most_common(80)
        if count >= 1
    ][:50]


def collect_style_samples(lines: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {key: [] for key in STYLE_BUCKETS}
    for row in lines:
        text = row["text"]
        if not 2 <= len(text) <= 26:
            continue
        for bucket, markers in STYLE_BUCKETS.items():
            if any(marker in text for marker in markers):
                append_example(buckets[bucket], row, limit=12)
    return buckets


def collect_storyline_terms(paths: Any, known_names: dict[str, dict[str, Any]]) -> dict[str, Any]:
    story_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            paths.agent_dir / "05_story_outline_authoritative.md",
            paths.agent_dir / "06_story_clues_verified_names_uncertain.md",
        )
        if path.exists()
    )
    if not story_text:
        return {"known_character_mentions": [], "plot_terms": []}

    name_counter: Counter[str] = Counter()
    matched_forms: dict[str, Counter[str]] = defaultdict(Counter)
    for form, meta in known_names.items():
        count = story_text.count(form)
        if not count:
            continue
        canonical = meta["canonical"]
        name_counter[canonical] += count
        matched_forms[canonical][form] += count

    plot_counter: Counter[str] = Counter()
    plot_patterns = (
        r"[\u4e00-\u9fffA-Za-z]{2,12}(?:公司|戏院|银行|训练班|董事局|地产|珠宝|外围马|遗嘱|警局|船期)",
        r"(?:孔氏电影公司|金娃|东非|巴拿马|天香楼|安平戏院|民间艺术欣赏会)",
    )
    for pattern in plot_patterns:
        for term in re.findall(pattern, story_text):
            term = normalize_candidate_term(term)
            if term and term not in STOP_PHRASES:
                plot_counter[term] += 1

    return {
        "known_character_mentions": [
            {
                "name": name,
                "count": count,
                "matched_forms": [
                    {"form": form, "count": form_count}
                    for form, form_count in matched_forms[name].most_common(6)
                ],
            }
            for name, count in name_counter.most_common(40)
        ],
        "plot_terms": [
            {"term": term, "count": count} for term, count in plot_counter.most_common(40)
        ],
    }


def render_reference_profile_md(profile: dict[str, Any]) -> str:
    lines = [
        "# 前8集人工精校字幕离线提取",
        "",
        profile["usage"],
        "",
        "## 高频人名/称谓",
    ]
    for item in profile["high_frequency_names"][:30]:
        forms = "、".join(f"{x['form']}({x['count']})" for x in item["matched_forms"][:5])
        lines.append(f"- {item['name']}：{item['count']} 次；匹配：{forms}")

    lines.extend(["", "## 角色互称/称谓例句"])
    for item in profile["relationship_address_examples"][:30]:
        mentions = "、".join(item["mentions"])
        lines.append(f"- EP{item['episode']} #{item['index']} [{mentions}] {item['text']}")

    lines.extend(["", "## 固定翻译写法/高频短语"])
    for item in profile["fixed_translation_phrases"][:40]:
        lines.append(f"- {item['phrase']}：{item['count']} 次")

    lines.extend(["", "## 可疑专有词"])
    for item in profile["suspicious_proper_terms"][:35]:
        lines.append(f"- {item['term']}：{item['count']} 次；{item['reason']}")

    story = profile.get("storyline_terms", {}) or {}
    lines.extend(["", "## 剧情简介关键项"])
    for item in story.get("known_character_mentions", [])[:20]:
        forms = "、".join(f"{x['form']}({x['count']})" for x in item.get("matched_forms", [])[:4])
        lines.append(f"- {item['name']}：剧情简介出现 {item['count']} 次；匹配：{forms}")
    for item in story.get("plot_terms", [])[:20]:
        lines.append(f"- {item['term']}：{item['count']} 次")

    lines.extend(["", "## 口语风格样例"])
    for bucket, samples in profile["style_samples"].items():
        lines.append(f"### {bucket}")
        for sample in samples[:8]:
            lines.append(f"- EP{sample['episode']} #{sample['index']} {sample['text']}")

    return "\n".join(lines) + "\n"


def write_reference_profile_markdown(profile: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_reference_profile_md(profile), encoding="utf-8", newline="\n")


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_punctuation(text: str) -> str:
    return re.sub(r"[，。！？、,.!?…：:；;“”\"'（）()《》\s]", "", text)


def line_phrases(text: str) -> list[str]:
    chunks = re.split(r"[，。！？、,.!?…：:；;“”\"'（）()《》\s]+", text)
    output: list[str] = []
    for chunk in chunks:
        phrase = normalize_candidate_term(chunk)
        if 4 <= len(phrase) <= 14:
            output.append(phrase)
    full = normalize_candidate_term(strip_punctuation(text))
    if 4 <= len(full) <= 14:
        output.append(full)
    return sorted(set(output))


def useful_phrase(phrase: str) -> bool:
    if len(phrase) < 4:
        return False
    if re.fullmatch(r"[的是了吗嘛呢啊啦喂哦唉呀]+", phrase):
        return False
    return True


def looks_like_name(term: str) -> bool:
    return bool(re.fullmatch(r"(?:阿[\u4e00-\u9fff]{1,2}|[\u4e00-\u9fff]{1,3}(?:哥|姐|叔|伯|爷|妈|仔))", term))


def normalize_candidate_term(term: str) -> str:
    term = strip_punctuation(term).strip()
    while len(term) > 2 and term[-1] in TRAILING_PARTICLES:
        term = term[:-1]
    return term


def guess_term_reason(term: str) -> str:
    if term.startswith("阿") or term.endswith(("哥", "姐", "叔", "伯", "爷", "妈", "仔")):
        return "疑似人物称谓或昵称，需和演员表/上下文核对"
    if term.endswith(("公司", "戏院", "茶楼", "银行", "学校", "训练班")):
        return "疑似机构、地点或行业词"
    if term.endswith(("导演", "老板", "经理", "小姐", "先生")):
        return "疑似职务/称呼"
    return "疑似专有词或固定说法"


def append_example(examples: list[dict[str, Any]], row: dict[str, Any], limit: int = 3) -> None:
    if len(examples) >= limit:
        return
    examples.append({"episode": row["episode"], "index": row["index"], "text": row["text"]})
