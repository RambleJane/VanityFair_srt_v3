from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any

from ..core.cache import use_cache
from ..core.json_utils import read_json, write_json
from ..core.models import Utterance, WordToken
from .schema import get_result_utterances, validate_doubao_result


PUNCTUATION = set("，。！？；：、,.!?;:…—～~（）()【】[]《》〈〉“”‘’\"'")


def _attach_trailing_punctuation(text: str, words: list[WordToken]) -> None:
    """Align word texts left-to-right and attach intervening punctuation to the prior token."""
    cursor = 0
    for index, word in enumerate(words):
        position = text.find(word.text, cursor)
        if position < 0:
            position = cursor
        word_end = position + len(word.text)
        next_position = len(text)
        if index + 1 < len(words):
            found = text.find(words[index + 1].text, word_end)
            if found >= 0:
                next_position = found
        between = text[word_end:next_position]
        word.trailing_punct = "".join(char for char in between if char in PUNCTUATION)
        cursor = max(word_end, next_position)


def utterance_from_dict(item: dict[str, Any]) -> Utterance:
    words = [WordToken(**word) for word in item.get("words", [])]
    return Utterance(
        index=int(item["index"]), start=float(item["start"]), end=float(item["end"]),
        text=str(item.get("text", "")), words=words,
    )


def normalize_result(episode: str, paths: Any, config: dict[str, Any]) -> list[Utterance]:
    source = paths.doubao_cache_dir / f"{episode}_result.json"
    target = paths.normalized_cache_dir / f"{episode}_utterances.json"
    overwrite = bool(config.get("cache", {}).get("overwrite_existing", False))
    if use_cache(target, overwrite):
        return [utterance_from_dict(item) for item in read_json(target)]
    if not source.is_file():
        raise FileNotFoundError(f"Doubao cache not found: {source}")
    result = read_json(source)
    validate_doubao_result(result)
    normalized: list[Utterance] = []
    for source_index, item in enumerate(get_result_utterances(result), start=1):
        words: list[WordToken] = []
        for source_word_index, raw_word in enumerate(item["words"], start=1):
            word_text = re.sub(r"\s+", " ", str(raw_word["text"])).strip()
            if not word_text:
                continue
            start = float(raw_word["start_time"]) / 1000.0
            end = float(raw_word["end_time"]) / 1000.0
            words.append(WordToken(
                text=word_text, start=start, end=max(start, end),
                source_utterance_index=source_index, word_index=source_word_index,
            ))
        _attach_trailing_punctuation(str(item["text"]), words)
        if words:
            normalized.append(Utterance(
                index=source_index,
                start=float(item["start_time"]) / 1000.0,
                end=float(item["end_time"]) / 1000.0,
                text=str(item["text"]), words=words,
            ))
    write_json(target, [asdict(item) for item in normalized])
    return normalized
