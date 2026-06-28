from __future__ import annotations

from typing import Any

from .models import make_hint, segment_to_dict


_THEME_MISHEARS = {
    "拼命醉": "拼命追",
    "深爱过": "身外过",
    "低落": "低堕",
    "低多": "低堕",
    "收到无忧": "收到又几多",
    "认真宣布": "认真算过",
}


def diagnose_theme_song(
    segments: list[Any], knowledge: dict[str, Any], config: dict[str, Any],
) -> list[dict[str, Any]]:
    del knowledge, config
    output: list[dict[str, Any]] = []
    for value in segments:
        segment = segment_to_dict(value)
        flags = set(segment.get("flags", []) or [])
        raw_text = str(segment.get("raw_text", ""))
        if "fixed_lyric" in flags:
            continue
        if "theme_ending_unmatched" in flags:
            output.append(make_hint(
                segment, span=raw_text, suggestion=None, confidence="medium",
                category="theme_song_unmatched",
                reason="该行位于已检测片尾曲区域，但未能高置信匹配固定歌词，需人工检查。",
                source="theme_song",
            ))
        for span, suggestion in _THEME_MISHEARS.items():
            if span not in raw_text:
                continue
            output.append(make_hint(
                segment, span=span, suggestion=suggestion, confidence="high",
                category="theme_song", reason="该片段符合主题曲固定歌词的常见ASR误识。",
                source="theme_song",
            ))
    return output
