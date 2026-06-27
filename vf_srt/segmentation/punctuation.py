SOFT_PUNCT = set("，,、；;：:…")
STRONG_PUNCT = set("。！？!?")


def is_soft_punct(p: str) -> bool:
    return any(char in SOFT_PUNCT for char in p)


def is_strong_punct(p: str) -> bool:
    return any(char in STRONG_PUNCT for char in p)


def has_punctuation(word: object) -> bool:
    return bool(getattr(word, "trailing_punct", ""))


def has_soft_punct_after(word: object) -> bool:
    return is_soft_punct(getattr(word, "trailing_punct", ""))


def has_strong_punct_after(word: object) -> bool:
    return is_strong_punct(getattr(word, "trailing_punct", ""))
