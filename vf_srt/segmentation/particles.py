TAIL_PARTICLES = {
    "啊", "呀", "啦", "喇", "咧", "嘞", "咩", "啫", "啰", "囉", "嘛", "吓",
    "噃", "添", "嘅", "㗎", "㖞", "喎", "啩", "啵",
}
HEAD_INTERJECTIONS = {"喂", "诶", "唉", "哦", "嗯", "嗱", "哎", "哎呀", "吓"}


def _clean(text: str) -> str:
    return "".join(char for char in text.strip() if char not in "，。！？；：、,.!?;:…—～~ \t\r\n")


def is_tail_particle(text: str) -> bool:
    return _clean(text) in TAIL_PARTICLES


def is_head_interjection(text: str) -> bool:
    return _clean(text) in HEAD_INTERJECTIONS


def is_particle_fragment(text: str) -> bool:
    cleaned = _clean(text)
    return bool(cleaned) and len(cleaned) <= 2 and all(char in TAIL_PARTICLES for char in cleaned)


def bad_cut_before_next_word(next_word: str) -> bool:
    return is_tail_particle(next_word)


def bad_island_or_segment_text(text: str) -> bool:
    return is_particle_fragment(text)
