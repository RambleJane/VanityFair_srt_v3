from __future__ import annotations

from typing import Any

from ..local_diagnosis.name_rules import diagnose_character_names
from ..local_diagnosis.term_rules import diagnose_terms
from ..local_diagnosis.theme_rules import diagnose_theme_song
from ..local_diagnosis.unknown_names import detect_unknown_name_candidates


def collect_local_review_hints(
    segments: list[Any],
    knowledge: dict[str, Any],
    reference_profile: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run the existing local rule engine and return non-applying hints."""
    hints = [
        *diagnose_character_names(segments, knowledge, reference_profile, config),
        *diagnose_terms(segments, knowledge, reference_profile, config),
        *diagnose_theme_song(segments, knowledge, config),
        *detect_unknown_name_candidates(segments, knowledge, reference_profile, config),
    ]
    return [{**hint, "do_not_auto_apply": True} for hint in hints]
