from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from vf_srt.core.config import DEFAULT_CONFIG, load_config, parse_episodes
from vf_srt.core.json_utils import read_json, write_json
from vf_srt.core.models import GapProfile, SpeechIsland, SubtitleSegment, Utterance, WordToken
from vf_srt.segmentation.candidates import build_cut_candidates
from vf_srt.segmentation.particles import is_particle_fragment, is_tail_particle
from vf_srt.segmentation.quality import flag_segments
from vf_srt.segmentation.repair import repair_segments
from vf_srt.segmentation.scorer import score_candidate
from vf_srt.segmentation.theme_song import detect_theme_song_matches


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.config = deepcopy(DEFAULT_CONFIG)

    def test_episode_parser(self):
        self.assertEqual(parse_episodes("09-11,13,09"), ["09", "10", "11", "13"])

    def test_atomic_json_roundtrip(self):
        local_temp = Path(__file__).resolve().parents[1] / ".codex_tmp"
        local_temp.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=local_temp) as directory:
            path = Path(directory) / "data.json"
            write_json(path, {"粤": "語"})
            self.assertEqual(read_json(path), {"粤": "語"})

    def test_particle_cut_is_penalized(self):
        words = [
            WordToken("你好", 0, 0.5, 1, 1, "，"),
            WordToken("啊", 0.9, 1.1, 1, 2),
            WordToken("今日", 1.2, 1.6, 1, 3),
        ]
        island = SpeechIsland(1, words, 0, 1.6, "你好，啊今日", "utterance_boundary")
        profile = GapProfile(.35, .5, .9, .2, .3, .4, .5)
        candidate = build_cut_candidates(island, profile, DEFAULT_CONFIG)[0]
        score_candidate(candidate, DEFAULT_CONFIG, profile)
        self.assertLess(candidate.score, 0)
        self.assertTrue(is_tail_particle("啊"))
        self.assertTrue(is_particle_fragment("啊，"))

    def test_repair_prevents_overlap(self):
        segments = [
            SubtitleSegment(0, "09", 1, 0, .3, "啊", [], {"natural_end": .3}),
            SubtitleSegment(0, "09", 1, .5, 1.8, "你去边度", [], {"natural_end": 1.8}),
        ]
        repaired = repair_segments(segments, DEFAULT_CONFIG)
        self.assertEqual(len(repaired), 1)
        self.assertEqual(repaired[0].raw_text, "啊你去边度")
        self.assertGreater(repaired[0].end, repaired[0].start)

    def test_theme_song_prefix_match_does_not_require_full_song(self):
        lyrics = [
            {"index": 1, "simplified": "他也在找我也在找找到名利几多"},
            {"index": 2, "simplified": "他拼命追我拼命追追到又如何"},
            {"index": 3, "simplified": "追到什么找到什么收到又几多"},
            {"index": 4, "simplified": "得了什么失了什么可有认真算过"},
        ]
        words = [
            WordToken(lyrics[0]["simplified"], 4.0, 8.0, 1, 1),
            WordToken(lyrics[1]["simplified"], 9.0, 13.0, 1, 2),
            WordToken(lyrics[2]["simplified"], 14.0, 18.0, 1, 3),
        ]
        utterances = [Utterance(1, 4.0, 18.0, "".join(word.text for word in words), words)]
        matches = detect_theme_song_matches(utterances, {"lyrics": lyrics}, self.config)
        self.assertEqual([item["lyric_index"] for item in matches], [1, 2, 3])

    def test_theme_song_does_not_match_ordinary_dialogue(self):
        lyrics = [
            {"index": 1, "simplified": "他也在找我也在找找到名利几多"},
            {"index": 2, "simplified": "他拼命追我拼命追追到又如何"},
        ]
        words = [
            WordToken("你今日去边度", 10.0, 11.0, 1, 1),
            WordToken("我返公司做嘢", 11.2, 12.2, 1, 2),
        ]
        utterances = [Utterance(1, 10.0, 12.2, "普通对白", words)]
        self.assertEqual(detect_theme_song_matches(utterances, {"lyrics": lyrics}, self.config), [])

    def test_short_head_interjection_merges_forward(self):
        segments = [
            SubtitleSegment(0, "09", 1, 0.0, 0.4, "嗱。", [], {"natural_end": 0.4}),
            SubtitleSegment(0, "09", 2, 0.45, 2.0, "你哋两位随便倾下咯", [], {"natural_end": 2.0}),
        ]
        repaired = repair_segments(segments, self.config)
        self.assertEqual(len(repaired), 1)
        self.assertEqual(repaired[0].raw_text, "嗱。你哋两位随便倾下咯")
        self.assertTrue(repaired[0].debug.get("merged_short_segment"))

    def test_short_segment_is_not_merged_across_large_gap(self):
        segments = [
            SubtitleSegment(0, "09", 1, 0.0, 0.5, "拜拜", [], {"natural_end": 0.5}),
            SubtitleSegment(0, "09", 1, 2.0, 3.0, "快啲啊嗱", [], {"natural_end": 3.0}),
        ]
        repaired = repair_segments(segments, self.config)
        self.assertEqual(len(repaired), 2)

    def test_short_reaction_and_interjection_flags(self):
        segments = [
            SubtitleSegment(1, "09", 1, 0.0, 0.4, "嗯。", [], {}),
        ]
        flagged = flag_segments(segments, self.config)
        self.assertIn("short_reaction", flagged[0].flags)
        self.assertIn("standalone_interjection", flagged[0].flags)

    def test_theme_song_is_not_flagged_as_possible_over_split(self):
        segments = [
            SubtitleSegment(1, "09", 0, 0.0, 0.4, "嗯", ["theme_song", "fixed_lyric"], {"theme_song": True}),
            SubtitleSegment(2, "09", 1, 0.45, 1.5, "普通对白", [], {}),
        ]
        flagged = flag_segments(segments, self.config)
        self.assertIn("theme_song", flagged[0].flags)
        self.assertNotIn("possible_over_split", flagged[0].flags)

    def test_forced_cut_requires_no_natural_boundary_reason(self):
        forced = SubtitleSegment(
            1, "09", 1, 0.0, 2.0, "这是一条测试字幕", [],
            {"cut_reasons": ["+4 over_hard_chars_before"]},
        )
        natural = SubtitleSegment(
            2, "09", 1, 3.0, 5.0, "这也是测试字幕", [],
            {"cut_reasons": ["+2 over_soft_chars_before", "+1 soft_punctuation"]},
        )
        flagged = flag_segments([forced, natural], self.config)
        self.assertIn("forced_cut", flagged[0].flags)
        self.assertNotIn("forced_cut", flagged[1].flags)


if __name__ == "__main__":
    unittest.main()
