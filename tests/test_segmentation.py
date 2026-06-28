from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from vf_srt.core.config import DEFAULT_CONFIG, load_config, parse_episodes
from vf_srt.core.json_utils import read_json, write_json
from vf_srt.core.models import GapProfile, SpeechIsland, SubtitleSegment, Utterance, WordToken
from vf_srt.segmentation.candidates import build_cut_candidates
from vf_srt.segmentation.cutter import cut_island_to_segments
from vf_srt.segmentation.particles import is_particle_fragment, is_tail_particle
from vf_srt.segmentation.quality import flag_segments
from vf_srt.segmentation.repair import repair_segments
from vf_srt.segmentation.scorer import score_candidate
from vf_srt.segmentation.theme_song import apply_theme_song_override, detect_theme_song_matches


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.config = deepcopy(DEFAULT_CONFIG)

    @staticmethod
    def _theme_fixture():
        return {"lyrics": [
            {"index": 1, "simplified": "他也在找我也在找找到名利几多"},
            {"index": 2, "simplified": "他拼命追我拼命追追到又如何"},
            {"index": 3, "simplified": "追到什么找到什么收到又几多"},
            {"index": 4, "simplified": "得了什么失了什么可有认真算过"},
            {"index": 5, "simplified": "何必呢何必呢可知一切都会身外过"},
        ]}

    @staticmethod
    def _theme_utterance(lines, start, utterance_index):
        words = []
        cursor = start
        for word_index, line in enumerate(lines, start=1):
            words.append(WordToken(line["simplified"], cursor, cursor + 3.0, utterance_index, word_index))
            cursor += 4.0
        return Utterance(
            utterance_index, words[0].start, words[-1].end,
            "".join(word.text for word in words), words,
        )

    def test_episode_parser(self):
        self.assertEqual(parse_episodes("09-11,13,09"), ["09", "10", "11", "13"])

    def test_legacy_flat_theme_interval_migrates_to_opening(self):
        local_temp = Path(__file__).resolve().parents[1] / ".codex_tmp"
        local_temp.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=local_temp) as directory:
            path = Path(directory) / "legacy.yaml"
            path.write_text(
                "theme_song:\n  search_start_seconds: 5.0\n  search_end_seconds: 90.0\n"
                "  min_matched_lines: 3\n",
                encoding="utf-8",
            )
            config = load_config(path)
        self.assertEqual(config["theme_song"]["opening"]["search_start_seconds"], 5.0)
        self.assertEqual(config["theme_song"]["opening"]["search_end_seconds"], 90.0)
        self.assertEqual(config["theme_song"]["opening"]["min_matched_lines"], 3)

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
        self.assertTrue(all(item["theme_region"] == "opening" for item in matches))

    def test_ending_theme_match_uses_last_seconds(self):
        theme = self._theme_fixture()
        utterance = self._theme_utterance(theme["lyrics"][:4], 2500.0, 1)
        matches = detect_theme_song_matches(
            [utterance], theme, self.config, region="ending", audio_duration=2600.0,
        )
        self.assertEqual(len(matches), 4)
        self.assertTrue(all(item["theme_region"] == "ending" for item in matches))

    def test_ending_theme_no_match_does_not_override_segments(self):
        words = [WordToken("这是普通片尾对白", 2500.0, 2501.0, 1, 1)]
        utterances = [Utterance(1, 2500.0, 2501.0, words[0].text, words)]
        theme = self._theme_fixture()
        self.assertEqual(
            detect_theme_song_matches(
                utterances, theme, self.config, region="ending", audio_duration=2600.0,
            ),
            [],
        )
        segments = [SubtitleSegment(1, "12", 1, 2499.5, 2501.5, "这是普通片尾对白", [], {})]
        local_temp = Path(__file__).resolve().parents[1] / ".codex_tmp"
        local_temp.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=local_temp) as directory:
            root = Path(directory)
            write_json(root / "agent" / "theme_song.json", theme)
            paths = SimpleNamespace(root=root, doubao_cache_dir=root / "cache" / "doubao")
            result = apply_theme_song_override("12", segments, utterances, paths, self.config)
        self.assertEqual(result, segments)

    def test_ending_partial_match_below_minimum_is_ignored(self):
        theme = self._theme_fixture()
        utterance = self._theme_utterance(theme["lyrics"][:1], 2500.0, 1)
        matches = detect_theme_song_matches(
            [utterance], theme, self.config, region="ending", audio_duration=2600.0,
        )
        self.assertEqual(matches, [])

    def test_opening_and_ending_are_both_overridden(self):
        theme = self._theme_fixture()
        opening = self._theme_utterance(theme["lyrics"][:4], 10.0, 1)
        ending = self._theme_utterance(theme["lyrics"][:3], 2500.0, 2)
        utterances = [opening, ending]
        segments = [
            SubtitleSegment(1, "12", 1, 9.5, 26.0, "开头错误歌词", [], {}),
            SubtitleSegment(2, "12", 2, 2499.5, 2512.0, "片尾错误歌词", [], {}),
        ]
        local_temp = Path(__file__).resolve().parents[1] / ".codex_tmp"
        local_temp.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=local_temp) as directory:
            root = Path(directory)
            write_json(root / "agent" / "theme_song.json", theme)
            paths = SimpleNamespace(root=root, doubao_cache_dir=root / "cache" / "doubao")
            result = apply_theme_song_override("12", segments, utterances, paths, self.config)
        opening_segments = [item for item in result if "theme_opening" in item.flags]
        ending_segments = [item for item in result if "theme_ending" in item.flags]
        self.assertEqual(len(opening_segments), 4)
        self.assertEqual(len(ending_segments), 3)
        self.assertTrue(all(item.debug["theme_region"] == "opening" for item in opening_segments))
        self.assertTrue(all(item.debug["theme_region"] == "ending" for item in ending_segments))

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
        self.assertNotIn("forced_cut", flagged[0].flags)
        self.assertNotIn("pressure_cut", flagged[0].flags)
        self.assertNotIn("hard_forced_cut", flagged[0].flags)
        self.assertNotIn("bad_forced_cut", flagged[0].flags)

    def test_soft_limit_lookahead_finds_later_natural_cut(self):
        words = []
        start = 0.0
        for index in range(26):
            end = start + 0.1
            words.append(WordToken(
                "字", start, end, 1, index + 1,
                "，" if index == 21 else "",
            ))
            start = end + (0.4 if index == 21 else 0.05)
        island = SpeechIsland(1, words, words[0].start, words[-1].end, "字" * 26, "utterance_boundary")
        profile = GapProfile(.35, .5, .9, .2, .3, .4, .5)
        candidates = build_cut_candidates(island, profile, self.config)
        for candidate in candidates:
            score_candidate(candidate, self.config, profile)
        segments = cut_island_to_segments(island, candidates, self.config, profile, "09")
        self.assertEqual(segments[0].debug["word_end_pos"], 21)
        self.assertEqual(segments[0].debug["cut_type"], "pressure_cut")
        self.assertTrue(segments[0].debug["cut_has_natural_reason"])
        self.assertNotEqual(segments[0].debug["cut_type"], "bad_forced_cut")

    def test_soft_pressure_decision_is_not_mislabeled_natural(self):
        words = [
            WordToken("字", index * 0.15, index * 0.15 + 0.1, 1, index + 1)
            for index in range(26)
        ]
        island = SpeechIsland(1, words, words[0].start, words[-1].end, "字" * 26, "utterance_boundary")
        profile = GapProfile(.35, .5, .9, .2, .3, .4, .5)
        candidates = build_cut_candidates(island, profile, self.config)
        for candidate in candidates:
            score_candidate(candidate, self.config, profile)
        first = cut_island_to_segments(island, candidates, self.config, profile, "09")[0]
        self.assertEqual(first.debug["cut_type"], "pressure_cut")
        self.assertTrue(first.debug["cut_pressure_reasons"])

    def test_pressure_cut_classification(self):
        segment = SubtitleSegment(
            1, "09", 1, 0.0, 4.0, "字" * 21, [],
            {
                "cut_reasons": ["+2 soft_gap", "over_soft_chars_before"],
                "cut_pressure_reasons": ["over_soft_chars_before"],
                "cut_has_natural_reason": True,
            },
        )
        flagged = flag_segments([segment], self.config)[0]
        self.assertIn("forced_cut", flagged.flags)
        self.assertIn("pressure_cut", flagged.flags)
        self.assertNotIn("bad_forced_cut", flagged.flags)

    def test_hard_forced_cut_with_natural_reason(self):
        segment = SubtitleSegment(
            1, "09", 1, 0.0, 6.0, "字" * 25, [],
            {
                "cut_reasons": ["+2 soft_gap", "over_hard_chars_before"],
                "cut_pressure_reasons": ["over_hard_chars_before"],
                "cut_has_natural_reason": True,
                "cut_hard_limit_reached": True,
            },
        )
        flagged = flag_segments([segment], self.config)[0]
        self.assertIn("forced_cut", flagged.flags)
        self.assertIn("hard_forced_cut", flagged.flags)
        self.assertNotIn("bad_forced_cut", flagged.flags)

    def test_bad_forced_cut_without_natural_reason(self):
        segment = SubtitleSegment(
            1, "09", 1, 0.0, 6.0, "字" * 25, [],
            {
                "cut_reasons": ["over_hard_chars_before"],
                "cut_pressure_reasons": ["over_hard_chars_before"],
                "cut_has_natural_reason": False,
                "cut_hard_limit_reached": True,
            },
        )
        flagged = flag_segments([segment], self.config)[0]
        self.assertIn("forced_cut", flagged.flags)
        self.assertIn("bad_forced_cut", flagged.flags)
        self.assertNotIn("hard_forced_cut", flagged.flags)


if __name__ == "__main__":
    unittest.main()
