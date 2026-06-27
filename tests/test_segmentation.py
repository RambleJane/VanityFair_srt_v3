from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from vf_srt.core.config import DEFAULT_CONFIG, load_config, parse_episodes
from vf_srt.core.json_utils import read_json, write_json
from vf_srt.core.models import GapProfile, SpeechIsland, SubtitleSegment, WordToken
from vf_srt.segmentation.candidates import build_cut_candidates
from vf_srt.segmentation.particles import is_particle_fragment, is_tail_particle
from vf_srt.segmentation.repair import repair_segments
from vf_srt.segmentation.scorer import score_candidate


class CoreTests(unittest.TestCase):
    def test_episode_parser(self):
        self.assertEqual(parse_episodes("09-11,13,09"), ["09", "10", "11", "13"])

    def test_atomic_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
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


if __name__ == "__main__":
    unittest.main()
