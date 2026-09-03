"""CPU contract tests for the D2 contiguous-window diagnosis."""
from __future__ import annotations

import unittest

from experiments.phase2.e3_v2.run_contiguous_window_diagnosis import (
    build_probe_segments,
    summarize_survival,
)


def _record(all_survived: bool, fraction: float) -> dict:
    return {
        "all_answer_tokens_survived": all_survived,
        "any_answer_token_survived": fraction > 0,
        "answer_token_retained_fraction": fraction,
    }


class ContiguousWindowDiagnosisTests(unittest.TestCase):
    def test_probe_segments_cover_partial_tail(self):
        segments = build_probe_segments(10, 4)
        self.assertEqual([(item.start, item.end) for item in segments], [(0, 4), (4, 8), (8, 10)])
        self.assertTrue(segments[-1].partial)
        self.assertTrue(segments[0].protected)
        self.assertTrue(segments[-1].protected)

    def test_continuation_requires_five_complete_and_strict_gain(self):
        rows = []
        for index in range(10):
            rows.append(
                {
                    "survival": {
                        "top_tokens": {
                            "8": _record(False, 0.25),
                            "16": _record(False, 0.5),
                        },
                        "max_mass_window": {
                            "8": _record(index < 4, 0.6),
                            "16": _record(index < 6, 0.8),
                        },
                    }
                }
            )
        result = summarize_survival(rows, (8, 16))
        self.assertEqual(result["selection"]["selected_width"], 16)
        self.assertEqual(result["selection"]["selected_complete_survival_cases"], 6)
        self.assertTrue(result["selection"]["continue_to_generation"])


if __name__ == "__main__":
    unittest.main()
