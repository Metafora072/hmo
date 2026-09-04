"""CPU tests for the post-hoc format-robust analysis."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.phase2.e3_v2.format_robust import (
    analyze_result_file,
    format_robust_contains,
)


class FormatRobustTests(unittest.TestCase):
    def test_clock_aliases_apply_only_to_needle_clock_truths(self):
        self.assertEqual(format_robust_contains("8:38", "838 o'clock", "needle"), 1.0)
        self.assertEqual(
            format_robust_contains("The answer is 4:03 PM.", "403 o'clock", "needle"),
            1.0,
        )
        self.assertEqual(
            format_robust_contains("8:38", "838 o'clock", "longeval_lines"), 0.0
        )
        self.assertEqual(format_robust_contains("90A0D", "90A0QD", "longeval_lines"), 0.0)

    def test_analysis_reports_upgrades_without_changing_primary(self):
        row = {
            "sample_id": "case-1",
            "stage": "16k",
            "dataset": "needle",
            "answer": "838 o'clock",
            "systems": {
                "contiguous_cf": {
                    "generated_text": "8:38",
                    "normalized_answer_contains": 0.0,
                },
                "scattered_cf": {
                    "generated_text": "unknown",
                    "normalized_answer_contains": 0.0,
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            result = analyze_result_file(path, "test")
        self.assertEqual(result["systems"]["contiguous_cf"]["primary_correct"], 0)
        self.assertEqual(result["systems"]["contiguous_cf"]["secondary_correct"], 1)
        self.assertEqual(result["systems"]["contiguous_cf"]["upgrades"], 1)
        self.assertEqual(result["systems"]["contiguous_cf"]["downgrades"], 0)


if __name__ == "__main__":
    unittest.main()
