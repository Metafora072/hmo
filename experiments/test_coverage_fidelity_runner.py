"""CPU aggregation tests for the D1b development runner."""
from __future__ import annotations

import unittest

from experiments.phase2.e3_v2.oracle import SegmentSpec
from experiments.phase2.e3_v2.run_coverage_fidelity import (
    SYSTEMS,
    restrict_eligible_signals,
    summarize_results,
)


class CoverageFidelityRunnerTests(unittest.TestCase):
    def test_probe_signals_are_restricted_to_eligible_middle(self):
        segments = tuple(
            SegmentSpec(
                index,
                index * 2,
                (index + 1) * 2,
                2,
                16,
                index in {0, 3},
                False,
                (index + 0.5) / 4,
                index,
            )
            for index in range(4)
        )
        attention, accessibility = restrict_eligible_signals(
            {0: 0.0, 1: 0.1, 2: 0.2, 3: 0.3},
            {0: 1.0, 1: 0.9, 2: 0.8, 3: 0.7},
            segments,
        )
        self.assertEqual(attention, {1: 0.1, 2: 0.2})
        self.assertEqual(accessibility, {1: 0.9, 2: 0.8})

    def test_summary_reports_causal_and_baseline_comparisons(self):
        rows = []
        for index, stage in enumerate(("8k", "16k")):
            values = {
                "cf_hmo": 1.0,
                "cf_hmo_no_access": float(index),
                "sparse_only": 0.0,
                "raw_alpha_exact_topk": float(index),
                "full_kv_reference": 1.0,
            }
            systems = {
                system: {
                    "normalized_answer_contains": value,
                    "normalized_exact_match": value,
                    "token_f1": value,
                    "post_query_resident_kv_bytes": (
                        100 if system != "full_kv_reference" else 1000
                    ),
                }
                for system, value in values.items()
            }
            plans = {
                system: {
                    "action_counts": {
                        "recurrent_only": 0,
                        "sparse": 3,
                        "exact": index + 1,
                    }
                }
                for system in ("cf_hmo", "cf_hmo_no_access", "sparse_only")
            }
            rows.append(
                {
                    "stage": stage,
                    "systems": systems,
                    "plans": plans,
                    "baseline_reproduction": {
                        "raw_alpha": True,
                        "full_kv_reference": True,
                    },
                }
            )
        summary = summarize_results(rows)
        self.assertEqual(set(summary["systems"]), set(SYSTEMS))
        causal = summary["comparisons"]["cf_hmo_vs_no_access"]
        self.assertEqual(causal["normalized_answer_contains"]["mean_delta"], 0.5)
        self.assertEqual(causal["normalized_answer_contains"]["wins"], 1)
        self.assertEqual(summary["p3_baseline_reproduced_cases"], 2)
        self.assertEqual(summary["mean_resident_fraction_of_full"]["cf_hmo"], 0.1)
        attention_led = summary["comparisons"][
            "no_access_vs_raw_alpha_exact_topk"
        ]["normalized_answer_contains"]
        self.assertEqual(attention_led["mean_delta"], 0.0)


if __name__ == "__main__":
    unittest.main()
