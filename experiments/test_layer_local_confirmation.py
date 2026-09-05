from __future__ import annotations

import unittest

from experiments.phase2.e3_v2.freeze_layer_local_confirmation_protocol import (
    select_stratified_cases,
)
from experiments.phase2.e3_v2.run_layer_local_confirmation import summarize_results


class LayerLocalConfirmationTests(unittest.TestCase):
    def test_selection_is_stratified_deterministic_and_outcome_free(self):
        candidates = [
            {
                "dataset": "qasper",
                "sample_id": f"sample_{index:02d}",
                "record_index": index,
                "context_tokens": 100 + index,
                "private_outcome": 1.0 if index % 2 else 0.0,
            }
            for index in range(40)
        ]
        first = select_stratified_cases(
            candidates, dataset="qasper", count=20, strata=4, seed=17
        )
        changed = [dict(item, private_outcome=1.0 - item["private_outcome"]) for item in candidates]
        second = select_stratified_cases(
            changed, dataset="qasper", count=20, strata=4, seed=17
        )
        self.assertEqual(
            [item["sample_id"] for item in first],
            [item["sample_id"] for item in second],
        )
        self.assertEqual(
            {stratum: sum(item["length_stratum"] == stratum for item in first) for stratum in range(4)},
            {0: 5, 1: 5, 2: 5, 3: 5},
        )

    def test_summary_audits_total_and_per_layer_equal_bytes(self):
        rows = []
        for index in range(2):
            systems = {}
            for name, score, resident in (
                ("hmo_legacy", 0.2, 100),
                ("hmo_layer_local", 0.4 + index * 0.2, 100),
                ("chunkkv", 0.3, 100),
                ("full_kv_reference", 0.7, 500),
            ):
                systems[name] = {
                    "official_qa_f1": score,
                    "normalized_answer_contains": float(score > 0.3),
                    "normalized_exact_match": 0.0,
                    "post_query_resident_kv_bytes": resident,
                    "post_query_layer_kv_bytes": {"3": resident // 2, "7": resident // 2},
                    "hit_generation_limit": False,
                }
            rows.append({"systems": systems})
        summary = summarize_results(rows)
        self.assertEqual(summary["equal_resident_byte_cases"], 2)
        self.assertEqual(summary["equal_layer_byte_cases"], 2)
        self.assertAlmostEqual(
            summary["comparisons"]["hmo_layer_local_vs_hmo_legacy"]
            ["official_qa_f1"]["mean_delta"],
            0.3,
        )


if __name__ == "__main__":
    unittest.main()
