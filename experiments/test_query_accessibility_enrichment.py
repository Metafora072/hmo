"""Contracts for query-accessibility enrichment and fixed scores."""
from __future__ import annotations

import unittest

from experiments.phase2.e3_v2.enrich_query_accessibility import (
    evaluate_accessibility_scores,
    replace_query_probe_rows,
)
from experiments.phase2.e3_v2.oracle import OracleContractError
from experiments.phase2.e3_v2.statistics import SegmentEvidence


class QueryAccessibilityEnrichmentTests(unittest.TestCase):
    def test_replacement_preserves_labels_and_adds_query_candidates(self):
        raw = [
            {
                "dataset": "needle",
                "segment_id": 1,
                "utility": 0.7,
                "alpha": 0.2,
                "normalized_position": 0.3,
                "candidates": {
                    "sigma_current": 0.5,
                    "delta_update": 0.25,
                    "phi_sigma_alpha": 0.1,
                    "phi_delta_alpha": 0.05,
                },
            }
        ]
        result = replace_query_probe_rows(
            raw,
            {0: 0.1, 1: 0.8},
            {0: 0.2, 1: 0.4},
            {0: 0.3, 1: 0.6},
            {0: 0.4, 1: -0.2},
            sample_id="new",
        )
        self.assertEqual(result[0].sample_id, "new")
        self.assertEqual(result[0].utility, 0.7)
        self.assertEqual(result[0].alpha, 0.8)
        self.assertAlmostEqual(result[0].candidates["alpha_read_share"], 0.48)
        self.assertAlmostEqual(result[0].candidates["phi_delta_alpha"], 0.2)

    def test_replacement_rejects_missing_segments(self):
        with self.assertRaisesRegex(OracleContractError, "disagree"):
            replace_query_probe_rows(
                [{"segment_id": 1}],
                {1: 0.2},
                {1: 0.2},
                {},
                {1: 0.2},
                sample_id="bad",
            )

    def test_direct_scores_report_budget_membership_changes(self):
        evidence = []
        for sample_id in ("a", "b"):
            for segment_id, (alpha, access, utility) in enumerate(
                ((0.9, 0.9, 0.1), (0.8, 0.1, 1.0), (0.1, 0.5, 0.0))
            ):
                evidence.append(
                    SegmentEvidence(
                        sample_id=sample_id,
                        dataset="needle",
                        segment_id=segment_id,
                        utility=utility,
                        alpha=alpha,
                        normalized_position=segment_id / 2,
                        candidates={"query_read_share": access},
                    )
                )
        result = evaluate_accessibility_scores(
            evidence,
            {"a": 1, "b": 1},
            bootstrap_samples=20,
            seed=5,
        )
        deficit = result["methods"]["access_deficit"]
        self.assertEqual(deficit["topk_changed_samples"], 2)
        self.assertGreater(deficit["ndcg_improvement"]["mean"], 0)
        self.assertLess(
            result["methods"]["access_excess"]["ndcg_improvement"]["mean"],
            deficit["ndcg_improvement"]["mean"],
        )


if __name__ == "__main__":
    unittest.main()
