"""Contracts for corrected-alpha evidence replacement."""
from __future__ import annotations

import unittest

from experiments.phase2.e3_v2.enrich_corrected_alpha import replace_alpha_rows
from experiments.phase2.e3_v2.oracle import OracleContractError


class CorrectedAlphaEnrichmentTest(unittest.TestCase):
    def test_replaces_alpha_and_dependent_products_only(self):
        raw = [
            {
                "sample_id": "old",
                "dataset": "needle",
                "segment_id": 1,
                "utility": 0.7,
                "alpha": 0.2,
                "normalized_position": 0.3,
                "candidates": {
                    "sigma_current": 0.5,
                    "delta_update": 0.25,
                    "suffix_interference": -1.0,
                    "phi_sigma_alpha": 0.1,
                    "phi_delta_alpha": 0.05,
                },
            }
        ]

        result = replace_alpha_rows(raw, {0: 0.1, 1: 0.8, 2: 0.2}, sample_id="new")

        self.assertEqual(result[0].sample_id, "new")
        self.assertEqual(result[0].alpha, 0.8)
        self.assertEqual(result[0].utility, 0.7)
        self.assertEqual(result[0].candidates["phi_sigma_alpha"], 0.4)
        self.assertEqual(result[0].candidates["phi_delta_alpha"], 0.2)
        self.assertEqual(result[0].candidates["suffix_interference"], -1.0)

    def test_rejects_segment_mismatch(self):
        with self.assertRaisesRegex(OracleContractError, "disagree"):
            replace_alpha_rows([], {1: 0.2}, sample_id="new")


if __name__ == "__main__":
    unittest.main()
