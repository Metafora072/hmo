"""CPU tests for staged C3 formal cost projection."""
from __future__ import annotations

import unittest

from experiments.phase2.estimate_c3_cost import estimate


class C3CostEstimatorTests(unittest.TestCase):
    def test_projection_uses_frozen_matrix_and_margin(self):
        summary = {
            "runtime": {"model_load_seconds": 100},
            "samples": [
                {
                    "stage": "formal_32k",
                    "budget_fraction": 0.1,
                    "sample_prepare_seconds": 10,
                    "systems": {
                        "contiguous_cf": {"prompt_intervention_seconds": 2, "decode_seconds": 1},
                        "chunkkv": {"prompt_intervention_seconds": 2, "decode_seconds": 1},
                        "global_fixed_chunk_topk": {"prompt_intervention_seconds": 2, "decode_seconds": 1},
                        "raw_alpha_exact_slack": {"prompt_intervention_seconds": 2, "decode_seconds": 1},
                        "full_kv_reference": {"prompt_intervention_seconds": 4, "decode_seconds": 2},
                    },
                }
            ],
        }
        result = estimate(summary, hourly_rate=2.0)
        expected = (
            100
            + 24 * (10 + 4 + 2 + 12 * 3)
            + 12 * (5 + 2 + 2 + 4 * (1 + 1))
            + 12 * (5 + 2 + 8 + 4 * (1 + 4))
        )
        self.assertEqual(result["projected_seconds_before_margin"]["total"], expected)
        self.assertAlmostEqual(result["projected_gpu_hours"], expected * 1.25 / 3600)
        self.assertAlmostEqual(result["projected_cost"], expected * 1.25 / 3600 * 2)

    def test_rejects_summary_without_formal_central_rows(self):
        with self.assertRaises(ValueError):
            estimate({"runtime": {}, "samples": []})


if __name__ == "__main__":
    unittest.main()
