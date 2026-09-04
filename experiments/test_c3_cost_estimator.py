"""CPU tests for C3 preflight cost projection."""
from __future__ import annotations

import unittest

from experiments.phase2.estimate_c3_cost import estimate


class C3CostEstimatorTests(unittest.TestCase):
    def test_projection_uses_frozen_matrix_and_margin(self):
        summary = {
            "runtime": {"model_load_seconds": 100},
            "samples": [
                {
                    "stage": "preflight_32k",
                    "budget_fraction": 0.1,
                    "sample_prepare_seconds": 10,
                    "systems": {
                        "contiguous_cf": {"system_elapsed_seconds": 2},
                        "full_kv_reference": {"system_elapsed_seconds": 3},
                    },
                }
            ],
        }
        result = estimate(summary, hourly_rate=2.0)
        expected = (
            100
            + 24 * (10 + 3 + 24)
            + 12 * (10 + 3 + 8)
            + 12 * (10 + 12 + 32)
        )
        self.assertEqual(result["projected_seconds_before_margin"]["total"], expected)
        self.assertAlmostEqual(result["projected_gpu_hours"], expected * 1.25 / 3600)
        self.assertAlmostEqual(result["projected_cost"], expected * 1.25 / 3600 * 2)

    def test_rejects_non_preflight_summary(self):
        with self.assertRaises(ValueError):
            estimate({"runtime": {}, "samples": []})


if __name__ == "__main__":
    unittest.main()
