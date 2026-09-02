"""Focused contract tests for the frozen boundary-exchange policy."""
from __future__ import annotations

import unittest

from experiments.phase2.e3_v2.boundary_exchange import boundary_exchange_scores


class BoundaryExchangeTest(unittest.TestCase):
    def test_swaps_lowest_alpha_safe_insider_for_highest_alpha_stressed_outsider(self):
        scores, regimes, exchange = boundary_exchange_scores(
            alpha=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            sigma_current=[0.1, 0.9, 0.2, 0.8, 0.7, 0.3],
            delta_update=[0.1, 0.2, 0.3, 0.9, 0.8, 0.4],
            k=3,
        )

        self.assertEqual(regimes[1], "SAFE")
        self.assertEqual(regimes[3], "STRESSED")
        self.assertEqual(exchange, {"safe_inside_index": 1, "stressed_outside_index": 3})
        selected = set(scores.argsort()[::-1][:3].tolist())
        self.assertEqual(selected, {0, 2, 3})

    def test_noops_when_either_boundary_candidate_is_missing(self):
        scores, _, exchange = boundary_exchange_scores(
            alpha=[0.9, 0.8, 0.7, 0.6],
            sigma_current=[0.1, 0.2, 0.8, 0.9],
            delta_update=[0.1, 0.2, 0.8, 0.9],
            k=2,
        )

        self.assertIsNone(exchange)
        self.assertEqual(scores.argsort()[::-1].tolist(), [0, 1, 2, 3])

    def test_rejects_invalid_budget(self):
        with self.assertRaises(ValueError):
            boundary_exchange_scores(
                alpha=[0.8, 0.7],
                sigma_current=[0.2, 0.9],
                delta_update=[0.1, 0.8],
                k=2,
            )


if __name__ == "__main__":
    unittest.main()
