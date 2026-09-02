"""No-GPU contract tests for conditional recurrent regime analysis."""
from __future__ import annotations

import unittest

from experiments.phase2.e3_v2.conditional_regime import analyze_conditional_regimes
from experiments.phase2.e3_v2.oracle import OracleContractError
from experiments.phase2.e3_v2.statistics import (
    SegmentEvidence,
    grouped_baseline_residuals,
)


def make_regime_evidence(*, reverse_needle: bool = False):
    evidence = []
    quadrants = (
        (0.0, 0.0),
        (0.0, 1.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (1.0, 1.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (0.0, 0.0),
    )
    for sample_index in range(8):
        dataset = "LongEval" if sample_index < 4 else "Needle"
        direction = -1.0 if reverse_needle and dataset == "Needle" else 1.0
        for segment_id, (sigma, delta) in enumerate(quadrants):
            effect = 0.0
            if sigma == 1.0 and delta == 0.0:
                effect = -2.0 * direction
            elif sigma == 1.0 and delta == 1.0:
                effect = 2.0 * direction
            evidence.append(
                SegmentEvidence(
                    sample_id=f"sample-{sample_index}",
                    dataset=dataset,
                    segment_id=segment_id,
                    utility=sample_index * 0.25 + effect,
                    alpha=0.0,
                    normalized_position=0.0,
                    candidates={
                        "sigma_current": sigma,
                        "delta_update": delta,
                    },
                )
            )
    return tuple(evidence)


class ConditionalRegimeTests(unittest.TestCase):
    def test_supports_same_direction_stressed_minus_safe_pattern(self):
        result = analyze_conditional_regimes(
            make_regime_evidence(),
            bootstrap_samples=200,
            seed=31,
        )
        contrast = result["q4_stressed_minus_q3_safe"]
        self.assertTrue(result["pattern_supported"])
        self.assertTrue(result["task_direction_consistent"])
        self.assertEqual(result["samples_with_q3_and_q4"], 8)
        self.assertAlmostEqual(
            contrast["sample_grouped_bootstrap"]["mean"],
            4.0,
        )
        self.assertEqual(contrast["sign_counts"]["positive"], 8)
        self.assertEqual(
            result["decision"],
            "freeze_minimal_three_state_controller",
        )

    def test_rejects_task_direction_reversal_without_threshold_search(self):
        result = analyze_conditional_regimes(
            make_regime_evidence(reverse_needle=True),
            bootstrap_samples=200,
            seed=37,
        )
        task_means = result["q4_stressed_minus_q3_safe"]["task_means"]
        self.assertGreater(task_means["LongEval"]["mean"], 0.0)
        self.assertLess(task_means["Needle"]["mean"], 0.0)
        self.assertFalse(result["task_direction_consistent"])
        self.assertFalse(result["pattern_supported"])
        self.assertFalse(result["regime_definition"]["threshold_search"])
        self.assertEqual(
            result["decision"],
            "do_not_tune_thresholds_return_to_openchat",
        )

    def test_grouped_residuals_reject_duplicate_segment_ids(self):
        evidence = make_regime_evidence()
        with self.assertRaisesRegex(OracleContractError, "unique"):
            grouped_baseline_residuals((evidence[0], evidence[0], *evidence[1:]))


if __name__ == "__main__":
    unittest.main()
