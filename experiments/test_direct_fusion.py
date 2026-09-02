import unittest

from experiments.phase2.e3_v2.direct_fusion import (
    conditional_controller_scores,
    evaluate_bounded_additive,
    evaluate_conditional_controller,
    evaluate_direct_fusions,
)
from experiments.phase2.e3_v2.statistics import SegmentEvidence


class DirectFusionTests(unittest.TestCase):
    def test_inverse_delta_rank_is_compared_against_raw_alpha(self):
        rows = []
        for sample_index in range(4):
            for segment_id, delta in enumerate((2.0, 0.0, 1.0)):
                rows.append(
                    SegmentEvidence(
                        sample_id=f"sample_{sample_index}",
                        dataset="synthetic",
                        segment_id=segment_id,
                        utility=2.0 - delta,
                        alpha=1.0,
                        normalized_position=0.0,
                        candidates={
                            "sigma_current": 1.0,
                            "delta_update": delta,
                        },
                    )
                )
        result = evaluate_direct_fusions(
            rows,
            {f"sample_{index}": 1 for index in range(4)},
            bootstrap_samples=100,
            seed=11,
        )
        inverse_delta = result["methods"]["alpha_inverse_delta_rank"]
        self.assertGreater(inverse_delta["pairwise_improvement"]["mean"], 0.0)
        self.assertGreater(inverse_delta["ndcg_improvement"]["mean"], 0.0)
        sigma = result["methods"]["alpha_sigma_product"]
        self.assertEqual(sigma["pairwise_improvement"]["mean"], 0.0)


    def test_bounded_additive_selects_smaller_positive_lambda_on_tie(self):
        rows = []
        for sample_index in range(4):
            for segment_id, utility in enumerate((0.0, 1.0, 2.0)):
                rows.append(
                    SegmentEvidence(
                        sample_id=f"sample_{sample_index}",
                        dataset="synthetic",
                        segment_id=segment_id,
                        utility=utility,
                        alpha=1.0,
                        normalized_position=0.0,
                        candidates={
                            "sigma_current": utility,
                            "delta_update": 0.0,
                        },
                    )
                )
        result = evaluate_bounded_additive(
            rows,
            {f"sample_{index}": 1 for index in range(4)},
            bootstrap_samples=100,
            seed=17,
        )
        self.assertEqual(result["selected_lambda"], 0.15)
        selected = result["methods"][result["selected_method"]]
        self.assertGreater(selected["pairwise_improvement"]["mean"], 0.0)
        self.assertGreater(selected["ndcg_improvement"]["mean"], 0.0)

    def test_bounded_additive_rejects_out_of_family_lambda(self):
        with self.assertRaises(ValueError):
            evaluate_bounded_additive(
                [],
                {},
                lambdas=(0.31,),
                bootstrap_samples=10,
                seed=1,
            )

    def test_conditional_controller_swaps_only_adjacent_regime_inversion(self):
        scores, regimes, swaps = conditional_controller_scores(
            alpha=[4.0, 3.0, 2.0, 1.0],
            sigma_current=[0.0, 3.0, 4.0, 1.0],
            delta_update=[0.0, 1.0, 4.0, 3.0],
        )
        self.assertEqual(regimes, ("NEUTRAL", "SAFE", "STRESSED", "NEUTRAL"))
        self.assertEqual(swaps, ((1, 2),))
        self.assertEqual(scores.tolist(), [4.0, 2.0, 3.0, 1.0])

    def test_conditional_controller_improves_synthetic_top_k(self):
        rows = []
        for sample_index in range(4):
            dataset = "task-a" if sample_index < 2 else "task-b"
            for segment_id, (utility, alpha, sigma, delta) in enumerate(
                zip(
                    (0.0, 1.0, 3.0, -1.0),
                    (4.0, 3.0, 2.0, 1.0),
                    (0.0, 3.0, 4.0, 1.0),
                    (0.0, 1.0, 4.0, 3.0),
                )
            ):
                rows.append(
                    SegmentEvidence(
                        sample_id=f"sample_{sample_index}",
                        dataset=dataset,
                        segment_id=segment_id,
                        utility=utility,
                        alpha=alpha,
                        normalized_position=0.0,
                        candidates={
                            "sigma_current": sigma,
                            "delta_update": delta,
                        },
                    )
                )
        result = evaluate_conditional_controller(
            rows,
            {f"sample_{index}": 2 for index in range(4)},
            bootstrap_samples=100,
            seed=23,
        )
        self.assertGreater(
            result["controller"]["pairwise_improvement"]["mean"],
            0.0,
        )
        self.assertGreater(
            result["controller"]["ndcg_improvement"]["mean"],
            0.0,
        )
        self.assertTrue(
            all(
                row["adjacent_swaps"] == [[1, 2]]
                for row in result["samples"].values()
            )
        )


if __name__ == "__main__":
    unittest.main()
