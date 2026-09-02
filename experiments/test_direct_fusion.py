import unittest

from experiments.phase2.e3_v2.direct_fusion import evaluate_direct_fusions
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


if __name__ == "__main__":
    unittest.main()
