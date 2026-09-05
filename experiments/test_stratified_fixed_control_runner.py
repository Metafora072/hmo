"""CPU contract tests for the frozen Stratified Fixed-Chunk control."""
from __future__ import annotations

import hashlib
import unittest

from experiments.phase2.e3_v2.run_stratified_fixed_control import (
    PROJECT_ROOT,
    load_control_protocol,
    summarize_control_results,
)


class StratifiedFixedControlRunnerTests(unittest.TestCase):
    def test_frozen_protocol_is_bound_to_package_b(self):
        parent_protocol = PROJECT_ROOT / (
            "refine-logs/contiguous_cf_pareto_protocol_legacy_v1.json"
        )
        parent_protocol_sha = hashlib.sha256(parent_protocol.read_bytes()).hexdigest()
        parent_results_sha = (
            "5757ff898b921c1b0fcc6ed1e76d195667070999d56bf71189a752e38d49e1ab"
        )
        protocol, protocol_sha = load_control_protocol(
            PROJECT_ROOT
            / "refine-logs/stratified_fixed_chunk_control_protocol.json",
            parent_protocol_sha256=parent_protocol_sha,
            parent_results_sha256=parent_results_sha,
        )
        self.assertEqual(len(protocol_sha), 64)
        self.assertEqual(protocol["method"]["window_start_alignment"], 16)
        self.assertEqual(protocol["execution"]["formal_sample_cases"], 24)
        self.assertFalse(protocol["execution"]["continuation_gate"])

    def test_summary_uses_aligned_minus_parent_direction(self):
        rows = []
        for index, scores in enumerate(((0.0, 1.0), (1.0, 0.0))):
            hmo_score, aligned_score = scores
            rows.append(
                {
                    "stage": "formal",
                    "sample_id": f"sample_{index}",
                    "dataset": "needle",
                    "geometry": {"retained_position_jaccard": 0.5},
                    "systems": {
                        "contiguous_cf_parent": {
                            "normalized_answer_contains": hmo_score,
                            "normalized_exact_match": hmo_score,
                            "token_f1": hmo_score,
                            "post_query_resident_kv_bytes": 100,
                            "generated_token_ids": [index],
                        },
                        "stratified_fixed_chunk": {
                            "normalized_answer_contains": aligned_score,
                            "normalized_exact_match": aligned_score,
                            "token_f1": aligned_score,
                            "post_query_resident_kv_bytes": 100,
                            "generated_token_ids": [1 - index],
                        },
                    },
                }
            )
        summary = summarize_control_results(rows)
        primary = summary["comparison"]["normalized_answer_contains"]
        self.assertEqual((primary["wins"], primary["ties"], primary["losses"]), (1, 0, 1))
        self.assertEqual(primary["mean_delta"], 0.0)
        self.assertEqual(summary["equal_resident_byte_cases"], 2)
        self.assertEqual(summary["mean_retained_position_jaccard"], 0.5)


if __name__ == "__main__":
    unittest.main()
