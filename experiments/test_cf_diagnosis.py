"""CPU contract tests for the development-only CF-HMO diagnosis."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from experiments.phase2.e3_v2.oracle import SegmentSpec
from experiments.phase2.e3_v2.run_cf_diagnosis import (
    build_survival_record,
    select_query_attention_positions,
    summarize_diagnosis,
)
from experiments.utils.kv_ops import select_token_skeleton_positions


class _Layer:
    def __init__(self, keys, values):
        self.keys = keys
        self.values = values

    def has_kv(self):
        return True


class CfDiagnosisTests(unittest.TestCase):
    def test_kv_norm_selector_matches_shared_sparse_rule(self):
        keys = torch.tensor([[[[1.0], [4.0], [2.0], [3.0]]]])
        values = torch.zeros_like(keys)
        cache = SimpleNamespace(layers=[_Layer(keys, values)])
        positions, original_bytes = select_token_skeleton_positions(
            cache, [0], 0, 4, 2
        )
        self.assertEqual(positions, [1, 3])
        self.assertEqual(original_bytes, 32)

    def test_query_attention_positions_are_global_and_ordered(self):
        segment = SegmentSpec(2, 4, 8, 4, 40, False, False, 0.5, 2)
        positions = select_query_attention_positions(
            [0.0, 0.0, 0.0, 0.0, 0.1, 0.9, 0.3, 0.8], segment, 2
        )
        self.assertEqual(positions, [5, 7])

    def test_survival_record_distinguishes_any_and_all(self):
        record = build_survival_record([5, 6], {2: [4, 5, 7]})
        self.assertTrue(record["any_answer_token_survived"])
        self.assertFalse(record["all_answer_tokens_survived"])
        self.assertEqual(record["answer_token_retained_fraction"], 0.5)

    def test_summary_preserves_teacher_and_sparse_axes(self):
        rows = []
        for index, delta in enumerate((1.0, -1.0)):
            systems = {
                "raw_alpha": {"mean_logprob": -2.0},
                "frozen_v2": {"mean_logprob": -2.0 + delta},
                "full_kv_reference": {"mean_logprob": -1.0},
            }
            sparse = {
                selector: {
                    "8": {
                        "all_answer_tokens_survived": index == 0,
                        "any_answer_token_survived": True,
                        "answer_token_retained_fraction": 1.0 - 0.5 * index,
                    },
                    "16": {
                        "all_answer_tokens_survived": True,
                        "any_answer_token_survived": True,
                        "answer_token_retained_fraction": 1.0,
                    },
                }
                for selector in ("kv_norm", "query_attention")
            }
            rows.append(
                {
                    "stage": "8k" if index == 0 else "16k",
                    "systems": systems,
                    "source_outcome": {"delta": delta},
                    "sparse_survival": sparse,
                }
            )
        summary = summarize_diagnosis(rows, (8, 16))
        self.assertEqual(summary["outcomes"], {"v2_wins": 1, "ties": 0, "v2_losses": 1})
        self.assertEqual(
            summary["sparse_survival"]["kv_norm"]["8"]
            ["all_answer_tokens_survived_cases"],
            1,
        )
        self.assertEqual(
            summary["teacher_forced"]["frozen_v2_minus_raw_alpha"]["mean_delta"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
