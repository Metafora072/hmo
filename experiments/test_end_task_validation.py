"""Focused tests for prospective equal-byte end-task validation."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from experiments.phase2.e3_v2.oracle import SegmentSpec
from experiments.phase2.e3_v2.run_end_task import (
    make_selected_segment_intervention,
    score_generated_text,
    select_equal_byte_segments,
    summarize_results,
)
from experiments.utils.dataset_utils import EvalSample


def _segments() -> tuple[SegmentSpec, ...]:
    return tuple(
        SegmentSpec(
            segment_id=index,
            start=index * 2,
            end=(index + 1) * 2,
            token_count=2,
            kv_bytes=16,
            protected=index in {0, 5},
            partial=False,
            normalized_position=(index + 0.5) / 6,
            position_bin=min(index, 3),
        )
        for index in range(6)
    )


class _Layer:
    def __init__(self, values: torch.Tensor):
        self.keys = values.clone()
        self.values = values.clone()

    def has_kv(self) -> bool:
        return True


class EndTaskSelectionTests(unittest.TestCase):
    def test_frozen_v2_can_change_membership_at_identical_budget(self):
        result = select_equal_byte_segments(
            {1: 0.40, 2: 0.30, 3: 0.20, 4: 0.10},
            {1: 1.00, 2: 0.00, 3: 0.75, 4: 0.25},
            _segments(),
            middle_kv_fraction=0.5,
        )
        self.assertTrue(result["gate_enabled"])
        self.assertTrue(result["membership_changed"])
        self.assertEqual(result["budget_slots"], 2)
        self.assertEqual(len(result["raw_alpha_segment_ids"]), 2)
        self.assertEqual(len(result["frozen_v2_segment_ids"]), 2)

    def test_intervention_keeps_protected_and_selected_segments(self):
        values = torch.arange(12, dtype=torch.float32).reshape(1, 1, 12, 1)
        cache = SimpleNamespace(layers=[_Layer(values)])
        intervention = make_selected_segment_intervention(
            _segments(), [0], [2, 4], context_tokens=12, name="selected"
        )
        result = intervention(cache, torch.zeros((1, 12), dtype=torch.long))
        expected = torch.tensor([0, 1, 4, 5, 8, 9, 10, 11])
        torch.testing.assert_close(result.active_context_positions.cpu(), expected)
        torch.testing.assert_close(
            cache.layers[0].keys.reshape(-1), expected.to(torch.float32)
        )
        self.assertEqual(result.metadata["context_resident_bytes"], 64)


class EndTaskMetricTests(unittest.TestCase):
    def test_contains_is_distinct_from_strict_exact_match(self):
        sample = EvalSample(
            dataset="needle",
            sample_id="sample",
            context="context",
            question="question",
            answer="ALPHA-321",
            answers=["ALPHA-321"],
            context_length=32,
        )
        metrics = score_generated_text("The answer is ALPHA-321.", sample)
        self.assertEqual(metrics["normalized_answer_contains"], 1.0)
        self.assertEqual(metrics["normalized_exact_match"], 0.0)
        self.assertGreater(metrics["token_f1"], 0.0)

    def test_summary_reports_paired_and_task_absolute_metrics(self):
        def row(sample_id: str, dataset: str, raw: float, v2: float) -> dict:
            systems = {}
            for system, score in (
                ("raw_alpha", raw),
                ("frozen_v2", v2),
                ("full_kv_reference", 1.0),
            ):
                systems[system] = {
                    "normalized_answer_contains": score,
                    "normalized_exact_match": score,
                    "token_f1": score,
                }
            return {
                "sample_id": sample_id,
                "dataset": dataset,
                "selection": {"gate_enabled": True, "membership_changed": raw != v2},
                "systems": systems,
            }

        summary = summarize_results(
            [row("a", "needle", 0.0, 1.0), row("b", "longeval_lines", 1.0, 1.0)],
            bootstrap_samples=50,
            seed=7,
        )
        primary = summary["paired_v2_vs_raw_alpha"]["normalized_answer_contains"]
        self.assertEqual(primary["wins"], 1)
        self.assertEqual(primary["losses"], 0)
        self.assertEqual(
            summary["task_systems"]["needle"]["frozen_v2"]
            ["normalized_answer_contains"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
