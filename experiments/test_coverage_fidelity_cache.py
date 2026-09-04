"""Tiny-cache tests for query-attention coverage-fidelity intervention."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from experiments.phase2.e3_v2.coverage_fidelity import allocate_coverage_fidelity
from experiments.phase2.e3_v2.coverage_fidelity_cache import (
    build_raw_exact_slack_position_plan,
    build_retained_position_plan,
    make_coverage_fidelity_intervention,
    select_max_attention_window_positions,
    select_query_attention_positions,
)
from experiments.phase2.e3_v2.oracle import OracleContractError, SegmentSpec


class _Layer:
    def __init__(self, values):
        self.keys = values.clone()
        self.values = values.clone()

    def has_kv(self):
        return True


def _segments():
    return tuple(
        SegmentSpec(
            segment_id=index,
            start=index * 4,
            end=(index + 1) * 4,
            token_count=4,
            kv_bytes=32,
            protected=index in {0, 3},
            partial=False,
            normalized_position=(index + 0.5) / 4,
            position_bin=index,
        )
        for index in range(4)
    )


class CoverageFidelityCacheTests(unittest.TestCase):
    def test_max_mass_window_is_contiguous_and_uses_earliest_tie(self):
        segment = _segments()[1]
        mass = [0.0] * 16
        mass[4:8] = [1.0, 2.0, 1.0, 2.0]
        self.assertEqual(
            select_max_attention_window_positions(mass, segment, 2), [4, 5]
        )

    def test_stable_query_attention_tie_break(self):
        segment = _segments()[1]
        mass = [0.0] * 16
        mass[4:8] = [0.5, 0.5, 0.2, 0.1]
        self.assertEqual(select_query_attention_positions(mass, segment, 1), [4])

    def test_intervention_materializes_actions_and_exact_bytes(self):
        segments = _segments()
        plan = allocate_coverage_fidelity(
            {1: 0.9, 2: 0.8},
            {1: 0.9, 2: 0.1},
            segments,
            middle_kv_fraction=0.5,
            sparse_width=1,
            enable_exact_upgrades=False,
        )
        mass = [float(index) for index in range(16)]
        positions = build_retained_position_plan(
            plan, segments, mass, context_tokens=16
        )
        values = torch.arange(16, dtype=torch.float32).reshape(1, 1, 16, 1)
        cache = SimpleNamespace(layers=[_Layer(values)])
        result = make_coverage_fidelity_intervention(
            positions, [0], name="cf_test"
        )(cache, torch.zeros((1, 16), dtype=torch.long))
        expected = torch.tensor(positions.active_positions)
        torch.testing.assert_close(result.active_context_positions.cpu(), expected)
        torch.testing.assert_close(
            cache.layers[0].keys.reshape(-1), expected.to(torch.float32)
        )
        self.assertEqual(result.metadata["context_resident_bytes"], 96)
        self.assertEqual(positions.context_charged_bytes, plan.total_charged_bytes)

    def test_window_intervention_preserves_byte_contract(self):
        segments = _segments()
        plan = allocate_coverage_fidelity(
            {1: 0.9, 2: 0.8},
            {1: 0.9, 2: 0.1},
            segments,
            middle_kv_fraction=0.5,
            sparse_width=1,
            enable_exact_upgrades=False,
        )
        positions = build_retained_position_plan(
            plan,
            segments,
            [float(index % 4) for index in range(16)],
            context_tokens=16,
            sparse_selector="max_mass_window",
        )
        self.assertEqual(positions.sparse_selector, "max_mass_window")
        self.assertEqual(positions.context_charged_bytes, 96)

    def test_raw_exact_slack_matches_target_with_global_top_tokens(self):
        segments = _segments()
        mass = [0.0] * 16
        mass[8:12] = [0.3, 0.9, 0.8, 0.1]
        positions = build_raw_exact_slack_position_plan(
            segments,
            [1],
            mass,
            context_tokens=16,
            target_context_charged_bytes=112,
        )
        self.assertEqual(
            positions.active_positions,
            (0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 14, 15),
        )
        self.assertEqual(positions.context_charged_bytes, 112)
        retention = {item.segment_id: item for item in positions.segments}
        self.assertEqual(retention[1].action, "exact")
        self.assertEqual(retention[2].action, "sparse")

    def test_raw_exact_slack_rejects_unmatchable_byte_target(self):
        with self.assertRaisesRegex(
            OracleContractError, "target cannot be matched exactly"
        ):
            build_raw_exact_slack_position_plan(
                _segments(),
                [1],
                [0.0] * 16,
                context_tokens=16,
                target_context_charged_bytes=113,
            )


if __name__ == "__main__":
    unittest.main()
