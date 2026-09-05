from __future__ import annotations

import itertools
import random
import unittest
from types import SimpleNamespace

import torch

from experiments.phase2.e3_v2.chunkkv_adapter import build_chunkkv_plan
from experiments.phase2.e3_v2.coverage_fidelity_cache import (
    RetainedPositionPlan,
    SegmentRetention,
)
from experiments.phase2.e3_v2.free_window_allocator import (
    FREE_WINDOW_SCHEMA,
    build_free_window_plan,
    build_layer_local_hmo_plan,
    make_layerwise_window_intervention,
    select_optimal_fixed_width_windows,
)
from experiments.phase2.e3_v2.oracle import SegmentSpec


def _segment(segment_id, start, end, *, protected=False, partial=False):
    return SegmentSpec(
        segment_id=segment_id,
        start=start,
        end=end,
        token_count=end - start,
        kv_bytes=(end - start) * 16,
        protected=protected,
        partial=partial,
        normalized_position=(start + end) / 50,
        position_bin=0,
    )


class _Layer:
    def __init__(self, tokens):
        values = torch.arange(tokens, dtype=torch.float32).reshape(1, 1, tokens, 1)
        self.keys = values.clone()
        self.values = values.clone()

    def has_kv(self):
        return True


class FreeWindowAllocatorTests(unittest.TestCase):
    def test_dp_matches_brute_force(self):
        rng = random.Random(20260905)
        for _ in range(120):
            length = rng.randint(3, 12)
            width = rng.randint(1, min(4, length))
            count = rng.randint(0, length // width)
            scores = [rng.randint(0, 7) for _ in range(length)]
            starts, mass = select_optimal_fixed_width_windows(
                scores,
                eligible_positions=range(length),
                fixed_positions=(),
                window_width=width,
                window_count=count,
            )
            candidates = range(length - width + 1)
            brute = []
            for choice in itertools.combinations(candidates, count):
                if all(a + width <= b for a, b in zip(choice, choice[1:])):
                    brute.append(sum(sum(scores[s : s + width]) for s in choice))
            expected = max(brute) if brute else 0
            self.assertEqual(mass, expected)
            self.assertEqual(len(starts), count)
            self.assertTrue(all(a + width <= b for a, b in zip(starts, starts[1:])))

    def test_fixed_short_fragments_preserve_chunkkv_feasibility(self):
        segments = (
            _segment(0, 0, 2, protected=True),
            _segment(1, 2, 23),
            _segment(2, 23, 25, protected=True, partial=True),
        )
        scores = [0.0] * 25
        scores[2:6] = [8.0] * 4
        scores[22] = 20.0
        scores[6:10] = [4.0] * 4
        baseline = build_chunkkv_plan(
            segments,
            {3: scores},
            context_tokens=25,
            target_context_charged_bytes=10 * 16,
            context_token_kv_bytes=16,
            observation_query_tokens=2,
            chunk_size=4,
        )
        plan = build_free_window_plan(segments, {3: scores}, baseline)
        layer = plan.layers[0]
        self.assertEqual(plan.schema_version, FREE_WINDOW_SCHEMA)
        self.assertEqual(len(layer.active_positions), 10)
        self.assertEqual(sum(len(x.positions) for x in layer.fixed_fragments), 2)
        self.assertGreaterEqual(
            layer.middle_attention_mass, layer.baseline_middle_attention_mass
        )
        self.assertEqual(plan.context_charged_bytes, baseline.context_charged_bytes)

    def test_layer_local_keeps_actions_and_improves_layer_mass(self):
        segments = (
            _segment(0, 0, 4, protected=True),
            _segment(1, 4, 12),
            _segment(2, 12, 16, protected=True),
        )
        legacy = RetainedPositionPlan(
            context_tokens=16,
            context_charged_bytes=10 * 16,
            sparse_selector="max_mass_window",
            active_positions=tuple(range(4)) + (4, 5) + tuple(range(12, 16)),
            segments=(
                SegmentRetention(0, "exact", tuple(range(4))),
                SegmentRetention(1, "sparse", (4, 5)),
                SegmentRetention(2, "exact", tuple(range(12, 16))),
            ),
        )
        first = [0.0] * 16
        second = [0.0] * 16
        first[8:10] = [5.0, 5.0]
        second[10:12] = [6.0, 6.0]
        plan = build_layer_local_hmo_plan(
            segments,
            {3: first, 7: second},
            legacy,
            context_token_kv_bytes=16,
        )
        self.assertEqual(plan.layers[0].window_starts, (8,))
        self.assertEqual(plan.layers[1].window_starts, (10,))
        self.assertTrue(
            all(
                layer.middle_attention_mass >= layer.baseline_middle_attention_mass
                for layer in plan.layers
            )
        )

    def test_layerwise_intervention_applies_distinct_positions(self):
        segments = (
            _segment(0, 0, 2, protected=True),
            _segment(1, 2, 10),
            _segment(2, 10, 12, protected=True),
        )
        legacy = RetainedPositionPlan(
            context_tokens=12,
            context_charged_bytes=8 * 16,
            sparse_selector="max_mass_window",
            active_positions=(0, 1, 2, 3, 4, 5, 10, 11),
            segments=(
                SegmentRetention(0, "exact", (0, 1)),
                SegmentRetention(1, "sparse", (2, 3, 4, 5)),
                SegmentRetention(2, "exact", (10, 11)),
            ),
        )
        first = [0.0] * 12
        second = [0.0] * 12
        first[4:8] = [3.0] * 4
        second[6:10] = [4.0] * 4
        plan = build_layer_local_hmo_plan(
            segments, {0: first, 1: second}, legacy, context_token_kv_bytes=16
        )
        cache = SimpleNamespace(layers=[_Layer(12), _Layer(12)])
        result = make_layerwise_window_intervention(plan, name="layer_local")(
            cache, torch.zeros((1, 12), dtype=torch.long)
        )
        self.assertEqual(result.metadata["context_resident_bytes"], 128)
        self.assertNotEqual(plan.layers[0].active_positions, plan.layers[1].active_positions)
        for layer, expected in zip(cache.layers, plan.layers):
            torch.testing.assert_close(
                layer.keys.reshape(-1),
                torch.tensor(expected.active_positions, dtype=torch.float32),
            )


if __name__ == "__main__":
    unittest.main()
