"""CPU contracts for the equal-byte hybrid ChunkKV adapter."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from experiments.phase2.e3_v2.chunkkv_adapter import (
    CHUNKKV_ADAPTER_SCHEMA,
    build_chunkkv_plan,
    make_chunkkv_intervention,
)
from experiments.phase2.e3_v2.oracle import OracleContractError, SegmentSpec


class _Layer:
    def __init__(self, tokens):
        values = torch.arange(tokens, dtype=torch.float32).reshape(1, 1, tokens, 1)
        self.keys = values.clone()
        self.values = values.clone()

    def has_kv(self):
        return True


def _segments():
    return tuple(
        SegmentSpec(
            segment_id=index,
            start=index * 10,
            end=(index + 1) * 10,
            token_count=10,
            kv_bytes=160,
            protected=index in {0, 3},
            partial=False,
            normalized_position=(index + 0.5) / 4,
            position_bin=index,
        )
        for index in range(4)
    )


class ChunkKVAdapterTests(unittest.TestCase):
    def test_layers_rank_chunks_independently_at_equal_count(self):
        first = [0.0] * 40
        second = [0.0] * 40
        first[10:20] = [2.0] * 10
        second[20:30] = [2.0] * 10
        plan = build_chunkkv_plan(
            _segments(),
            {0: first, 1: second},
            context_tokens=40,
            target_context_charged_bytes=480,
            context_token_kv_bytes=16,
            observation_query_tokens=7,
        )
        self.assertEqual(plan.layers[0].selected_chunk_starts, (10,))
        self.assertEqual(plan.layers[1].selected_chunk_starts, (20,))
        self.assertEqual(len(plan.layers[0].active_positions), 30)
        self.assertEqual(len(plan.layers[1].active_positions), 30)
        self.assertNotEqual(
            plan.layers[0].active_positions, plan.layers[1].active_positions
        )
        self.assertEqual(plan.to_dict()["schema_version"], CHUNKKV_ADAPTER_SCHEMA)

    def test_partial_chunk_uses_fixed_prefix(self):
        scores = [0.0] * 40
        scores[20:30] = [3.0] * 10
        plan = build_chunkkv_plan(
            _segments(),
            {0: scores},
            context_tokens=40,
            target_context_charged_bytes=208,
            context_token_kv_bytes=8,
            observation_query_tokens=4,
        )
        layer = plan.layers[0]
        self.assertEqual(layer.partial_chunk_start, 20)
        self.assertEqual(layer.partial_chunk_tokens, 6)
        self.assertTrue(set(range(20, 26)) <= set(layer.active_positions))

    def test_intervention_applies_per_layer_positions_and_exact_bytes(self):
        first = [0.0] * 40
        second = [0.0] * 40
        first[10:20] = [2.0] * 10
        second[20:30] = [2.0] * 10
        plan = build_chunkkv_plan(
            _segments(),
            {0: first, 1: second},
            context_tokens=40,
            target_context_charged_bytes=480,
            context_token_kv_bytes=16,
            observation_query_tokens=5,
        )
        cache = SimpleNamespace(layers=[_Layer(40), _Layer(40), SimpleNamespace()])
        result = make_chunkkv_intervention(plan)(
            cache, torch.zeros((1, 40), dtype=torch.long)
        )
        for layer, layer_plan in zip(cache.layers, plan.layers):
            self.assertEqual(layer.keys.shape[-2], 30)
            torch.testing.assert_close(
                layer.keys.reshape(-1),
                torch.tensor(layer_plan.active_positions, dtype=torch.float32),
            )
        self.assertEqual(result.metadata["context_resident_bytes"], 480)
        self.assertEqual(len(result.metadata["layer_position_hashes"]), 2)

    def test_rejects_unmatchable_target(self):
        with self.assertRaisesRegex(OracleContractError, "invalid"):
            build_chunkkv_plan(
                _segments(),
                {0: [0.0] * 40},
                context_tokens=40,
                target_context_charged_bytes=201,
                context_token_kv_bytes=8,
                observation_query_tokens=4,
            )


if __name__ == "__main__":
    unittest.main()
