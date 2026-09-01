"""No-GPU integrity tests for E3-v2 P0-C recurrent candidates."""
from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import torch

from experiments.phase2.e3_v2.protocol import (
    P0C_PROTOCOL_VERSION,
    recurrent_signal_protocol,
)
from experiments.phase2.e3_v2.recurrent_signals import (
    DeltaRuleTrace,
    LayerRecurrentCandidates,
    Qwen35RecurrentCandidateHookManager,
    RecurrentSignalError,
    aggregate_recurrent_candidates,
    chunk_gated_delta_trace,
    sequential_gated_delta_trace,
    summarize_recurrent_trace,
)
from experiments.utils.run_manifest import build_run_spec


class DeltaRuleEquivalenceTests(unittest.TestCase):
    def test_chunk_trace_matches_token_reference(self):
        torch.manual_seed(7)
        for tokens, chunk_size in ((1, 4), (7, 4), (9, 8), (17, 8)):
            with self.subTest(tokens=tokens, chunk_size=chunk_size):
                key = torch.randn(2, tokens, 3, 5)
                value = torch.randn(2, tokens, 3, 4)
                beta = torch.sigmoid(torch.randn(2, tokens, 3))
                log_decay = -torch.rand(2, tokens, 3) * 0.7
                initial_state = torch.randn(2, 3, 5, 4) * 0.1
                reference = sequential_gated_delta_trace(
                    key,
                    value,
                    beta,
                    log_decay,
                    initial_state=initial_state,
                )
                chunked = chunk_gated_delta_trace(
                    key,
                    value,
                    beta,
                    log_decay,
                    initial_state=initial_state,
                    chunk_size=chunk_size,
                )
                torch.testing.assert_close(
                    chunked.delta_residuals,
                    reference.delta_residuals,
                    rtol=2e-5,
                    atol=2e-6,
                )
                torch.testing.assert_close(
                    chunked.final_state,
                    reference.final_state,
                    rtol=2e-5,
                    atol=2e-6,
                )

    def test_surviving_token_updates_reconstruct_final_state(self):
        torch.manual_seed(9)
        key = torch.randn(1, 13, 2, 4)
        value = torch.randn(1, 13, 2, 3)
        beta = torch.sigmoid(torch.randn(1, 13, 2))
        log_decay = -torch.rand(1, 13, 2)
        trace = chunk_gated_delta_trace(
            key,
            value,
            beta,
            log_decay,
            chunk_size=4,
        )
        reverse_cumulative = torch.flip(
            torch.cumsum(torch.flip(log_decay, (1,)), dim=1),
            (1,),
        )
        log_survival_after_token = reverse_cumulative - log_decay
        reconstructed = torch.einsum(
            "bthk,bthv->bhkv",
            trace.normalized_keys,
            trace.delta_residuals * log_survival_after_token.exp().unsqueeze(-1),
        )
        torch.testing.assert_close(
            reconstructed,
            trace.final_state,
            rtol=2e-5,
            atol=2e-6,
        )

    def test_positive_log_decay_fails_closed(self):
        with self.assertRaisesRegex(RecurrentSignalError, "non-positive"):
            chunk_gated_delta_trace(
                torch.ones(1, 2, 1, 1),
                torch.ones(1, 2, 1, 1),
                torch.full((1, 2, 1), 0.5),
                torch.tensor([[[-0.1], [0.1]]]),
            )


class CandidateSemanticsTests(unittest.TestCase):
    @staticmethod
    def trace(delta_values, log_decay_values):
        delta = torch.tensor(delta_values, dtype=torch.float32).reshape(1, -1, 1, 1)
        log_decay = torch.tensor(log_decay_values, dtype=torch.float32).reshape(1, -1, 1)
        return DeltaRuleTrace(
            normalized_keys=torch.ones_like(delta),
            delta_residuals=delta,
            log_decay=log_decay,
            final_state=torch.zeros(1, 1, 1, 1),
        )

    def test_exp_g_retention_direction_and_suffix_sum(self):
        weak = summarize_recurrent_trace(
            self.trace([1, 1, 1, 1], [-0.1, -0.1, -0.1, -0.1]),
            layer_idx=0,
            segment_length=2,
        )
        strong = summarize_recurrent_trace(
            self.trace([1, 1, 1, 1], [-1.0, -1.0, -1.0, -1.0]),
            layer_idx=0,
            segment_length=2,
        )
        self.assertAlmostEqual(weak.log_survival[0], -0.2, places=6)
        self.assertAlmostEqual(strong.log_survival[0], -2.0, places=6)
        self.assertGreater(math.exp(weak.log_survival[0]), math.exp(strong.log_survival[0]))
        self.assertLess(weak.decay_risk[0], strong.decay_risk[0])
        self.assertEqual(weak.log_survival[-1], 0.0)
        self.assertEqual(weak.decay_risk[-1], 0.0)

    def test_destructive_suffix_has_positive_interference(self):
        destructive = summarize_recurrent_trace(
            self.trace([1, 1, -1, -1], [0, 0, 0, 0]),
            layer_idx=0,
            segment_length=2,
        )
        aligned = summarize_recurrent_trace(
            self.trace([1, 1, 1, 1], [0, 0, 0, 0]),
            layer_idx=0,
            segment_length=2,
        )
        self.assertAlmostEqual(destructive.suffix_interference[0], 1.0)
        self.assertAlmostEqual(aligned.suffix_interference[0], -1.0)
        self.assertEqual(destructive.suffix_interference[-1], 0.0)

    def test_partial_tail_is_retained_and_flagged(self):
        signal = summarize_recurrent_trace(
            self.trace([1, 2, 3, 4, 5], [-0.1] * 5),
            layer_idx=4,
            segment_length=2,
        )
        self.assertEqual(signal.segment_starts, (0, 2, 4))
        self.assertEqual(signal.segment_ends, (2, 4, 5))
        self.assertEqual(signal.partial_segments, (False, False, True))
        self.assertEqual(signal.n_segments, 3)

    def test_tiny_partial_tail_is_excluded_like_sigma_current(self):
        signal = summarize_recurrent_trace(
            self.trace([1] * 17, [-0.1] * 17),
            layer_idx=4,
            segment_length=8,
        )
        self.assertEqual(signal.segment_starts, (0, 8))
        self.assertEqual(signal.segment_ends, (8, 16))
        self.assertEqual(signal.partial_segments, (False, False))

    def test_layer_aggregation_is_frozen_mean(self):
        first = summarize_recurrent_trace(
            self.trace([1, 1, 1, 1], [-0.1] * 4),
            layer_idx=1,
            segment_length=2,
        )
        second_raw = summarize_recurrent_trace(
            self.trace([3, 3, 3, 3], [-0.3] * 4),
            layer_idx=3,
            segment_length=2,
        )
        aggregated = aggregate_recurrent_candidates({1: first, 3: second_raw})
        self.assertEqual(aggregated.layer_indices, (1, 3))
        self.assertAlmostEqual(
            aggregated.delta_update[0],
            (first.delta_update_rms[0] + second_raw.delta_update_rms[0]) / 2,
        )
        self.assertAlmostEqual(aggregated.survival_retention[0], -0.4)
        self.assertAlmostEqual(aggregated.decay_risk[0], 0.4)

    def test_layer_key_or_boundary_mismatch_fails_closed(self):
        signal = summarize_recurrent_trace(
            self.trace([1, 1, 1, 1], [-0.1] * 4),
            layer_idx=1,
            segment_length=2,
        )
        with self.assertRaisesRegex(RecurrentSignalError, "layer index key"):
            aggregate_recurrent_candidates({7: signal})
        mismatch = LayerRecurrentCandidates(
            layer_idx=2,
            segment_starts=(0,),
            segment_ends=(4,),
            partial_segments=(False,),
            delta_update_rms=(1.0,),
            log_survival=(0.0,),
            decay_risk=(0.0,),
            suffix_interference=(0.0,),
            surviving_write_norm=(1.0,),
        )
        with self.assertRaisesRegex(RecurrentSignalError, "boundaries"):
            aggregate_recurrent_candidates({1: signal, 2: mismatch})


class FakeDeltaNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_k_heads = 1
        self.num_v_heads = 1
        self.head_k_dim = 2
        self.head_v_dim = 2
        self.key_dim = 2
        self.value_dim = 2
        self.activation = "silu"
        self.causal_conv1d_fn = None
        self.in_proj_qkv = torch.nn.Linear(3, 6, bias=False)
        self.in_proj_b = torch.nn.Linear(3, 1, bias=False)
        self.in_proj_a = torch.nn.Linear(3, 1, bias=False)
        self.conv1d = torch.nn.Conv1d(6, 6, 1, groups=6, bias=False)
        self.A_log = torch.nn.Parameter(torch.zeros(1))
        self.dt_bias = torch.nn.Parameter(torch.zeros(1))
        with torch.no_grad():
            self.conv1d.weight.fill_(1.0)

    def forward(self, hidden_states, cache_params=None, attention_mask=None):
        del cache_params, attention_mask
        return hidden_states


class FakeCache:
    def __init__(self, previous=False):
        self.previous = previous

    def has_previous_state(self, layer_idx):
        del layer_idx
        return self.previous


class HookContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.module = FakeDeltaNet()
        layer = SimpleNamespace(linear_attn=self.module)
        model = SimpleNamespace(layers=[layer])
        self.model = SimpleNamespace(model=model)
        self.manager = Qwen35RecurrentCandidateHookManager(
            self.model,
            [0],
            segment_length=2,
        )

    def tearDown(self):
        self.manager.remove()

    def test_one_fresh_context_capture_then_detach(self):
        self.manager.attach()
        hidden = torch.randn(1, 5, 3)
        self.module(hidden_states=hidden, cache_params=FakeCache(False))
        signals = self.manager.finalize_context()
        self.assertEqual(tuple(signals), (0,))
        self.assertEqual(signals[0].segment_ends, (2, 4, 5))
        before = signals[0]
        self.module(hidden_states=torch.randn(1, 1, 3), cache_params=FakeCache(True))
        self.assertEqual(self.manager.get_signals()[0], before)

    def test_second_capture_before_detach_fails_closed(self):
        self.manager.attach()
        self.module(hidden_states=torch.randn(1, 4, 3), cache_params=FakeCache(False))
        with self.assertRaisesRegex(RecurrentSignalError, "more than once"):
            self.module(hidden_states=torch.randn(1, 1, 3), cache_params=FakeCache(True))

    def test_cached_or_padded_capture_fails_closed(self):
        self.manager.attach()
        with self.assertRaisesRegex(RecurrentSignalError, "fresh context"):
            self.module(hidden_states=torch.randn(1, 2, 3), cache_params=FakeCache(True))
        self.manager.attach()
        with self.assertRaisesRegex(RecurrentSignalError, "padded"):
            self.module(
                hidden_states=torch.randn(1, 2, 3),
                cache_params=FakeCache(False),
                attention_mask=torch.tensor([[1, 0]]),
            )

    def test_duplicate_layer_indices_fail_closed(self):
        with self.assertRaisesRegex(RecurrentSignalError, "unique"):
            Qwen35RecurrentCandidateHookManager(
                self.model,
                [0, 0],
                segment_length=2,
            )

    def test_missing_layer_capture_fails_closed(self):
        self.manager.attach()
        with self.assertRaisesRegex(RecurrentSignalError, "missing"):
            self.manager.finalize_context()


class SignalManifestTests(unittest.TestCase):
    def test_protocol_freezes_formula_and_aggregation(self):
        protocol = recurrent_signal_protocol()
        self.assertEqual(protocol["version"], P0C_PROTOCOL_VERSION)
        self.assertIn("exp(g_t)", protocol["state_update"])
        self.assertEqual(protocol["normalization"], "none_at_collection_discovery_fit_only")
        self.assertEqual(
            protocol["candidates"]["suffix_interference"]["layer_aggregation"],
            "mean",
        )

    def test_run_spec_embeds_signal_protocol(self):
        run_spec = build_run_spec(
            experiment="p0c_test",
            args={"seed": 42},
            selections={},
            model={},
            code={},
            environment={},
        )
        self.assertEqual(
            run_spec["recurrent_signals"]["version"],
            P0C_PROTOCOL_VERSION,
        )


if __name__ == "__main__":
    unittest.main()
