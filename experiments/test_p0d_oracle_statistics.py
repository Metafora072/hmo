"""No-GPU integrity tests for E3-v2 P0-D oracle and statistics."""
from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.phase2.e3_v2.alpha_probe import collect_isolated_query_alpha
from experiments.phase2.e3_v2.context_query import (
    PostInterventionState,
    TokenizedPromptSplit,
)
from experiments.phase2.e3_v2.integrity import (
    REQUIRED_INTEGRITY_CHECKS,
    IntegrityCheck,
    require_integrity_gate,
)
from experiments.phase2.e3_v2.oracle import (
    ArmQuality,
    OracleConfig,
    OracleContractError,
    OracleManifestError,
    PairObservation,
    SegmentSpec,
    aggregate_pair_observations,
    audit_equal_byte_pair,
    build_oracle_plan,
    build_pair_observation,
    build_segment_catalog,
    ensure_oracle_manifest,
    load_oracle_manifest,
    make_oracle_intervention,
)
from experiments.phase2.e3_v2.protocol import P0D_PROTOCOL_VERSION, oracle_protocol
from experiments.phase2.e3_v2.statistics import (
    CandidateCVResult,
    SegmentEvidence,
    alpha_bin_pairwise_accuracy,
    evaluate_candidate_grouped_cv,
    ndcg_at_k,
    pairwise_ranking_accuracy,
    residual_correlation,
    sample_grouped_bootstrap_interval,
    select_discovery_candidate,
    spearman_correlation,
)
from experiments.utils.eval_harness import PromptTextParts
from experiments.utils.run_manifest import build_run_spec


class FakeAttentionLayer:
    def __init__(self, tokens, offset=0):
        values = torch.arange(offset, offset + tokens, dtype=torch.float32)
        values = values.reshape(1, 1, tokens, 1).repeat(1, 2, 1, 3)
        self.keys = values.clone()
        self.values = values.clone() + 0.5


class FakeRecurrentLayer:
    def __init__(self):
        self.recurrent_states = torch.tensor([17.0])


class FakeCache:
    def __init__(self, tokens, attention_layers=(0, 1)):
        max_index = max(attention_layers)
        self.layers = [None] * (max_index + 2)
        for offset, layer_index in enumerate(attention_layers):
            self.layers[layer_index] = FakeAttentionLayer(tokens, offset * 100)
        for index, layer in enumerate(self.layers):
            if layer is None:
                self.layers[index] = FakeRecurrentLayer()


def append_fake_query(cache, attention_layers, query_tokens):
    for layer_index in attention_layers:
        layer = cache.layers[layer_index]
        shape = (*layer.keys.shape[:-2], query_tokens, layer.keys.shape[-1])
        layer.keys = torch.cat([layer.keys, torch.zeros(shape)], dim=-2)
        layer.values = torch.cat([layer.values, torch.zeros(shape)], dim=-2)


def make_plan():
    attention_layers = (0, 1)
    context_tokens = 20
    cache = FakeCache(context_tokens, attention_layers)
    config = OracleConfig(
        segment_length=2,
        middle_kv_fraction=0.5,
        donors_per_segment=2,
        backgrounds_per_pair=2,
        position_bins=4,
        seed=11,
    )
    catalog = build_segment_catalog(
        cache,
        attention_layers,
        context_tokens=context_tokens,
        config=config,
    )
    plan = build_oracle_plan(
        sample_id="sample-1",
        context_tokens=context_tokens,
        attention_layer_indices=attention_layers,
        segments=catalog,
        config=config,
    )
    return plan


class OraclePlanningTests(unittest.TestCase):
    def test_plan_is_deterministic_balanced_and_equal_byte(self):
        first = make_plan()
        second = make_plan()
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.protocol_version, P0D_PROTOCOL_VERSION)
        self.assertEqual(first.middle_budget_slots, 4)
        self.assertGreater(len(first.comparisons), 0)

        pair_counts = {}
        degree = {segment_id: set() for segment_id in first.eligible_segment_ids}
        for comparison in first.comparisons:
            pair = (comparison.target_segment, comparison.donor_segment)
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
            degree[pair[0]].add(pair[1])
            degree[pair[1]].add(pair[0])
            self.assertEqual(
                comparison.target_exact_middle,
                tuple(sorted((*comparison.background_segments, comparison.target_segment))),
            )
            self.assertEqual(
                comparison.donor_exact_middle,
                tuple(sorted((*comparison.background_segments, comparison.donor_segment))),
            )
            self.assertEqual(comparison.middle_charged_bytes, first.middle_budget_slots * 96 * 2)
        self.assertTrue(all(count == 2 for count in pair_counts.values()))
        self.assertTrue(all(len(donors) >= 2 for donors in degree.values()))

    def test_planner_rejects_multiple_identical_empty_backgrounds(self):
        plan = make_plan()
        config = OracleConfig(
            segment_length=2,
            middle_kv_fraction=0.13,
            donors_per_segment=2,
            backgrounds_per_pair=2,
            seed=11,
        )
        with self.assertRaisesRegex(OracleContractError, "multiple backgrounds"):
            build_oracle_plan(
                sample_id=plan.sample_id,
                context_tokens=plan.context_tokens,
                attention_layer_indices=plan.attention_layer_indices,
                segments=plan.segments,
                config=config,
            )

    def test_manifest_roundtrip_is_immutable_and_hash_bound(self):
        plan = make_plan()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oracle_manifest.json"
            first = ensure_oracle_manifest(path, plan)
            second = ensure_oracle_manifest(path, plan)
            self.assertEqual(first, second)
            loaded = load_oracle_manifest(path)
            self.assertEqual(loaded.to_dict(), plan.to_dict())

            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["middle_budget_slots"] += 1
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(OracleManifestError):
                load_oracle_manifest(path)
            with self.assertRaises(OracleManifestError):
                ensure_oracle_manifest(path, plan)

    def test_manifest_semantics_are_recomputed_after_hash_validation(self):
        plan = make_plan()
        comparison = plan.comparisons[0]
        tampered = replace(
            plan,
            comparisons=(
                replace(
                    comparison,
                    context_resident_bytes=comparison.context_resident_bytes + 1,
                ),
                *plan.comparisons[1:],
            ),
        )
        raw = tampered.to_dict()
        with self.assertRaisesRegex(OracleContractError, "resident context bytes"):
            type(plan).from_dict(raw)


class EqualByteInterventionTests(unittest.TestCase):
    def build_state(self, plan, comparison_id, arm):
        cache = FakeCache(plan.context_tokens, plan.attention_layer_indices)
        recurrent_index = max(plan.attention_layer_indices) + 1
        recurrent_before = cache.layers[recurrent_index].recurrent_states.clone()
        context_ids = torch.arange(plan.context_tokens).unsqueeze(0)
        result = make_oracle_intervention(plan, comparison_id, arm)(cache, context_ids)
        torch.testing.assert_close(
            cache.layers[recurrent_index].recurrent_states,
            recurrent_before,
        )
        append_fake_query(cache, plan.attention_layer_indices, query_tokens=3)
        return PostInterventionState(
            cache=cache,
            first_answer_logits=torch.zeros(1, 8),
            context_tokens=plan.context_tokens,
            query_tokens=3,
            logical_position=plan.context_tokens + 3,
            resident_kv_tokens=result.active_context_positions.numel() + 3,
            active_context_positions=result.active_context_positions,
            intervention=result,
            events=(),
        )

    def test_target_and_donor_have_equal_charged_and_resident_bytes(self):
        plan = make_plan()
        comparison = plan.comparisons[0]
        target = self.build_state(plan, comparison.comparison_id, "target")
        donor = self.build_state(plan, comparison.comparison_id, "donor")
        audit = audit_equal_byte_pair(target, donor, plan.attention_layer_indices)
        self.assertEqual(audit.middle_charged_bytes, comparison.middle_charged_bytes)
        self.assertEqual(audit.context_resident_bytes, comparison.context_resident_bytes)
        self.assertEqual(
            target.active_context_positions.numel(),
            donor.active_context_positions.numel(),
        )
        self.assertNotEqual(
            target.active_context_positions.tolist(),
            donor.active_context_positions.tolist(),
        )

    def test_post_query_byte_mismatch_fails_closed(self):
        plan = make_plan()
        comparison = plan.comparisons[0]
        target = self.build_state(plan, comparison.comparison_id, "target")
        donor = self.build_state(plan, comparison.comparison_id, "donor")
        append_fake_query(donor.cache, plan.attention_layer_indices, query_tokens=1)
        with self.assertRaisesRegex(OracleContractError, "decode-resident"):
            audit_equal_byte_pair(target, donor, plan.attention_layer_indices)

    def test_manifest_context_mismatch_fails_before_mutation(self):
        plan = make_plan()
        comparison = plan.comparisons[0]
        cache = FakeCache(plan.context_tokens, plan.attention_layer_indices)
        with self.assertRaisesRegex(OracleContractError, "context tokens"):
            make_oracle_intervention(plan, comparison.comparison_id, "target")(
                cache,
                torch.zeros(1, plan.context_tokens - 1, dtype=torch.long),
            )
        self.assertEqual(cache.layers[0].keys.shape[-2], plan.context_tokens)


class ObservationAggregationTests(unittest.TestCase):
    def test_backgrounds_reduce_before_signed_segment_utility(self):
        plan = make_plan()
        observations = []
        latent = {segment_id: float(segment_id) for segment_id in plan.eligible_segment_ids}
        for comparison in plan.comparisons:
            observations.append(
                build_pair_observation(
                    plan,
                    comparison.comparison_id,
                    ArmQuality(latent[comparison.target_segment], 0.8),
                    ArmQuality(latent[comparison.donor_segment], 0.5),
                )
            )
        pairs, utility = aggregate_pair_observations(plan, observations)
        self.assertTrue(all(pair.background_count == 2 for pair in pairs))
        self.assertEqual(set(utility), set(plan.eligible_segment_ids))
        lowest = min(plan.eligible_segment_ids)
        highest = max(plan.eligible_segment_ids)
        self.assertLess(utility[lowest], utility[highest])

    def test_incomplete_or_unrecoverable_observations_fail_closed(self):
        plan = make_plan()
        comparison = plan.comparisons[0]
        valid = build_pair_observation(
            plan,
            comparison.comparison_id,
            ArmQuality(1.0),
            ArmQuality(0.0),
        )
        with self.assertRaisesRegex(OracleContractError, "incomplete"):
            aggregate_pair_observations(plan, [valid])
        invalid = PairObservation(
            **{
                **valid.__dict__,
                "background_segments": (999,),
            }
        )
        with self.assertRaisesRegex(OracleContractError, "recovered"):
            aggregate_pair_observations(plan, [invalid], require_complete=False)


class FakeAlphaBase:
    def __init__(self, trace, attention_layers=(0, 1), emit_attentions=True):
        self.trace = trace
        self.attention_layers = attention_layers
        self.emit_attentions = emit_attentions
        self.config = SimpleNamespace(_attn_implementation="sdpa")

    def __call__(
        self,
        input_ids,
        *,
        past_key_values=None,
        use_cache=True,
        output_attentions=False,
        position_ids=None,
        cache_position=None,
        return_dict=True,
    ):
        del use_cache, position_ids, cache_position, return_dict
        if past_key_values is None:
            cache = FakeCache(input_ids.shape[1], self.attention_layers)
            self.trace.append(("context", id(cache)))
            return SimpleNamespace(past_key_values=cache)
        cache = past_key_values
        self.trace.append(("query", id(cache)))
        query_tokens = input_ids.shape[1]
        context_tokens = cache.layers[self.attention_layers[0]].keys.shape[-2]
        append_fake_query(cache, self.attention_layers, query_tokens)
        attentions = [None] * (max(self.attention_layers) + 1)
        if output_attentions and self.emit_attentions:
            token_weights = torch.arange(1, context_tokens + 1, dtype=torch.float32)
            full = torch.cat([token_weights, torch.zeros(query_tokens)])
            weights = full.reshape(1, 1, 1, -1).repeat(1, 2, query_tokens, 1)
            for layer_index in self.attention_layers:
                attentions[layer_index] = weights.clone()
        return SimpleNamespace(
            past_key_values=cache,
            attentions=tuple(attentions) if self.emit_attentions else None,
        )


class FakeAlphaModel:
    def __init__(self, emit_attentions=True):
        self.device = torch.device("cpu")
        self.trace = []
        self.config = SimpleNamespace(
            _attn_implementation="sdpa",
            text_config=SimpleNamespace(_attn_implementation="sdpa"),
        )
        self.model = FakeAlphaBase(self.trace, emit_attentions=emit_attentions)


class AlphaIsolationTests(unittest.TestCase):
    def make_prompt_and_segments(self):
        context = torch.arange(20).unsqueeze(0)
        query = torch.tensor([[21, 22, 23]])
        prompt = TokenizedPromptSplit(
            text=PromptTextParts("context", "query"),
            context_ids=context,
            query_ids=query,
            full_ids=torch.cat([context, query], dim=1),
            split_token_index=20,
        )
        plan = make_plan()
        return prompt, plan.segments

    def test_alpha_uses_private_full_kv_cache_and_restores_config(self):
        prompt, segments = self.make_prompt_and_segments()
        model = FakeAlphaModel()
        result = collect_isolated_query_alpha(
            model,
            prompt,
            attention_layer_indices=[0, 1],
            segments=segments,
        )
        self.assertEqual(model.trace[0][0], "context")
        self.assertEqual(model.trace[1], ("query", model.trace[0][1]))
        self.assertEqual(model.config._attn_implementation, "sdpa")
        self.assertEqual(model.config.text_config._attn_implementation, "sdpa")
        self.assertEqual(model.model.config._attn_implementation, "sdpa")
        self.assertFalse(hasattr(result, "cache"))
        self.assertAlmostEqual(result.attention_mass[0], 1.0 + 2.0)
        self.assertAlmostEqual(result.attention_mass[1], 3.0 + 4.0)

        model.model(prompt.context_ids, use_cache=True, return_dict=True)
        self.assertNotEqual(model.trace[-1][1], model.trace[0][1])

    def test_alpha_failure_still_restores_attention_backend(self):
        prompt, segments = self.make_prompt_and_segments()
        model = FakeAlphaModel(emit_attentions=False)
        with self.assertRaisesRegex(OracleContractError, "no attention"):
            collect_isolated_query_alpha(
                model,
                prompt,
                attention_layer_indices=[0, 1],
                segments=segments,
            )
        self.assertEqual(model.config._attn_implementation, "sdpa")
        self.assertEqual(model.model.config._attn_implementation, "sdpa")


class StatisticalAnalysisTests(unittest.TestCase):
    def test_ranking_ndcg_and_spearman_primitives(self):
        utility = [0.0, 1.0, 3.0, 2.0]
        self.assertEqual(pairwise_ranking_accuracy(utility, utility), 1.0)
        self.assertEqual(pairwise_ranking_accuracy([-x for x in utility], utility), 0.0)
        self.assertAlmostEqual(ndcg_at_k(utility, utility, 2), 1.0)
        self.assertAlmostEqual(spearman_correlation(utility, utility), 1.0)

    def test_residual_and_alpha_bin_association(self):
        alpha = np.repeat([0.1, 0.2, 0.8, 0.9], 3)
        position = np.tile([0.1, 0.5, 0.9], 4)
        candidate = np.asarray(
            [-1.0, 1.0, 0.0, 0.0, -1.0, 1.0, 1.0, 0.0, -1.0, -1.0, 0.0, 1.0]
        )
        utility = 4 * alpha + 2 * position + 3 * candidate
        self.assertGreater(
            residual_correlation(candidate, utility, alpha, position),
            0.99,
        )
        self.assertEqual(
            alpha_bin_pairwise_accuracy(candidate, utility, alpha, bins=4),
            1.0,
        )

    def test_bootstrap_is_deterministic_and_sample_grouped(self):
        values = {f"s{index}": float(index - 2) for index in range(6)}
        first = sample_grouped_bootstrap_interval(values, n_bootstrap=200, seed=5)
        second = sample_grouped_bootstrap_interval(values, n_bootstrap=200, seed=5)
        self.assertEqual(first, second)
        self.assertEqual(first.n_samples, 6)
        self.assertLessEqual(first.lower, first.mean)
        self.assertGreaterEqual(first.upper, first.mean)

    def test_grouped_cv_detects_incremental_candidate_and_stratifies_tasks(self):
        rng = np.random.default_rng(12)
        evidence = []
        k_by_sample = {}
        for sample_index in range(10):
            sample_id = f"sample-{sample_index}"
            dataset = "task-a" if sample_index < 5 else "task-b"
            k_by_sample[sample_id] = 2
            for segment_id in range(7):
                candidate = rng.normal()
                alpha = rng.normal()
                position = segment_id / 6
                utility = 2.5 * candidate + 0.05 * rng.normal()
                evidence.append(
                    SegmentEvidence(
                        sample_id=sample_id,
                        dataset=dataset,
                        segment_id=segment_id,
                        utility=utility,
                        alpha=alpha,
                        normalized_position=position,
                        candidates={"delta_update": candidate},
                    )
                )
        result = evaluate_candidate_grouped_cv(
            evidence,
            "delta_update",
            k_by_sample=k_by_sample,
            folds=5,
            bootstrap_samples=200,
            seed=9,
        )
        self.assertGreater(result.pairwise_improvement.mean, 0.3)
        self.assertGreater(result.ndcg_improvement.mean, 0.2)
        self.assertEqual(set(result.task_pairwise_improvement), {"task-a", "task-b"})

        weaker = CandidateCVResult(
            candidate="weaker",
            baseline_metrics=result.baseline_metrics,
            augmented_metrics=result.baseline_metrics,
            pairwise_improvement=sample_grouped_bootstrap_interval(
                {f"s{i}": 0.0 for i in range(3)}, n_bootstrap=10
            ),
            ndcg_improvement=sample_grouped_bootstrap_interval(
                {f"s{i}": 0.0 for i in range(3)}, n_bootstrap=10
            ),
            task_pairwise_improvement={},
            task_ndcg_improvement={},
        )
        self.assertEqual(select_discovery_candidate([weaker, result]).candidate, "delta_update")


class IntegrityGateTests(unittest.TestCase):
    def passing_checks(self):
        return dict(
            (name, IntegrityCheck(True, f"{name} evidence"))
            for name in REQUIRED_INTEGRITY_CHECKS
        )

    def test_all_eight_checks_are_required_and_serializable(self):
        report = require_integrity_gate(self.passing_checks())
        payload = report.to_dict()
        self.assertEqual(payload["status"], "pass")
        self.assertEqual(tuple(payload["checks"]), REQUIRED_INTEGRITY_CHECKS)

    def test_missing_or_failed_check_blocks_scientific_execution(self):
        missing = self.passing_checks()
        missing.pop(REQUIRED_INTEGRITY_CHECKS[-1])
        with self.assertRaisesRegex(OracleContractError, "check set mismatch"):
            require_integrity_gate(missing)

        failed = self.passing_checks()
        failed["controlled_needle_logit_effect"] = IntegrityCheck(False, "no effect")
        with self.assertRaisesRegex(OracleContractError, "controlled_needle"):
            require_integrity_gate(failed)

    def test_evidence_is_mandatory(self):
        checks = self.passing_checks()
        checks["alpha_isolation"] = IntegrityCheck(True, " ")
        with self.assertRaisesRegex(OracleContractError, "requires.*evidence"):
            require_integrity_gate(checks)


class P0DManifestProtocolTests(unittest.TestCase):
    def test_run_spec_pins_oracle_protocol(self):
        protocol = oracle_protocol()
        self.assertEqual(protocol["version"], P0D_PROTOCOL_VERSION)
        self.assertEqual(protocol["quality"]["background_reduction"], "mean_per_unordered_pair_before_ranking")
        run_spec = build_run_spec(
            experiment="p0d_test",
            args={"seed": 42},
            selections={},
            model={},
            code={},
            environment={},
        )
        self.assertEqual(run_spec["oracle"]["version"], P0D_PROTOCOL_VERSION)


if __name__ == "__main__":
    unittest.main()
