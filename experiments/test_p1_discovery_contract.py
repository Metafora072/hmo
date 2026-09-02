import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments.phase2.e3_v2.oracle import OracleContractError, SegmentSpec
from experiments.phase2.e3_v2.recurrent_signals import AggregatedRecurrentCandidates
from experiments.phase2.e3_v2.run_discovery import (
    EVALUATED_CANDIDATES,
    _build_samples,
    analyze_discovery,
    build_segment_evidence,
    load_frozen_scorer_config,
    load_pair_observations,
)
from experiments.phase2.e3_v2.statistics import SegmentEvidence
from experiments.utils.dataset_utils import EvalSample


class ObservationResumeTests(unittest.TestCase):
    def test_load_pair_observations_roundtrip(self):
        row = {
            "oracle_manifest_id": "manifest",
            "sample_id": "sample",
            "comparison_id": "comparison",
            "target_segment": 1,
            "donor_segment": 2,
            "background_segments": [3],
            "delta_logprob": 0.25,
            "delta_secondary": None,
            "target_mean_gold_logprob": -1.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            loaded = load_pair_observations(path)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].background_segments, (3,))
        self.assertEqual(loaded[0].delta_logprob, 0.25)

    def test_duplicate_resume_row_fails_closed(self):
        row = {
            "oracle_manifest_id": "manifest",
            "sample_id": "sample",
            "comparison_id": "comparison",
            "target_segment": 1,
            "donor_segment": 2,
            "background_segments": [],
            "delta_logprob": 0.0,
            "delta_secondary": None,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            encoded = json.dumps(row) + "\n"
            path.write_text(encoded + encoded, encoding="utf-8")
            with self.assertRaises(OracleContractError):
                load_pair_observations(path)


class FrozenScorerConfigTests(unittest.TestCase):
    def _payload(self):
        return {
            "schema_version": "p1-bounded-additive-v1",
            "formula": "rank01(alpha)+lambda*(rank01(sigma_current)-0.5)",
            "feature": "sigma_current",
            "normalization": "within_sample_average_rank01",
            "selected_lambda": 0.15,
            "lambda_candidates": [-0.30, -0.15, 0.15, 0.30],
            "selection_rule": "max_mean_pairwise_then_ndcg_then_smaller_abs_lambda",
            "development": {
                "scope": "combined_discovery_only_not_confirmation",
                "sample_count": 12,
                "sources": [{"manifest_id": "development"}],
            },
        }

    def _controller_payload(self):
        return {
            "schema_version": "p1-conditional-rank-v1",
            "formula": "single_top_down_adjacent_regime_inversion_pass",
            "features": ["sigma_current", "delta_update"],
            "configuration": {
                "normalization": "within_sample_average_rank01",
                "threshold": 0.5,
                "threshold_search": False,
                "regime_priority": {"SAFE": 0, "NEUTRAL": 1, "STRESSED": 2},
                "rank_adjustment": {"SAFE": 1, "NEUTRAL": 0, "STRESSED": -1},
                "collision_policy": (
                    "swap_adjacent_alpha_ranks_when_lower_regime_priority_is_higher_"
                    "and_neither_segment_has_moved"
                ),
            },
            "development": {
                "scope": (
                    "combined_discovery_conditional_regime_only_not_confirmation"
                ),
                "sample_count": 12,
                "segment_count": 360,
                "pattern_supported": True,
                "sources": [{"manifest_id": "development"}],
            },
        }

    def test_valid_frozen_controller_is_hashed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "controller.json"
            path.write_text(json.dumps(self._controller_payload()), encoding="utf-8")
            loaded, digest = load_frozen_scorer_config(path)
        self.assertEqual(loaded["configuration"]["threshold"], 0.5)
        self.assertEqual(len(digest), 64)

    def test_controller_collision_policy_is_immutable(self):
        payload = self._controller_payload()
        payload["configuration"]["collision_policy"] = "global_sort"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "controller.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(OracleContractError):
                load_frozen_scorer_config(path)

    def test_valid_frozen_scorer_is_hashed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scorer.json"
            path.write_text(json.dumps(self._payload()), encoding="utf-8")
            loaded, digest = load_frozen_scorer_config(path)
        self.assertEqual(loaded["selected_lambda"], 0.15)
        self.assertEqual(len(digest), 64)

    def test_non_discovery_scorer_provenance_fails_closed(self):
        payload = self._payload()
        payload["development"]["scope"] = "held_out_confirmation"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scorer.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(OracleContractError):
                load_frozen_scorer_config(path)


class HeldOutSampleTests(unittest.TestCase):
    def test_confirmation_prefix_makes_sample_id_distinct(self):
        sample = EvalSample(
            dataset="needle",
            sample_id="needle_0000",
            context="context",
            question="question",
            answer="answer",
            context_length=8192,
        )
        args = SimpleNamespace(
            datasets="needle",
            seed=99,
            samples_per_dataset=1,
            context_length=8192,
            sample_id_prefix="confirm_seed99_",
        )
        with patch(
            "experiments.phase2.e3_v2.run_discovery.make_needle_samples",
            return_value=[sample],
        ):
            built = _build_samples(object(), args)
        self.assertEqual(built[0].sample_id, "confirm_seed99_needle_0000")
        self.assertEqual(sample.sample_id, "needle_0000")


class SegmentEvidenceTests(unittest.TestCase):
    def test_candidates_align_with_oracle_segments(self):
        segments = tuple(
            SegmentSpec(
                segment_id=index,
                start=index * 4,
                end=(index + 1) * 4,
                token_count=4,
                kv_bytes=16,
                protected=index in (0, 3),
                partial=False,
                normalized_position=(index + 0.5) / 4,
                position_bin=index,
            )
            for index in range(4)
        )
        plan = SimpleNamespace(
            sample_id="sample",
            eligible_segment_ids=(1, 2),
            segments=segments,
        )
        recurrent = AggregatedRecurrentCandidates(
            layer_indices=(0,),
            segment_starts=(0, 4, 8, 12),
            segment_ends=(4, 8, 12, 16),
            partial_segments=(False, False, False, False),
            delta_update=(1.0, 2.0, 3.0, 4.0),
            survival_retention=(-4.0, -3.0, -2.0, -1.0),
            decay_risk=(4.0, 3.0, 2.0, 1.0),
            suffix_interference=(0.0, 0.1, 0.2, 0.3),
            surviving_write_norm=(4.0, 3.0, 2.0, 1.0),
        )
        rows = build_segment_evidence(
            plan=plan,
            dataset="needle",
            utility={1: 0.5, 2: -0.5},
            alpha={1: 0.2, 2: 0.4},
            recurrent=recurrent,
            sigma_current=(0.0, 0.6, 0.8, 1.0),
        )
        self.assertEqual([row.segment_id for row in rows], [1, 2])
        self.assertAlmostEqual(rows[0].candidates["phi_sigma_alpha"], 0.12)
        self.assertAlmostEqual(rows[1].candidates["phi_delta_alpha"], 1.2)


class DiscoveryAnalysisTests(unittest.TestCase):
    def test_grouped_analysis_selects_incremental_candidate(self):
        rows = []
        patterns = (
            (0.0, 2.0, 1.0),
            (2.0, 0.0, 1.0),
            (1.0, 0.0, 2.0),
            (0.0, 1.0, 2.0),
        )
        for sample_index, pattern in enumerate(patterns):
            for segment_id, utility in enumerate(pattern):
                candidates = {name: 0.0 for name in EVALUATED_CANDIDATES}
                candidates["delta_update"] = utility
                rows.append(
                    SegmentEvidence(
                        sample_id=f"sample_{sample_index}",
                        dataset="synthetic",
                        segment_id=segment_id,
                        utility=utility,
                        alpha=0.0,
                        normalized_position=0.0,
                        candidates=candidates,
                    )
                )
        result = analyze_discovery(
            rows,
            {f"sample_{index}": 1 for index in range(4)},
            bootstrap_samples=100,
            seed=7,
        )
        self.assertEqual(result["selected_candidate"], "delta_update")
        self.assertEqual(result["direction"], "positive")
        self.assertGreater(
            result["candidate_results"]["delta_update"]
            ["pairwise_improvement"]["mean"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
