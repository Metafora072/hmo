"""Contracts for query-accessibility enrichment and fixed scores."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.phase2.e3_v2.enrich_query_accessibility import (
    evaluate_accessibility_scores,
    load_frozen_v2_config,
    load_prospective_protocol,
    replace_query_probe_rows,
    validate_prospective_stage,
)
from experiments.phase2.e3_v2.oracle import OracleContractError
from experiments.phase2.e3_v2.statistics import SegmentEvidence


class QueryAccessibilityEnrichmentTests(unittest.TestCase):
    def _frozen_v2(self):
        return {
            "schema_version": "hmo.dual_confidence_abstention.v2",
            "status": "frozen",
            "need_score": "alpha*(1-rank01(query_read_share))",
            "normalized_alpha_entropy_threshold": 0.45,
            "alpha_access_spearman_threshold": 0.75,
            "threshold_search_after_freeze": False,
            "task_identity_used_at_inference": False,
            "oracle_labels_used_at_inference": False,
        }

    def test_frozen_v2_and_prospective_stage_are_hashed_and_aligned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            method_path = root / "method.json"
            method_path.write_text(json.dumps(self._frozen_v2()), encoding="utf-8")
            _, method_hash = load_frozen_v2_config(method_path)
            stage = {
                "datasets": "needle,longeval_lines",
                "samples_per_dataset": 6,
                "context_length": 8192,
                "segment_length": 256,
                "middle_kv_fraction": 0.1,
                "donors_per_segment": 2,
                "backgrounds_per_pair": 1,
                "seed": 19,
                "sample_id_prefix": "fresh_",
            }
            protocol_path = root / "protocol.json"
            protocol_path.write_text(
                json.dumps(
                    {
                        "schema_version": (
                            "hmo.query_accessibility.prospective_protocol.v1"
                        ),
                        "status": "frozen_before_outcomes",
                        "frozen_v2_sha256": method_hash,
                        "stages": {"8k": stage, "16k": {**stage, "context_length": 16384}},
                    }
                ),
                encoding="utf-8",
            )
            protocol, protocol_hash = load_prospective_protocol(protocol_path)
        validate_prospective_stage(
            {"scope": "prospective_oracle", **stage},
            protocol,
            "8k",
            frozen_v2_sha256=method_hash,
        )
        self.assertEqual(len(method_hash), 64)
        self.assertEqual(len(protocol_hash), 64)

    def test_prospective_stage_rejects_parameter_drift(self):
        stage = {
            "datasets": "needle,longeval_lines",
            "samples_per_dataset": 6,
            "context_length": 8192,
            "segment_length": 256,
            "middle_kv_fraction": 0.1,
            "donors_per_segment": 2,
            "backgrounds_per_pair": 1,
            "seed": 19,
            "sample_id_prefix": "fresh_",
        }
        protocol = {
            "frozen_v2_sha256": "a" * 64,
            "stages": {"8k": stage},
        }
        with self.assertRaisesRegex(OracleContractError, "disagrees"):
            validate_prospective_stage(
                {"scope": "prospective_oracle", **stage, "seed": 20},
                protocol,
                "8k",
                frozen_v2_sha256="a" * 64,
            )

    def test_replacement_preserves_labels_and_adds_query_candidates(self):
        raw = [
            {
                "dataset": "needle",
                "segment_id": 1,
                "utility": 0.7,
                "alpha": 0.2,
                "normalized_position": 0.3,
                "candidates": {
                    "sigma_current": 0.5,
                    "delta_update": 0.25,
                    "phi_sigma_alpha": 0.1,
                    "phi_delta_alpha": 0.05,
                },
            }
        ]
        result = replace_query_probe_rows(
            raw,
            {0: 0.1, 1: 0.8},
            {0: 0.2, 1: 0.4},
            {0: 0.3, 1: 0.6},
            {0: 0.4, 1: -0.2},
            sample_id="new",
        )
        self.assertEqual(result[0].sample_id, "new")
        self.assertEqual(result[0].utility, 0.7)
        self.assertEqual(result[0].alpha, 0.8)
        self.assertAlmostEqual(result[0].candidates["alpha_read_share"], 0.48)
        self.assertAlmostEqual(result[0].candidates["phi_delta_alpha"], 0.2)

    def test_replacement_rejects_missing_segments(self):
        with self.assertRaisesRegex(OracleContractError, "disagree"):
            replace_query_probe_rows(
                [{"segment_id": 1}],
                {1: 0.2},
                {1: 0.2},
                {},
                {1: 0.2},
                sample_id="bad",
            )

    def test_direct_scores_report_budget_membership_changes(self):
        evidence = []
        for sample_id in ("a", "b"):
            for segment_id, (alpha, access, utility) in enumerate(
                ((0.9, 0.9, 0.1), (0.8, 0.1, 1.0), (0.1, 0.5, 0.0))
            ):
                evidence.append(
                    SegmentEvidence(
                        sample_id=sample_id,
                        dataset="needle",
                        segment_id=segment_id,
                        utility=utility,
                        alpha=alpha,
                        normalized_position=segment_id / 2,
                        candidates={"query_read_share": access},
                    )
                )
        result = evaluate_accessibility_scores(
            evidence,
            {"a": 1, "b": 1},
            bootstrap_samples=20,
            seed=5,
        )
        deficit = result["methods"]["access_deficit"]
        self.assertEqual(deficit["topk_changed_samples"], 2)
        self.assertGreater(deficit["ndcg_improvement"]["mean"], 0)
        self.assertLess(
            result["methods"]["access_excess"]["ndcg_improvement"]["mean"],
            deficit["ndcg_improvement"]["mean"],
        )


if __name__ == "__main__":
    unittest.main()
