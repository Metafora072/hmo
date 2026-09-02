import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from experiments.phase2.e3_v2.oracle import OracleContractError, SegmentSpec
from experiments.phase2.e3_v2.recurrent_signals import AggregatedRecurrentCandidates
from experiments.phase2.e3_v2.run_discovery import (
    EVALUATED_CANDIDATES,
    analyze_discovery,
    build_segment_evidence,
    load_pair_observations,
)
from experiments.phase2.e3_v2.statistics import SegmentEvidence


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
