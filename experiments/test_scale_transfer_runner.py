"""CPU contract tests for the frozen 9B scale-transfer runner."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.phase2.e3_v2.run_scale_transfer import (
    SYSTEMS,
    load_scale_transfer_protocol,
    summarize_scale_transfer_results,
)


def _protocol() -> dict:
    return {
        "schema_version": "hmo.contiguous_cf.scale_transfer_protocol.v1",
        "status": "frozen_before_outcomes",
        "model_id": "Qwen/Qwen3.5-9B",
        "model_revision": "revision",
        "systems": list(SYSTEMS),
        "primary_comparison": [
            "contiguous_cf",
            "raw_alpha_exact_slack",
        ],
        "primary_metric": "normalized_answer_contains",
        "max_new_tokens": 32,
        "inference_seed": 20261010,
        "method": {
            "allocator": "attention_led",
            "middle_kv_fraction": 0.1,
            "sparse_selector": "max_mass_window",
            "sparse_width": 16,
            "raw_slack_selector": "global_top_tokens_slack",
            "protected_prefix_segments": 1,
            "protected_suffix_segments": 1,
        },
        "stages": {
            "smoke": {
                "datasets": "needle",
                "samples_per_dataset": 1,
                "context_length": 8192,
                "segment_length": 256,
                "middle_kv_fraction": 0.1,
                "seed": 20261011,
                "sample_id_prefix": "scale9b_smoke_",
            },
            "8k": {
                "datasets": "needle,longeval_lines",
                "samples_per_dataset": 6,
                "context_length": 8192,
                "segment_length": 256,
                "middle_kv_fraction": 0.1,
                "seed": 20261012,
                "sample_id_prefix": "scale9b_8k_",
            },
            "16k": {
                "datasets": "needle,longeval_lines",
                "samples_per_dataset": 6,
                "context_length": 16384,
                "segment_length": 256,
                "middle_kv_fraction": 0.1,
                "seed": 20261013,
                "sample_id_prefix": "scale9b_16k_",
            },
        },
    }


class ScaleTransferProtocolTests(unittest.TestCase):
    def test_protocol_accepts_frozen_9b_transfer_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            path.write_text(json.dumps(_protocol()), encoding="utf-8")
            protocol, digest = load_scale_transfer_protocol(path)
        self.assertEqual(protocol["model_id"], "Qwen/Qwen3.5-9B")
        self.assertEqual(len(digest), 64)

    def test_summary_reports_primary_comparison_and_equal_bytes(self):
        rows = []
        for index, stage in enumerate(("8k", "16k")):
            systems = {}
            for system in SYSTEMS:
                score = float(
                    system in {"contiguous_cf", "full_kv_reference"}
                    or (index == 1 and system == "raw_alpha_exact_slack")
                )
                systems[system] = {
                    "normalized_answer_contains": score,
                    "normalized_exact_match": score,
                    "token_f1": score,
                    "post_query_resident_kv_bytes": (
                        1000 if system == "full_kv_reference" else 100
                    ),
                }
            systems["raw_alpha_exact_topk"]["post_query_resident_kv_bytes"] = 90
            rows.append(
                {
                    "stage": stage,
                    "dataset": "needle",
                    "systems": systems,
                }
            )
        summary = summarize_scale_transfer_results(rows)
        primary = summary["comparisons"][
            "contiguous_cf_vs_raw_alpha_exact_slack"
        ]["normalized_answer_contains"]
        self.assertEqual(primary["wins"], 1)
        self.assertEqual(primary["ties"], 1)
        self.assertEqual(summary["equal_compressed_resident_byte_cases"], 2)


if __name__ == "__main__":
    unittest.main()
