from __future__ import annotations

import unittest

from experiments.phase2.e3_v2.freeze_free_window_dev_protocol import DATASET_ORDER
from experiments.phase2.e3_v2.freeze_layer_local_completion_protocol import build_cases
from experiments.phase2.e3_v2.run_layer_local_completion import summarize


def _source_row(dataset: str, index: int) -> dict:
    return {
        "dataset": dataset,
        "sample_id": f"{dataset}_{index}",
        "record_index": index,
        "record_sha256": f"sha-{dataset}-{index}",
        "context_tokens": 100 + index,
        "query_tokens": 4,
        "query_probe": {"probe_id": f"probe-{dataset}-{index}"},
    }


class LayerLocalCompletionTests(unittest.TestCase):
    def test_build_cases_covers_parents_and_assigns_rank_strata(self):
        source = [
            _source_row(dataset, index)
            for dataset in DATASET_ORDER
            for index in range(8)
        ]
        dev = [row for row in source if row["record_index"] in {1, 6}]
        cases = build_cases(source, dev, strata=4)
        self.assertEqual(len(cases), 48)
        self.assertEqual(
            sum(case["layer_local_execution"] == "reuse_sha_pinned_development" for case in cases),
            12,
        )
        for dataset in DATASET_ORDER:
            members = [case for case in cases if case["dataset"] == dataset]
            self.assertEqual(
                [sum(case["length_stratum"] == value for case in members) for value in range(4)],
                [2, 2, 2, 2],
            )

    def test_summary_reports_dataset_strata_and_equal_bytes(self):
        rows = []
        for dataset in DATASET_ORDER:
            for stratum in range(4):
                systems = {}
                for system, score in (
                    ("hmo_legacy", 0.2),
                    ("hmo_layer_local", 0.3),
                    ("chunkkv", 0.25),
                    ("full_kv_reference", 0.4),
                ):
                    systems[system] = {
                        "official_qa_f1": score,
                        "normalized_answer_contains": 0.0,
                        "normalized_exact_match": 0.0,
                        "post_query_resident_kv_bytes": 500 if system == "full_kv_reference" else 100,
                        "generated_token_ids": [1, 2],
                    }
                rows.append(
                    {
                        "dataset": dataset,
                        "length_stratum": stratum,
                        "context_tokens": 1000 + stratum,
                        "max_new_tokens": 8,
                        "systems": systems,
                    }
                )
        result = summarize(rows)
        self.assertEqual(result["overall"]["case_count"], 24)
        self.assertEqual(result["overall"]["equal_resident_byte_cases"], 24)
        self.assertEqual(set(result["by_dataset"]), set(DATASET_ORDER))
        self.assertEqual(set(result["by_length_stratum"]), {"0", "1", "2", "3"})
        self.assertAlmostEqual(
            result["overall"]["comparisons"]["hmo_layer_local_vs_chunkkv"]
            ["official_qa_f1"]["mean_delta"],
            0.05,
        )


if __name__ == "__main__":
    unittest.main()
