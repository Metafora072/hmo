"""Official prompt, metric, and generation contracts for six LongBench QA tasks."""
from __future__ import annotations

import unittest

from experiments.utils.dataset_utils import EvalSample
from experiments.utils.eval_harness import (
    _build_raw_prompt_parts,
    get_primary_metric_name,
    resolve_max_new_tokens,
)


EXPECTED_MAX = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
}


class LongBenchSixTaskProtocolTests(unittest.TestCase):
    def test_all_tasks_use_registered_context_boundary_and_qa_f1(self):
        for name, maximum in EXPECTED_MAX.items():
            sample = EvalSample(
                dataset=f"longbench_{name}",
                sample_id=name,
                context="UNIQUE_CONTEXT",
                question="UNIQUE_QUESTION",
                answer="answer",
                context_length=0,
            )
            parts = _build_raw_prompt_parts(sample)
            self.assertEqual(parts.full_prompt.count("UNIQUE_CONTEXT"), 1)
            self.assertIn("UNIQUE_QUESTION", parts.query_suffix)
            self.assertEqual(get_primary_metric_name(sample.dataset), "f1")
            self.assertEqual(resolve_max_new_tokens(sample.dataset, 7), maximum)


if __name__ == "__main__":
    unittest.main()
