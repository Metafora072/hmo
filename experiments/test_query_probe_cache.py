"""Contracts for persistent query-probe artifacts."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from experiments.phase2.e3_v2.alpha_probe import AlphaProbeResult
from experiments.phase2.e3_v2.context_query import TokenizedPromptSplit
from experiments.phase2.e3_v2.oracle import OracleContractError, SegmentSpec
from experiments.phase2.e3_v2.query_accessibility import (
    HybridQueryTokenProbeResult,
    QueryAccessibilityResult,
)
from experiments.phase2.e3_v2.query_probe_cache import (
    QUERY_PROBE_AGGREGATION,
    QUERY_PROBE_CACHE_SCHEMA,
    get_or_create_query_probe,
    retained_positions_sha256,
)
from experiments.utils.eval_harness import PromptTextParts


def _segments() -> tuple[SegmentSpec, ...]:
    return tuple(
        SegmentSpec(
            segment_id=index,
            start=index * 2,
            end=(index + 1) * 2,
            token_count=2,
            kv_bytes=16,
            protected=index in {0, 2},
            partial=False,
            normalized_position=(index + 0.5) / 3,
            position_bin=index,
        )
        for index in range(3)
    )


def _prompt(query_ids: tuple[int, ...] = (7, 8)) -> TokenizedPromptSplit:
    context = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    query = torch.tensor([list(query_ids)], dtype=torch.long)
    return TokenizedPromptSplit(
        text=PromptTextParts("context", "query"),
        context_ids=context,
        query_ids=query,
        full_ids=torch.cat([context, query], dim=1),
        split_token_index=context.shape[1],
    )


def _result() -> HybridQueryTokenProbeResult:
    segment_ids = (0, 1, 2)
    return HybridQueryTokenProbeResult(
        alpha=AlphaProbeResult(
            context_tokens=6,
            query_tokens=2,
            attention_layer_indices=(1,),
            segment_ids=segment_ids,
            attention_mass=(0.3, 0.7, 1.1),
        ),
        accessibility=QueryAccessibilityResult(
            context_tokens=6,
            query_tokens=2,
            recurrent_layer_indices=(0, 2),
            segment_ids=segment_ids,
            read_norm=(1.0, 2.0, 3.0),
            read_share=(0.2, 0.3, 0.5),
            read_alignment=(0.5, 0.6, 0.7),
        ),
        token_attention_mass=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
    )


MODEL_IDENTITY = {
    "model_id": "Qwen/test",
    "revision": "a" * 40,
    "config_sha256": "b" * 64,
    "weight_index_sha256": "c" * 64,
    "weight_files": [{"name": "model.safetensors", "sha256": "d" * 64}],
}


class QueryProbeCacheTests(unittest.TestCase):
    def _call(self, directory: Path, collector, prompt=None):
        return get_or_create_query_probe(
            directory,
            model_identity=MODEL_IDENTITY,
            prompt=prompt or _prompt(),
            attention_layer_indices=(1,),
            recurrent_layer_indices=(0, 2),
            segments=_segments(),
            segment_length=2,
            collector=collector,
        )

    def test_collects_once_then_reuses_exact_artifact(self):
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            first = self._call(
                Path(directory), lambda: calls.append("collect") or _result()
            )
            second = self._call(
                Path(directory),
                lambda: self.fail("cache hit must not recollect the probe"),
            )
            self.assertEqual(calls, ["collect"])
            self.assertFalse(first.cache_hit)
            self.assertTrue(second.cache_hit)
            self.assertEqual(first.probe_id, second.probe_id)
            self.assertEqual(first.token_scores_sha256, second.token_scores_sha256)
            self.assertEqual(first.result, second.result)
            self.assertEqual(
                first.result.alpha.attention_mass,
                (
                    float(torch.tensor(0.1).item() + torch.tensor(0.2).item()),
                    float(torch.tensor(0.3).item() + torch.tensor(0.4).item()),
                    float(torch.tensor(0.5).item() + torch.tensor(0.6).item()),
                ),
            )
            self.assertEqual(
                first.provenance()["aggregation_version"],
                QUERY_PROBE_AGGREGATION,
            )

    def test_retained_position_hash_is_stable_and_order_sensitive(self):
        self.assertEqual(
            retained_positions_sha256((0, 4, 9)),
            retained_positions_sha256([0, 4, 9]),
        )
        self.assertNotEqual(
            retained_positions_sha256((0, 4, 9)),
            retained_positions_sha256((0, 9, 4)),
        )
        with self.assertRaisesRegex(OracleContractError, "nonnegative"):
            retained_positions_sha256((0, -1))

    def test_prompt_boundary_changes_probe_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self._call(Path(directory), _result)
            second = self._call(Path(directory), _result, prompt=_prompt((7, 9)))
            self.assertNotEqual(first.probe_id, second.probe_id)

    def test_corrupt_score_hash_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            cached = self._call(Path(directory), _result)
            cached.score_path.write_bytes(b"corrupt")
            with self.assertRaisesRegex(
                OracleContractError, "cannot load persistent query probe"
            ):
                self._call(Path(directory), _result)

    def test_metadata_identity_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            cached = self._call(Path(directory), _result)
            metadata = json.loads(cached.metadata_path.read_text(encoding="utf-8"))
            metadata["identity"]["query_tokens"] = 99
            cached.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(
                OracleContractError, "identity or hash validation"
            ):
                self._call(Path(directory), _result)

    def test_incomplete_artifact_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            cached = self._call(Path(directory), _result)
            cached.metadata_path.unlink()
            with self.assertRaisesRegex(OracleContractError, "incomplete"):
                self._call(Path(directory), _result)

    def test_metadata_records_schema_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            cached = self._call(Path(directory), _result)
            metadata = json.loads(cached.metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["schema_version"], QUERY_PROBE_CACHE_SCHEMA)
            self.assertEqual(
                metadata["token_scores_sha256"], cached.token_scores_sha256
            )
            self.assertEqual(metadata["token_scores_dtype"], "float32")


if __name__ == "__main__":
    unittest.main()
