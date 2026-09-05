"""Contract tests for the frozen 9B HMO-versus-ChunkKV mechanism run."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.phase2.e3_v2.oracle import OracleContractError
from experiments.phase2.e3_v2.run_pareto import (
    CHUNKKV_SCALE_PROTOCOL_SCHEMA,
    _manifest_stage_spec,
    load_pareto_protocol,
    resolve_pareto_stage_set,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    PROJECT_ROOT
    / "refine-logs/chunkkv_mechanism_transfer_9b_protocol.json"
)


class ChunkKVMechanismTransferTests(unittest.TestCase):
    def test_protocol_reuses_frozen_scale_samples(self):
        payload, digest = load_pareto_protocol(PROTOCOL)
        self.assertEqual(payload["schema_version"], CHUNKKV_SCALE_PROTOCOL_SCHEMA)
        self.assertEqual(len(digest), 64)
        stages, systems, equal_bytes, budgets = resolve_pareto_stage_set(
            payload, "formal"
        )
        self.assertEqual(stages, ("8k", "16k"))
        self.assertEqual(
            systems,
            ("contiguous_cf", "chunkkv", "full_kv_reference"),
        )
        self.assertEqual(equal_bytes, ("contiguous_cf", "chunkkv"))
        self.assertEqual(budgets, (0.1,))
        self.assertEqual(payload["execution"]["formal_sample_cases"], 24)
        self.assertEqual(payload["execution"]["expected_generation_cells"], 72)
        self.assertEqual(
            _manifest_stage_spec(payload, "formal"),
            ("formal", (0.1,)),
        )

    def test_parent_protocol_hash_tampering_is_rejected(self):
        payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        payload["parent_protocol_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            path = Path(directory) / "protocol.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(OracleContractError):
                load_pareto_protocol(path)


if __name__ == "__main__":
    unittest.main()
