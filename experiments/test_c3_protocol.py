"""CPU contract tests for the frozen C3 27B package."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.phase2.e3_v2.c3_protocol import (
    C3_MODEL_REVISION,
    C3ProtocolError,
    load_c3_protocol,
    native_protocol_view,
    pareto_protocol_view,
)
from experiments.phase2.e3_v2.run_native_tasks import load_native_protocol
from experiments.phase2.e3_v2.run_pareto import (
    _manifest_stage_spec,
    load_pareto_protocol,
    resolve_pareto_stage_set,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = PROJECT_ROOT / "refine-logs/c3_27b_protocol.json"


class C3ProtocolTests(unittest.TestCase):
    def test_frozen_package_has_expected_cell_counts(self):
        payload, digest, parent = load_c3_protocol(PROTOCOL, PROJECT_ROOT)
        self.assertEqual(len(digest), 64)
        self.assertEqual(payload["mandatory_core"]["total_generation_cells"], 432)
        formal = payload["synthetic"]["stages"]["formal_32k"]
        self.assertEqual(formal["expected_generation_cells"], 312)
        native_cases = sum(
            len(spec["cases"]) for spec in parent["datasets"].values()
        )
        self.assertEqual(native_cases, 24)

    def test_runner_views_pin_27b_and_reuse_native_cases(self):
        payload, _, parent = load_c3_protocol(PROTOCOL, PROJECT_ROOT)
        pareto = pareto_protocol_view(payload)
        native = native_protocol_view(payload, parent)
        self.assertEqual(pareto["model_revision"], C3_MODEL_REVISION)
        self.assertEqual(
            pareto["stage_sets"]["central"]["budget_fractions"], [0.1]
        )
        self.assertEqual(native["model_revision"], C3_MODEL_REVISION)
        self.assertEqual(native["datasets"], parent["datasets"])

    def test_existing_runners_accept_the_master_protocol(self):
        pareto, pareto_digest = load_pareto_protocol(PROTOCOL)
        native, native_digest = load_native_protocol(PROTOCOL)
        self.assertEqual(pareto_digest, native_digest)
        self.assertEqual(pareto["model_revision"], C3_MODEL_REVISION)
        self.assertEqual(native["model_revision"], C3_MODEL_REVISION)
        central = resolve_pareto_stage_set(pareto, "central")
        side = resolve_pareto_stage_set(pareto, "side")
        self.assertEqual(central[0], ("formal_32k",))
        self.assertEqual(central[3], (0.1,))
        self.assertEqual(len(central[2]), 4)
        self.assertEqual(side[3], (0.05, 0.2))
        self.assertEqual(
            _manifest_stage_spec(pareto, "central"),
            ("formal", (0.05, 0.1, 0.2)),
        )
        self.assertEqual(
            _manifest_stage_spec(pareto, "side"),
            ("formal", (0.05, 0.1, 0.2)),
        )

    def test_cell_count_tampering_is_rejected(self):
        payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        payload["mandatory_core"]["total_generation_cells"] = 431
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            path = Path(directory) / "c3.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(C3ProtocolError):
                load_c3_protocol(path, PROJECT_ROOT)


if __name__ == "__main__":
    unittest.main()
