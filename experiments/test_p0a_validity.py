"""No-GPU validity tests for P0-A metrics and provenance gates."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.phase2.runner import ExperimentCell, save_cell
from experiments.utils.dataset_utils import EvalSample
from experiments.utils.eval_harness import score_prediction
from experiments.utils.run_manifest import (
    ManifestMismatchError,
    UnmanagedResultDirectoryError,
    ensure_run_manifest,
)


def sample(dataset: str, answer: str, answers: list[str] | None = None) -> EvalSample:
    return EvalSample(
        dataset=dataset,
        sample_id="case-1",
        context="context",
        question="question",
        answer=answer,
        answers=answers,
        context_length=128,
    )


class OfficialMetricTests(unittest.TestCase):
    def test_qa_f1_preserves_duplicate_token_counts(self):
        scores = score_prediction(
            "red red red",
            sample("longbench_hotpotqa", "red red"),
        )
        self.assertEqual(scores.primary_metric, "f1")
        self.assertAlmostEqual(scores.f1, 0.8)
        self.assertAlmostEqual(scores.primary_score, 0.8)

    def test_longbench_uses_max_over_all_ground_truths(self):
        scores = score_prediction(
            "target answer",
            sample("longbench_narrativeqa", "wrong", ["wrong", "target answer"]),
        )
        self.assertEqual(scores.primary_score, 1.0)

    def test_lcc_extracts_first_non_comment_code_line(self):
        prediction = "\n```python\n# explanation\nvalue = total + 1\nignored"
        scores = score_prediction(
            prediction,
            sample("longbench_lcc", "value = total + 1"),
        )
        self.assertEqual(scores.primary_metric, "code_sim")
        self.assertEqual(scores.code_sim, 1.0)
        self.assertEqual(scores.primary_score, 1.0)

    def test_govreport_uses_official_rouge_l(self):
        scores = score_prediction(
            "a concise summary",
            sample("longbench_gov_report", "a concise summary"),
        )
        self.assertEqual(scores.primary_metric, "rouge_l")
        self.assertAlmostEqual(scores.rouge_l, 1.0)
        self.assertAlmostEqual(scores.primary_score, 1.0)

    def test_unknown_longbench_metric_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "No pinned official"):
            score_prediction("answer", sample("longbench_unknown", "answer"))


class ImmutableManifestTests(unittest.TestCase):
    code = {"commit": "a" * 40, "branch": "test", "dirty": False}
    environment = {"python": "3.11", "platform": "test", "packages": {}}
    model = {
        "alias": "test-model",
        "registry_path": "test-model",
        "revision": "model-revision",
    }

    def create(
        self,
        run_dir: Path,
        args: dict | None = None,
        environment: dict | None = None,
    ):
        return ensure_run_manifest(
            run_dir,
            experiment="p0a_test",
            args=args or {"seed": 42, "resume": False},
            selections={"benchmarks": ["longbench_hotpotqa"]},
            model=self.model,
            project_root=Path("."),
            code_provenance=self.code,
            environment=environment or self.environment,
            argv=["test"],
        )

    def test_identical_resume_reuses_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            first = self.create(run_dir)
            before = (run_dir / "run_manifest.json").read_bytes()
            second = self.create(run_dir, {"seed": 42, "resume": True})
            after = (run_dir / "run_manifest.json").read_bytes()
            self.assertEqual(first["manifest_id"], second["manifest_id"])
            self.assertEqual(before, after)

    def test_changed_scientific_argument_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            self.create(run_dir)
            with self.assertRaises(ManifestMismatchError):
                self.create(run_dir, {"seed": 7, "resume": True})

    def test_changed_environment_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            self.create(run_dir)
            with self.assertRaises(ManifestMismatchError):
                self.create(
                    run_dir,
                    environment={"python": "3.12", "platform": "test", "packages": {}},
                )

    def test_nonempty_unmanaged_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "old_results.jsonl").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(UnmanagedResultDirectoryError):
                self.create(run_dir)

    def test_result_rows_are_bound_to_manifest_id(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            manifest = self.create(run_dir)
            output_path = run_dir / "results.jsonl"
            save_cell(ExperimentCell(sample_id="case-1"), output_path)
            row = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(row["manifest_id"], manifest["manifest_id"])
            with self.assertRaisesRegex(ValueError, "does not match"):
                save_cell(ExperimentCell(manifest_id="wrong"), output_path)

    def test_manifest_id_covers_pinned_metric_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.create(Path(directory))
            longbench = manifest["run_spec"]["metrics"]["longbench"]
            self.assertEqual(len(longbench["revision"]), 40)
            self.assertEqual(longbench["datasets"]["longbench_lcc"], "code_sim_score")


if __name__ == "__main__":
    unittest.main()
