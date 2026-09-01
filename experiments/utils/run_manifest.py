"""Immutable provenance manifests for formal experiment runs."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.vendor.longbench_metrics import (
    LONG_BENCH_REVISION,
    LONG_BENCH_SOURCE,
)

MANIFEST_FILENAME = "run_manifest.json"
MANIFEST_SCHEMA_VERSION = 1
OPERATIONAL_ARGS = frozenset({"resume"})


class ManifestError(RuntimeError):
    """Base class for run-manifest contract violations."""


class DirtyRepositoryError(ManifestError):
    pass


class ManifestMismatchError(ManifestError):
    pass


class UnmanagedResultDirectoryError(ManifestError):
    pass


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _run_git(project_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=project_root, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    return result.stdout.strip()


def collect_code_provenance(project_root: Path, require_clean: bool = True) -> dict[str, Any]:
    try:
        commit = _run_git(project_root, "rev-parse", "HEAD")
        branch = _run_git(project_root, "branch", "--show-current") or None
        dirty_entries = _run_git(
            project_root,
            "status", "--porcelain", "--untracked-files=all", "--", ".",
            ":(exclude)experiments/results/**",
        ).splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise ManifestError(f"Cannot collect Git provenance from {project_root}") from exc

    if require_clean and dirty_entries:
        preview = ", ".join(entry[:120] for entry in dirty_entries[:5])
        raise DirtyRepositoryError(
            "Formal runs require a clean Git worktree; commit the implementation first. "
            f"Dirty entries: {preview}"
        )
    return {"commit": commit, "branch": branch, "dirty": bool(dirty_entries)}


def collect_environment() -> dict[str, Any]:
    packages = {}
    for package in (
        "torch", "transformers", "accelerate", "datasets", "numpy", "scipy",
        "compressed-tensors", "gptqmodel", "triton", "fuzzywuzzy",
        "python-Levenshtein", "rouge",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    runtime = {"cuda_available": False}
    try:
        import torch
        runtime["cuda_build"] = torch.version.cuda
        runtime["cuda_available"] = torch.cuda.is_available()
        if runtime["cuda_available"]:
            runtime["visible_devices"] = [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
    except Exception as exc:
        runtime["probe_error"] = type(exc).__name__

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
        "runtime": runtime,
    }


def metric_protocol() -> dict[str, Any]:
    from experiments.phase2.e3_v2.protocol import (
        P0B_ANSWER_PREFIX,
        P0B_EXECUTION_EVENTS,
        P0B_PROTOCOL_VERSION,
    )

    return {
        "longbench": {
            "revision": LONG_BENCH_REVISION,
            "source": LONG_BENCH_SOURCE,
            "ground_truth_reduction": "max",
            "datasets": {
                "longbench_hotpotqa": "qa_f1_score",
                "longbench_narrativeqa": "qa_f1_score",
                "longbench_qasper": "qa_f1_score",
                "longbench_gov_report": "rouge_score",
                "longbench_lcc": "code_sim_score",
            },
        },
        "synthetic": {
            "needle": "normalized_answer_contains",
            "longeval_lines": "normalized_answer_contains",
        },
        "post_intervention": {
            "version": P0B_PROTOCOL_VERSION,
            "prompt_split": "single_tokenization_verified_offset_boundary",
            "execution_order": list(P0B_EXECUTION_EVENTS),
            "primary_quality": "mean_gold_answer_logprob_per_token",
            "answer_prefix": P0B_ANSWER_PREFIX,
            "rotary_positions": "original_logical_positions",
            "causal_mask_positions": "resident_kv_order",
        },
    }


def normalize_args(args: Namespace | Mapping[str, Any]) -> dict[str, Any]:
    values = vars(args) if isinstance(args, Namespace) else dict(args)
    return {
        key.replace("-", "_"): value
        for key, value in sorted(values.items())
        if key not in OPERATIONAL_ARGS
    }


def build_run_spec(
    *,
    experiment: str,
    args: Namespace | Mapping[str, Any],
    selections: Mapping[str, Any],
    model: Mapping[str, Any],
    code: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    from experiments.phase2.e3_v2.protocol import recurrent_signal_protocol

    return {
        "experiment": experiment,
        "arguments": normalize_args(args),
        "selections": dict(selections),
        "model": dict(model),
        "code": dict(code),
        "environment": dict(environment),
        "metrics": metric_protocol(),
        "recurrent_signals": recurrent_signal_protocol(),
    }


def _existing_artifacts(run_dir: Path) -> list[str]:
    if not run_dir.exists():
        return []
    return sorted(
        path.name for path in run_dir.iterdir()
        if path.name != MANIFEST_FILENAME and not path.name.startswith(".")
    )


def _describe_mismatch(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> str:
    keys = sorted(set(expected) | set(actual))
    changed = [key for key in keys if expected.get(key) != actual.get(key)]
    return ", ".join(changed) or "unknown"


def ensure_run_manifest(
    run_dir: Path,
    *,
    experiment: str,
    args: Namespace | Mapping[str, Any],
    selections: Mapping[str, Any],
    model: Mapping[str, Any],
    project_root: Path,
    require_clean: bool = True,
    code_provenance: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
    argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Create once or validate an identical immutable run manifest."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / MANIFEST_FILENAME
    code = dict(code_provenance or collect_code_provenance(project_root, require_clean))
    environment_signature = dict(environment or collect_environment())
    spec = build_run_spec(
        experiment=experiment,
        args=args,
        selections=selections,
        model=model,
        code=code,
        environment=environment_signature,
    )
    manifest_id = hashlib.sha256(_canonical_json(spec).encode("utf-8")).hexdigest()

    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as handle:
            existing = json.load(handle)
        if (
            existing.get("schema_version") != MANIFEST_SCHEMA_VERSION
            or existing.get("run_spec") != spec
            or existing.get("manifest_id") != manifest_id
        ):
            changed = _describe_mismatch(existing.get("run_spec", {}), spec)
            raise ManifestMismatchError(
                f"Refusing to reuse {run_dir}: immutable manifest differs in {changed}"
            )
        return existing

    artifacts = _existing_artifacts(run_dir)
    if artifacts:
        raise UnmanagedResultDirectoryError(
            f"Refusing to attach a manifest to non-empty {run_dir}; existing artifacts: "
            + ", ".join(artifacts[:10])
        )

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_id": manifest_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_spec": spec,
        "invocation": list(argv if argv is not None else sys.argv),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        with manifest_path.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        return ensure_run_manifest(
            run_dir,
            experiment=experiment,
            args=args,
            selections=selections,
            model=model,
            project_root=project_root,
            require_clean=require_clean,
            code_provenance=code,
            environment=environment_signature,
            argv=argv,
        )
    return manifest


def read_manifest_id(run_dir: Path) -> str:
    path = Path(run_dir) / MANIFEST_FILENAME
    if not path.exists():
        raise ManifestError(f"Missing {MANIFEST_FILENAME} in formal run directory {run_dir}")
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    return str(manifest["manifest_id"])
