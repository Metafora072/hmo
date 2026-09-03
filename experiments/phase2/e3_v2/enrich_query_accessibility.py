"""Enrich completed oracle evidence with query-conditioned recurrent accessibility."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from experiments.phase2.e3_v2.direct_fusion import _rank01
from experiments.phase2.e3_v2.enrich_corrected_alpha import (
    _alpha_shift_summary,
    _read_json,
)
from experiments.phase2.e3_v2.context_query import tokenize_sample_prompt
from experiments.phase2.e3_v2.oracle import OracleContractError, load_oracle_manifest
from experiments.phase2.e3_v2.query_accessibility import collect_hybrid_query_probe
from experiments.phase2.e3_v2.real_model_preflight import (
    _force_torch_reference_backend,
    _load_model,
    model_provenance,
)
from experiments.phase2.e3_v2.run_discovery import _build_samples, _cleanup_cuda
from experiments.phase2.e3_v2.statistics import (
    SegmentEvidence,
    evaluate_candidate_grouped_cv,
    ndcg_at_k,
    pairwise_ranking_accuracy,
    sample_grouped_bootstrap_interval,
    spearman_correlation,
)
from experiments.utils.model_loader import (
    get_full_attention_indices,
    get_linear_attention_indices,
)


ACCESSIBILITY_CANDIDATES = (
    "query_read_norm",
    "query_read_share",
    "query_read_alignment",
    "alpha_read_share",
)
ALPHA_ENTROPY_THRESHOLD = 0.45
ALPHA_ACCESS_AGREEMENT_THRESHOLD = 0.75
FROZEN_V2_SCHEMA = "hmo.dual_confidence_abstention.v2"
PROSPECTIVE_PROTOCOL_SCHEMA = "hmo.query_accessibility.prospective_protocol.v1"


def load_frozen_v2_config(path: Path) -> tuple[dict, str]:
    try:
        encoded = path.read_bytes()
        payload = json.loads(encoded)
    except (OSError, json.JSONDecodeError) as exc:
        raise OracleContractError(f"cannot read frozen V2 config: {path}") from exc
    if (
        payload.get("schema_version") != FROZEN_V2_SCHEMA
        or payload.get("status") != "frozen"
        or payload.get("need_score")
        != "alpha*(1-rank01(query_read_share))"
        or payload.get("normalized_alpha_entropy_threshold")
        != ALPHA_ENTROPY_THRESHOLD
        or payload.get("alpha_access_spearman_threshold")
        != ALPHA_ACCESS_AGREEMENT_THRESHOLD
        or payload.get("threshold_search_after_freeze") is not False
        or payload.get("task_identity_used_at_inference") is not False
        or payload.get("oracle_labels_used_at_inference") is not False
    ):
        raise OracleContractError("frozen V2 configuration mismatch")
    return payload, hashlib.sha256(encoded).hexdigest()


def load_prospective_protocol(path: Path) -> tuple[dict, str]:
    try:
        encoded = path.read_bytes()
        payload = json.loads(encoded)
    except (OSError, json.JSONDecodeError) as exc:
        raise OracleContractError(f"cannot read prospective protocol: {path}") from exc
    if (
        payload.get("schema_version") != PROSPECTIVE_PROTOCOL_SCHEMA
        or payload.get("status") != "frozen_before_outcomes"
        or set(payload.get("stages", {})) != {"8k", "16k"}
    ):
        raise OracleContractError("prospective protocol mismatch")
    return payload, hashlib.sha256(encoded).hexdigest()


def validate_prospective_stage(
    source_args: Mapping,
    protocol: Mapping,
    stage_name: str,
    *,
    frozen_v2_sha256: str,
) -> None:
    if protocol.get("frozen_v2_sha256") != frozen_v2_sha256:
        raise OracleContractError("prospective protocol method hash mismatch")
    try:
        stage = protocol["stages"][stage_name]
    except KeyError as exc:
        raise OracleContractError("unknown prospective protocol stage") from exc
    expected = {
        "scope": "prospective_oracle",
        "datasets": stage["datasets"],
        "samples_per_dataset": stage["samples_per_dataset"],
        "context_length": stage["context_length"],
        "segment_length": stage["segment_length"],
        "middle_kv_fraction": stage["middle_kv_fraction"],
        "donors_per_segment": stage["donors_per_segment"],
        "backgrounds_per_pair": stage["backgrounds_per_pair"],
        "seed": stage["seed"],
        "sample_id_prefix": stage["sample_id_prefix"],
    }
    if any(source_args.get(key) != value for key, value in expected.items()):
        raise OracleContractError("source run disagrees with prospective protocol stage")


def replace_query_probe_rows(
    raw_rows: Sequence[Mapping],
    corrected_alpha: Mapping[int, float],
    read_norm: Mapping[int, float],
    read_share: Mapping[int, float],
    read_alignment: Mapping[int, float],
    *,
    sample_id: str,
) -> tuple[SegmentEvidence, ...]:
    raw_ids = {int(row["segment_id"]) for row in raw_rows}
    mappings = (corrected_alpha, read_norm, read_share, read_alignment)
    if not raw_ids or any(not raw_ids.issubset(values) for values in mappings):
        raise OracleContractError("query probe and segment evidence disagree")
    rows = []
    for raw in raw_rows:
        segment_id = int(raw["segment_id"])
        alpha = float(corrected_alpha[segment_id])
        share = float(read_share[segment_id])
        candidates = {
            str(name): float(value) for name, value in raw["candidates"].items()
        }
        candidates.update(
            {
                "query_read_norm": float(read_norm[segment_id]),
                "query_read_share": share,
                "query_read_alignment": float(read_alignment[segment_id]),
                "alpha_read_share": alpha * share,
                "phi_sigma_alpha": alpha * candidates["sigma_current"],
                "phi_delta_alpha": alpha * candidates["delta_update"],
            }
        )
        rows.append(
            SegmentEvidence(
                sample_id=sample_id,
                dataset=str(raw["dataset"]),
                segment_id=segment_id,
                utility=float(raw["utility"]),
                alpha=alpha,
                normalized_position=float(raw["normalized_position"]),
                candidates=candidates,
            )
        )
    return tuple(rows)


def normalized_entropy(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or np.any(array < 0):
        raise OracleContractError("normalized entropy requires nonnegative vector")
    total = float(array.sum())
    if not np.isfinite(total) or total <= 0:
        raise OracleContractError("normalized entropy requires positive finite mass")
    probability = array / total
    return float(
        -(probability * np.log(probability + 1e-30)).sum() / np.log(len(array))
    )


def evaluate_accessibility_scores(
    evidence: Sequence[SegmentEvidence],
    k_by_sample: Mapping[str, int],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    by_sample: dict[str, list[SegmentEvidence]] = {}
    for row in evidence:
        by_sample.setdefault(row.sample_id, []).append(row)
    if not by_sample or set(by_sample) != set(k_by_sample):
        raise OracleContractError("accessibility evidence and budgets disagree")

    sample_rows = {}
    for sample_id, rows in sorted(by_sample.items()):
        rows = sorted(rows, key=lambda row: row.segment_id)
        alpha = np.asarray([row.alpha for row in rows], dtype=np.float64)
        access = np.asarray(
            [row.candidates["query_read_share"] for row in rows],
            dtype=np.float64,
        )
        access_rank = _rank01(access)
        utility = np.asarray([row.utility for row in rows], dtype=np.float64)
        entropy = normalized_entropy(alpha)
        agreement = spearman_correlation(alpha, access)
        access_deficit = alpha * (1.0 - access_rank)
        dual_gate_enabled = (
            entropy >= ALPHA_ENTROPY_THRESHOLD
            and agreement < ALPHA_ACCESS_AGREEMENT_THRESHOLD
        )
        scores = {
            "alpha": alpha,
            "access_deficit": access_deficit,
            "access_excess": alpha * access_rank,
            "confidence_abstention": (
                access_deficit if entropy >= ALPHA_ENTROPY_THRESHOLD else alpha
            ),
            "dual_confidence_abstention": (
                access_deficit if dual_gate_enabled else alpha
            ),
        }
        k = k_by_sample[sample_id]
        baseline_topk = set(np.argsort(-alpha, kind="stable")[:k].tolist())
        sample_rows[sample_id] = {
            "dataset": rows[0].dataset,
            "normalized_alpha_entropy": entropy,
            "entropy_threshold": ALPHA_ENTROPY_THRESHOLD,
            "alpha_access_agreement": agreement,
            "agreement_threshold": ALPHA_ACCESS_AGREEMENT_THRESHOLD,
            "confidence_abstained": entropy < ALPHA_ENTROPY_THRESHOLD,
            "dual_confidence_abstained": not dual_gate_enabled,
            "dual_gate_enabled": dual_gate_enabled,
            "topk_changed": {
                name: set(np.argsort(-score, kind="stable")[:k].tolist())
                != baseline_topk
                for name, score in scores.items()
                if name != "alpha"
            },
            "metrics": {
                name: {
                    "pairwise_accuracy": pairwise_ranking_accuracy(score, utility),
                    "ndcg": ndcg_at_k(score, utility, k),
                }
                for name, score in scores.items()
            },
        }

    results = {}
    methods = (
        "access_deficit",
        "access_excess",
        "confidence_abstention",
        "dual_confidence_abstention",
    )
    for method_index, method in enumerate(methods):
        pairwise = {
            sample_id: row["metrics"][method]["pairwise_accuracy"]
            - row["metrics"]["alpha"]["pairwise_accuracy"]
            for sample_id, row in sample_rows.items()
        }
        ndcg = {
            sample_id: row["metrics"][method]["ndcg"]
            - row["metrics"]["alpha"]["ndcg"]
            for sample_id, row in sample_rows.items()
        }
        tasks = sorted({row["dataset"] for row in sample_rows.values()})
        results[method] = {
            "pairwise_improvement": asdict(
                sample_grouped_bootstrap_interval(
                    pairwise,
                    n_bootstrap=bootstrap_samples,
                    seed=seed + method_index * 2,
                )
            ),
            "ndcg_improvement": asdict(
                sample_grouped_bootstrap_interval(
                    ndcg,
                    n_bootstrap=bootstrap_samples,
                    seed=seed + method_index * 2 + 1,
                )
            ),
            "task_pairwise_improvement": {
                task: float(
                    np.mean(
                        [
                            pairwise[sample_id]
                            for sample_id, row in sample_rows.items()
                            if row["dataset"] == task
                        ]
                    )
                )
                for task in tasks
            },
            "task_ndcg_improvement": {
                task: float(
                    np.mean(
                        [
                            ndcg[sample_id]
                            for sample_id, row in sample_rows.items()
                            if row["dataset"] == task
                        ]
                    )
                )
                for task in tasks
            },
            "topk_changed_samples": sum(
                row["topk_changed"][method] for row in sample_rows.values()
            ),
        }
    return {
        "baseline": "corrected_raw_alpha",
        "formulae": {
            "access_deficit": "alpha*(1-rank01(query_read_share))",
            "access_excess": "alpha*rank01(query_read_share)",
            "confidence_abstention": (
                "access_deficit if H(alpha)/log(n)>=0.45 else alpha"
            ),
            "dual_confidence_abstention": (
                "access_deficit if entropy>=0.45 and "
                "spearman(alpha,access)<0.75 else alpha"
            ),
        },
        "methods": results,
        "samples": sample_rows,
    }


def _load_completed_summary(run_dir: Path) -> tuple[dict, str]:
    for filename in (
        "prospective_oracle_summary.json",
        "discovery_summary.json",
        "confirmation_summary.json",
    ):
        path = run_dir / filename
        if path.is_file():
            summary = _read_json(path)
            if summary.get("status") != "complete":
                raise OracleContractError(f"incomplete source run {run_dir}")
            return summary, filename
    raise OracleContractError(f"source run has no completed summary: {run_dir}")


def enrich(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("query accessibility enrichment requires exactly one visible GPU")
    frozen_v2 = None
    frozen_v2_sha256 = None
    prospective_protocol = None
    prospective_protocol_sha256 = None
    if getattr(args, "frozen_v2_config", None):
        if len(args.run_dir) != 1:
            raise OracleContractError("prospective V2 evaluation requires one source run")
        frozen_v2, frozen_v2_sha256 = load_frozen_v2_config(
            Path(args.frozen_v2_config).resolve()
        )
        if not getattr(args, "prospective_protocol", None) or not getattr(
            args, "protocol_stage", None
        ):
            raise OracleContractError(
                "prospective V2 evaluation requires a protocol and stage"
            )
        prospective_protocol, prospective_protocol_sha256 = (
            load_prospective_protocol(Path(args.prospective_protocol).resolve())
        )
    model_path = Path(args.model_path).resolve()
    identity = model_provenance(
        model_path,
        args.model_id,
        revision=args.model_revision,
    )
    model, tokenizer = _load_model(model_path)
    attention_layers = tuple(get_full_attention_indices(model.config))
    recurrent_layers = tuple(get_linear_attention_indices(model.config))
    _force_torch_reference_backend(model, recurrent_layers)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    evidence = []
    k_by_sample = {}
    sources = []
    alpha_shifts = {}
    for run_index, value in enumerate(args.run_dir):
        run_dir = Path(value).resolve()
        manifest = _read_json(run_dir / "run_manifest.json")
        summary, summary_file = _load_completed_summary(run_dir)
        if frozen_v2 is not None and summary.get("scope") != (
            "prospective_oracle_acquisition_only"
        ):
            raise OracleContractError(
                "prospective V2 evaluation requires a prospective oracle source"
            )
        if manifest["run_spec"]["model"] != identity:
            raise OracleContractError("source run model provenance mismatch")
        source_args = manifest["run_spec"]["arguments"]
        if frozen_v2 is not None:
            validate_prospective_stage(
                source_args,
                prospective_protocol,
                args.protocol_stage,
                frozen_v2_sha256=frozen_v2_sha256,
            )
        samples = _build_samples(
            tokenizer,
            Namespace(
                datasets=source_args["datasets"],
                samples_per_dataset=source_args["samples_per_dataset"],
                context_length=source_args["context_length"],
                seed=source_args["seed"],
                sample_id_prefix=source_args.get("sample_id_prefix", ""),
            ),
        )
        expected_by_id = {
            row["sample_id"]: row for row in summary["sample_summaries"]
        }
        namespace = f"run{run_index}_{manifest['manifest_id'][:8]}"
        for sample in samples:
            prompt = tokenize_sample_prompt(sample, tokenizer)
            expected = expected_by_id[sample.sample_id]
            if prompt.context_tokens != expected["context_tokens"]:
                raise OracleContractError("regenerated prompt disagrees with source run")
            plan = load_oracle_manifest(
                run_dir / "samples" / sample.sample_id / "oracle_manifest.json"
            )
            probe = collect_hybrid_query_probe(
                model,
                prompt,
                attention_layer_indices=attention_layers,
                recurrent_layer_indices=recurrent_layers,
                segments=plan.segments,
                segment_length=source_args["segment_length"],
            )
            alpha = probe.alpha.as_dict()
            access = probe.accessibility
            raw_rows = _read_json(
                run_dir / "samples" / sample.sample_id / "segment_evidence.json"
            )["rows"]
            namespaced_id = f"{namespace}:{sample.sample_id}"
            updated = replace_query_probe_rows(
                raw_rows,
                alpha,
                access.field_dict("read_norm"),
                access.field_dict("read_share"),
                access.field_dict("read_alignment"),
                sample_id=namespaced_id,
            )
            evidence.extend(updated)
            k = int(expected["middle_budget_slots"])
            alpha_shifts[namespaced_id] = _alpha_shift_summary(
                [float(row["alpha"]) for row in raw_rows],
                [row.alpha for row in updated],
                k=k,
            )
            k_by_sample[namespaced_id] = k
            print(f"captured query accessibility for {namespaced_id}", flush=True)
            _cleanup_cuda()
        sources.append(
            {
                "run_dir": str(run_dir),
                "manifest_id": manifest["manifest_id"],
                "summary_file": summary_file,
                "seed": source_args["seed"],
                "sample_count": len(samples),
            }
        )

    cv = {}
    if frozen_v2 is None:
        for index, candidate in enumerate(ACCESSIBILITY_CANDIDATES):
            cv[candidate] = asdict(
                evaluate_candidate_grouped_cv(
                    evidence,
                    candidate,
                    k_by_sample=k_by_sample,
                    folds=min(4, len(k_by_sample)),
                    bootstrap_samples=args.bootstrap_samples,
                    seed=args.seed + index * 100,
                )
            )
    shifts = list(alpha_shifts.values())
    direct_scores = evaluate_accessibility_scores(
        evidence,
        k_by_sample,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed + 1000,
    )
    if frozen_v2 is not None:
        method_name = "dual_confidence_abstention"
        direct_scores = {
            "baseline": direct_scores["baseline"],
            "formula": direct_scores["formulae"][method_name],
            "method": direct_scores["methods"][method_name],
            "samples": {
                sample_id: {
                    "dataset": row["dataset"],
                    "normalized_alpha_entropy": row["normalized_alpha_entropy"],
                    "alpha_access_agreement": row["alpha_access_agreement"],
                    "dual_gate_enabled": row["dual_gate_enabled"],
                    "topk_changed": row["topk_changed"][method_name],
                    "metrics": {
                        "alpha": row["metrics"]["alpha"],
                        method_name: row["metrics"][method_name],
                    },
                }
                for sample_id, row in direct_scores["samples"].items()
            },
        }
    return {
        "schema_version": (
            "hmo.query_accessibility_prospective_v2.v1"
            if frozen_v2 is not None
            else "hmo.query_accessibility_enrichment.v1"
        ),
        "status": "complete",
        "scope": (
            "prospective_frozen_v2_evaluation"
            if frozen_v2 is not None
            else "query_conditioned_recurrent_accessibility_reusing_oracle_labels"
        ),
        "frozen_v2_sha256": frozen_v2_sha256,
        "prospective_protocol_sha256": prospective_protocol_sha256,
        "protocol_stage": getattr(args, "protocol_stage", None),
        "sources": sources,
        "sample_count": len(k_by_sample),
        "model": identity,
        "alpha_shift": {
            "mean_spearman": float(np.mean([row["spearman"] for row in shifts])),
            "mean_topk_overlap": float(
                np.mean([row["topk_overlap"] for row in shifts])
            ),
            "argmax_changed_samples": sum(row["argmax_changed"] for row in shifts),
            "samples": alpha_shifts,
        },
        "candidate_cv": cv,
        "direct_scores": direct_scores,
        "k_by_sample": k_by_sample,
        "evidence": [asdict(row) for row in evidence],
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "gpu_name": torch.cuda.get_device_name(0),
            "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--model-revision")
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frozen-v2-config")
    parser.add_argument("--prospective-protocol")
    parser.add_argument("--protocol-stage", choices=("8k", "16k"))
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()
    if args.bootstrap_samples <= 0:
        parser.error("bootstrap-samples must be positive")
    prospective_args = (
        args.frozen_v2_config,
        args.prospective_protocol,
        args.protocol_stage,
    )
    if any(prospective_args) and not all(prospective_args):
        parser.error(
            "frozen V2 config, prospective protocol, and protocol stage are required together"
        )
    return args


def main() -> int:
    args = parse_args()
    payload = enrich(args)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {key: value for key, value in payload.items() if key != "evidence"}
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
