"""Prospective equal-byte generation validation for frozen accessibility V2."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from argparse import Namespace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from experiments.phase2.e3_v2.context_query import (
    InterventionResult,
    full_kv_intervention,
    generate_greedy,
    run_post_intervention_prompt,
    tokenize_sample_prompt,
)
from experiments.phase2.e3_v2.direct_fusion import _rank01
from experiments.phase2.e3_v2.enrich_query_accessibility import (
    ALPHA_ACCESS_AGREEMENT_THRESHOLD,
    ALPHA_ENTROPY_THRESHOLD,
    load_frozen_v2_config,
    normalized_entropy,
)
from experiments.phase2.e3_v2.oracle import (
    OracleConfig,
    OracleContractError,
    SegmentSpec,
    build_segment_catalog,
)
from experiments.phase2.e3_v2.query_accessibility import collect_hybrid_query_probe
from experiments.phase2.e3_v2.real_model_preflight import (
    REFERENCE_BACKEND,
    _force_torch_reference_backend,
    _load_model,
    model_provenance,
)
from experiments.phase2.e3_v2.run_discovery import _build_samples
from experiments.phase2.e3_v2.statistics import (
    sample_grouped_bootstrap_interval,
    spearman_correlation,
)
from experiments.utils.cache_access import get_cache_layer
from experiments.utils.eval_harness import get_ground_truths
from experiments.utils.memory_accounting import get_active_kv_bytes
from experiments.utils.metrics import compute_f1, normalize_answer
from experiments.utils.model_loader import (
    get_full_attention_indices,
    get_linear_attention_indices,
)
from experiments.utils.run_manifest import collect_environment, ensure_run_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_SCHEMA = "hmo.query_accessibility.end_task_protocol.v1"
RESULTS_FILENAME = "end_task_results.jsonl"
SUMMARY_FILENAME = "end_task_summary.json"
SYSTEMS = ("raw_alpha", "frozen_v2", "full_kv_reference")


def _atomic_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_protocol(path: Path) -> tuple[dict, str]:
    try:
        encoded = path.read_bytes()
        payload = json.loads(encoded)
    except (OSError, json.JSONDecodeError) as exc:
        raise OracleContractError(f"cannot read end-task protocol: {path}") from exc
    if (
        payload.get("schema_version") != PROTOCOL_SCHEMA
        or payload.get("status") != "frozen_before_outcomes"
        or tuple(payload.get("systems", ())) != SYSTEMS
        or payload.get("equal_byte_comparison") != ["raw_alpha", "frozen_v2"]
        or payload.get("primary_metric") != "normalized_answer_contains"
        or payload.get("model_id") != "Qwen/Qwen3.5-0.8B"
        or not payload.get("model_revision")
        or set(payload.get("stages", {})) != {"smoke", "8k", "16k"}
        or int(payload.get("max_new_tokens", 0)) <= 0
        or int(payload.get("bootstrap_samples", 0)) <= 0
    ):
        raise OracleContractError("end-task protocol mismatch")
    return payload, hashlib.sha256(encoded).hexdigest()


def select_equal_byte_segments(
    alpha_by_segment: Mapping[int, float],
    access_by_segment: Mapping[int, float] | None,
    segments: Sequence[SegmentSpec],
    *,
    middle_kv_fraction: float,
) -> dict:
    eligible = tuple(segment for segment in segments if segment.eligible)
    if len(eligible) < 2 or not 0 < middle_kv_fraction < 1:
        raise OracleContractError("invalid end-task segment budget")
    costs = {segment.kv_bytes for segment in eligible}
    if len(costs) != 1 or next(iter(costs)) <= 0:
        raise OracleContractError("eligible segments must have one positive KV cost")
    if any(segment.segment_id not in alpha_by_segment for segment in eligible):
        raise OracleContractError("attention does not cover eligible segments")
    if access_by_segment is not None and any(
        segment.segment_id not in access_by_segment for segment in eligible
    ):
        raise OracleContractError("accessibility does not cover eligible segments")

    unit_cost = next(iter(costs))
    budget_limit = math.floor(sum(segment.kv_bytes for segment in eligible) * middle_kv_fraction)
    budget_slots = budget_limit // unit_cost
    if budget_slots < 1 or budget_slots >= len(eligible):
        raise OracleContractError("end-task middle budget has invalid slot count")

    segment_ids = np.asarray([segment.segment_id for segment in eligible], dtype=np.int64)
    alpha = np.asarray([alpha_by_segment[int(index)] for index in segment_ids], dtype=np.float64)
    if not np.all(np.isfinite(alpha)):
        raise OracleContractError("attention must be finite")
    entropy = normalized_entropy(alpha)
    if access_by_segment is None:
        agreement = None
        gate_enabled = False
        v2_score = alpha
    else:
        access = np.asarray(
            [access_by_segment[int(index)] for index in segment_ids], dtype=np.float64
        )
        if not np.all(np.isfinite(access)):
            raise OracleContractError("accessibility must be finite")
        agreement = spearman_correlation(alpha, access)
        gate_enabled = (
            entropy >= ALPHA_ENTROPY_THRESHOLD
            and agreement < ALPHA_ACCESS_AGREEMENT_THRESHOLD
        )
        v2_score = alpha * (1.0 - _rank01(access)) if gate_enabled else alpha
    raw_order = np.argsort(-alpha, kind="stable")
    v2_order = np.argsort(-v2_score, kind="stable")
    raw_selected = tuple(sorted(int(value) for value in segment_ids[raw_order[:budget_slots]]))
    v2_selected = tuple(sorted(int(value) for value in segment_ids[v2_order[:budget_slots]]))
    return {
        "eligible_segment_ids": tuple(int(value) for value in segment_ids),
        "budget_limit_bytes": int(budget_limit),
        "budget_slots": int(budget_slots),
        "unit_segment_bytes": int(unit_cost),
        "normalized_alpha_entropy": float(entropy),
        "accessibility_measured": access_by_segment is not None,
        "alpha_access_spearman": None if agreement is None else float(agreement),
        "gate_enabled": bool(gate_enabled),
        "raw_alpha_segment_ids": raw_selected,
        "frozen_v2_segment_ids": v2_selected,
        "membership_changed": raw_selected != v2_selected,
    }


def make_selected_segment_intervention(
    segments: Sequence[SegmentSpec],
    attention_layer_indices: Sequence[int],
    selected_middle_ids: Sequence[int],
    *,
    context_tokens: int,
    name: str,
):
    segment_tuple = tuple(segments)
    selected = tuple(sorted(int(value) for value in selected_middle_ids))
    eligible_ids = {segment.segment_id for segment in segment_tuple if segment.eligible}
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= eligible_ids:
        raise OracleContractError("selected middle segments are invalid")
    active_segments = tuple(
        segment for segment in segment_tuple
        if segment.protected or segment.segment_id in selected
    )
    expected_bytes = sum(segment.kv_bytes for segment in active_segments)
    expected_full_bytes = sum(segment.kv_bytes for segment in segment_tuple)

    def intervene(cache, context_ids: torch.Tensor) -> InterventionResult:
        if context_ids.shape != (1, context_tokens):
            raise OracleContractError("selected intervention context mismatch")
        before_bytes = get_active_kv_bytes(cache, list(attention_layer_indices))
        if before_bytes != expected_full_bytes:
            raise OracleContractError("selected intervention did not receive Full-KV context")
        positions = torch.cat(
            [
                torch.arange(segment.start, segment.end, device=context_ids.device)
                for segment in active_segments
            ]
        ).to(torch.long).sort().values
        for layer_index in attention_layer_indices:
            layer = get_cache_layer(cache, int(layer_index))
            if not layer.has_kv() or layer.keys.shape[-2] != context_tokens:
                raise OracleContractError("attention cache does not match segment catalog")
            layer.keys = layer.keys.index_select(-2, positions)
            layer.values = layer.values.index_select(-2, positions)
        after_bytes = get_active_kv_bytes(cache, list(attention_layer_indices))
        if after_bytes != expected_bytes:
            raise OracleContractError("selected intervention resident bytes mismatch")
        return InterventionResult(
            name=name,
            active_context_positions=positions,
            metadata={
                "selected_middle_segment_ids": selected,
                "context_resident_bytes": int(after_bytes),
            },
        )

    return intervene


def score_generated_text(text: str, sample) -> dict:
    prediction = normalize_answer(text)
    truths = [normalize_answer(value) for value in get_ground_truths(sample)]
    truths = [value for value in truths if value]
    if not truths:
        raise OracleContractError("end-task sample has no ground truth")
    return {
        "normalized_answer_contains": float(any(value in prediction for value in truths)),
        "normalized_exact_match": float(prediction in truths),
        "token_f1": float(max(compute_f1(text, value) for value in get_ground_truths(sample))),
    }


def summarize_results(rows: Sequence[Mapping], *, bootstrap_samples: int, seed: int) -> dict:
    if not rows:
        raise OracleContractError("end-task summary requires results")
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise OracleContractError("end-task sample results must be unique")
    metrics = ("normalized_answer_contains", "normalized_exact_match", "token_f1")
    systems = {
        system: {
            metric: float(np.mean([row["systems"][system][metric] for row in rows]))
            for metric in metrics
        }
        for system in SYSTEMS
    }
    task_systems = {
        dataset: {
            system: {
                metric: float(np.mean([
                    row["systems"][system][metric]
                    for row in rows if row["dataset"] == dataset
                ]))
                for metric in metrics
            }
            for system in SYSTEMS
        }
        for dataset in sorted({str(row["dataset"]) for row in rows})
    }
    paired = {}
    for metric_index, metric in enumerate(metrics):
        deltas = {
            str(row["sample_id"]): float(
                row["systems"]["frozen_v2"][metric]
                - row["systems"]["raw_alpha"][metric]
            )
            for row in rows
        }
        task_deltas = {
            dataset: float(np.mean([
                deltas[str(row["sample_id"])] for row in rows if row["dataset"] == dataset
            ]))
            for dataset in sorted({str(row["dataset"]) for row in rows})
        }
        paired[metric] = {
            "improvement": sample_grouped_bootstrap_interval(
                deltas,
                n_bootstrap=bootstrap_samples,
                seed=seed + metric_index,
            ).__dict__,
            "task_improvement": task_deltas,
            "wins": sum(value > 1e-12 for value in deltas.values()),
            "ties": sum(abs(value) <= 1e-12 for value in deltas.values()),
            "losses": sum(value < -1e-12 for value in deltas.values()),
        }
    return {
        "systems": systems,
        "task_systems": task_systems,
        "paired_v2_vs_raw_alpha": paired,
        "gate_enabled_samples": sum(bool(row["selection"]["gate_enabled"]) for row in rows),
        "membership_changed_samples": sum(
            bool(row["selection"]["membership_changed"]) for row in rows
        ),
        "task_membership_changed_samples": {
            dataset: sum(
                bool(row["selection"]["membership_changed"])
                for row in rows if row["dataset"] == dataset
            )
            for dataset in sorted({str(row["dataset"]) for row in rows})
        },
    }


def _load_completed(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise OracleContractError("persisted end-task results contain duplicate samples")
    return rows


def _cleanup_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("end-task validation requires exactly one visible GPU")
    method, method_sha = load_frozen_v2_config(Path(args.frozen_v2_config).resolve())
    protocol, protocol_sha = _load_protocol(Path(args.protocol).resolve())
    if protocol["frozen_v2_sha256"] != method_sha:
        raise OracleContractError("end-task protocol method hash mismatch")
    if (
        args.model_id != protocol["model_id"]
        or args.model_revision != protocol["model_revision"]
        or method.get("model") != protocol["model_id"]
    ):
        raise OracleContractError("end-task model identity disagrees with frozen protocol")
    stage = protocol["stages"][args.stage]
    run_dir = Path(args.run_dir).resolve()
    model_path = Path(args.model_path).resolve()
    model_identity = model_provenance(model_path, args.model_id, revision=args.model_revision)
    scientific_args = {
        **stage,
        "stage": args.stage,
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "max_new_tokens": protocol["max_new_tokens"],
        "bootstrap_samples": protocol["bootstrap_samples"],
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    manifest = ensure_run_manifest(
        run_dir,
        experiment="query_accessibility_v2_end_task",
        args=scientific_args,
        selections={
            "systems": list(SYSTEMS),
            "equal_byte_comparison": ["raw_alpha", "frozen_v2"],
            "frozen_v2": method,
            "frozen_v2_sha256": method_sha,
            "protocol_sha256": protocol_sha,
            "protocol_stage": args.stage,
            "candidate_search": False,
        },
        model=model_identity,
        project_root=PROJECT_ROOT,
        require_clean=True,
        environment=collect_environment(),
    )

    torch.manual_seed(stage["seed"])
    torch.cuda.manual_seed_all(stage["seed"])
    torch.set_float32_matmul_precision("highest")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model, tokenizer = _load_model(model_path)
    attention_layers = tuple(get_full_attention_indices(model.config))
    recurrent_layers = tuple(get_linear_attention_indices(model.config))
    _force_torch_reference_backend(model, recurrent_layers)
    samples = _build_samples(tokenizer, Namespace(**stage))
    results_path = run_dir / RESULTS_FILENAME
    completed = _load_completed(results_path)
    completed_ids = {row["sample_id"] for row in completed}
    rows = list(completed)
    oracle_config = OracleConfig(
        segment_length=stage["segment_length"],
        middle_kv_fraction=stage["middle_kv_fraction"],
    )

    for sample_index, sample in enumerate(samples, start=1):
        if sample.sample_id in completed_ids:
            continue
        sample_started = time.perf_counter()
        prompt = tokenize_sample_prompt(sample, tokenizer)
        with torch.no_grad():
            context_outputs = model.model(
                prompt.context_ids.to(model.device), use_cache=True, return_dict=True
            )
        segments = build_segment_catalog(
            context_outputs.past_key_values,
            attention_layers,
            context_tokens=prompt.context_tokens,
            config=oracle_config,
        )
        del context_outputs
        _cleanup_cuda()
        probe = collect_hybrid_query_probe(
            model,
            prompt,
            attention_layer_indices=attention_layers,
            recurrent_layer_indices=recurrent_layers,
            segments=segments,
            segment_length=stage["segment_length"],
        )
        selection = select_equal_byte_segments(
            probe.alpha.as_dict(),
            probe.accessibility.field_dict("read_share"),
            segments,
            middle_kv_fraction=stage["middle_kv_fraction"],
        )
        _cleanup_cuda()

        generated = {}
        arm_specs = (
            (
                "raw_alpha",
                make_selected_segment_intervention(
                    segments,
                    attention_layers,
                    selection["raw_alpha_segment_ids"],
                    context_tokens=prompt.context_tokens,
                    name="raw_alpha_topk",
                ),
            ),
            (
                "frozen_v2",
                make_selected_segment_intervention(
                    segments,
                    attention_layers,
                    selection["frozen_v2_segment_ids"],
                    context_tokens=prompt.context_tokens,
                    name="frozen_v2_topk",
                ),
            ),
            ("full_kv_reference", full_kv_intervention),
        )
        for system, intervention in arm_specs:
            arm_started = time.perf_counter()
            state = run_post_intervention_prompt(
                model,
                prompt,
                attention_layer_indices=attention_layers,
                recurrent_layer_indices=recurrent_layers,
                intervention=intervention,
            )
            resident_bytes = get_active_kv_bytes(state.cache, list(attention_layers))
            answer = generate_greedy(
                model, tokenizer, state, max_new_tokens=protocol["max_new_tokens"]
            )
            generated[system] = {
                **score_generated_text(answer.text, sample),
                "generated_text": answer.text,
                "generated_token_ids": [int(value) for value in answer.token_ids[0].tolist()],
                "context_segment_ids": (
                    list(selection[f"{system}_segment_ids"])
                    if system != "full_kv_reference"
                    else [segment.segment_id for segment in segments]
                ),
                "post_query_resident_kv_bytes": int(resident_bytes),
                "elapsed_seconds": time.perf_counter() - arm_started,
            }
            _cleanup_cuda()

        raw = generated["raw_alpha"]
        v2 = generated["frozen_v2"]
        if raw["post_query_resident_kv_bytes"] != v2["post_query_resident_kv_bytes"]:
            raise OracleContractError("raw alpha and V2 are not equal-byte at decode")
        if (
            not selection["membership_changed"]
            and raw["generated_token_ids"] != v2["generated_token_ids"]
        ):
            raise OracleContractError("identical selected sets produced different greedy outputs")
        row = {
            "sample_id": sample.sample_id,
            "dataset": sample.dataset,
            "answer": sample.answer,
            "context_tokens": prompt.context_tokens,
            "query_tokens": prompt.query_tokens,
            "selection": selection,
            "systems": generated,
            "elapsed_seconds": time.perf_counter() - sample_started,
        }
        rows.append(row)
        _append_jsonl(results_path, row)
        print(
            f"[{sample_index}/{len(samples)}] {sample.sample_id}: "
            f"changed={selection['membership_changed']} "
            f"raw={raw['normalized_answer_contains']:.0f} "
            f"v2={v2['normalized_answer_contains']:.0f}",
            flush=True,
        )

    expected_ids = {sample.sample_id for sample in samples}
    if {row["sample_id"] for row in rows} != expected_ids:
        raise OracleContractError("end-task run did not complete the frozen sample set")
    rows.sort(key=lambda row: row["sample_id"])
    summary = {
        "schema_version": "hmo.query_accessibility.end_task_result.v1",
        "status": "complete",
        "scope": "prospective_equal_byte_end_task_generation",
        "manifest_id": manifest["manifest_id"],
        "stage": args.stage,
        "frozen_v2_sha256": method_sha,
        "protocol_sha256": protocol_sha,
        "sample_count": len(rows),
        "analysis": summarize_results(
            rows,
            bootstrap_samples=protocol["bootstrap_samples"],
            seed=stage["seed"],
        ),
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "gpu_name": torch.cuda.get_device_name(0),
            "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "samples": rows,
    }
    _atomic_json(run_dir / SUMMARY_FILENAME, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--model-revision")
    parser.add_argument("--frozen-v2-config", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--stage", choices=("smoke", "8k", "16k"), required=True)
    parser.add_argument("--run-dir", required=True)
    return parser.parse_args()


def main() -> int:
    payload = run(parse_args())
    print(json.dumps({key: value for key, value in payload.items() if key != "samples"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
