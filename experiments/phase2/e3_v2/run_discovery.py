"""Lightweight real-model discovery pilot for E3-v2 recurrent signals."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from experiments.phase2.e3_v2.alpha_probe import collect_isolated_query_alpha
from experiments.phase2.e3_v2.context_query import (
    run_post_intervention_prompt,
    score_gold_answer_logprob,
    tokenize_answer_continuation,
    tokenize_sample_prompt,
)
from experiments.phase2.e3_v2.oracle import (
    ArmQuality,
    OracleConfig,
    OracleContractError,
    PairObservation,
    aggregate_pair_observations,
    build_oracle_plan,
    build_pair_observation,
    build_segment_catalog,
    ensure_oracle_manifest,
    make_oracle_intervention,
)
from experiments.phase2.e3_v2.real_model_preflight import (
    REFERENCE_BACKEND,
    _force_torch_reference_backend,
    _load_model,
    model_provenance,
)
from experiments.phase2.e3_v2.recurrent_signals import (
    AggregatedRecurrentCandidates,
    Qwen35RecurrentCandidateHookManager,
    aggregate_recurrent_candidates,
)
from experiments.phase2.e3_v2.statistics import (
    SegmentEvidence,
    evaluate_candidate_grouped_cv,
    residual_correlation,
    select_discovery_candidate,
    spearman_correlation,
)
from experiments.utils.dataset_utils import (
    make_longeval_lines_samples,
    make_needle_samples,
)
from experiments.utils.hooks import DeltaNetHookManager
from experiments.utils.model_loader import (
    get_full_attention_indices,
    get_linear_attention_indices,
)
from experiments.utils.run_manifest import collect_environment, ensure_run_manifest
from experiments.utils.saturation import compute_segment_saturation


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OBSERVATIONS_FILENAME = "pair_observations.jsonl"
SUMMARY_FILENAME = "discovery_summary.json"
EVALUATED_CANDIDATES = (
    "sigma_current",
    "delta_update",
    "survival_retention",
    "suffix_interference",
    "phi_sigma_alpha",
    "phi_delta_alpha",
)


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


def load_pair_observations(path: Path) -> tuple[PairObservation, ...]:
    if not path.exists():
        return ()
    rows = []
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                observation = PairObservation(
                    oracle_manifest_id=str(raw["oracle_manifest_id"]),
                    sample_id=str(raw["sample_id"]),
                    comparison_id=str(raw["comparison_id"]),
                    target_segment=int(raw["target_segment"]),
                    donor_segment=int(raw["donor_segment"]),
                    background_segments=tuple(raw["background_segments"]),
                    delta_logprob=float(raw["delta_logprob"]),
                    delta_secondary=(
                        None
                        if raw.get("delta_secondary") is None
                        else float(raw["delta_secondary"])
                    ),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise OracleContractError(
                    f"invalid observation at {path}:{line_number}"
                ) from exc
            key = (observation.sample_id, observation.comparison_id)
            if key in seen:
                raise OracleContractError(f"duplicate persisted observation {key}")
            seen.add(key)
            rows.append(observation)
    return tuple(rows)


def build_segment_evidence(
    *,
    plan,
    dataset: str,
    utility: Mapping[int, float],
    alpha: Mapping[int, float],
    recurrent: AggregatedRecurrentCandidates,
    sigma_current: Sequence[float],
) -> tuple[SegmentEvidence, ...]:
    fields = {
        "delta_update": recurrent.delta_update,
        "survival_retention": recurrent.survival_retention,
        "decay_risk": recurrent.decay_risk,
        "suffix_interference": recurrent.suffix_interference,
        "surviving_write_norm": recurrent.surviving_write_norm,
    }
    rows = []
    for segment_id in plan.eligible_segment_ids:
        if (
            segment_id not in utility
            or segment_id not in alpha
            or segment_id >= len(sigma_current)
            or any(segment_id >= len(values) for values in fields.values())
        ):
            raise OracleContractError(
                f"candidate alignment is incomplete for segment {segment_id}"
            )
        segment_alpha = float(alpha[segment_id])
        sigma = float(sigma_current[segment_id])
        delta = float(fields["delta_update"][segment_id])
        candidates = {
            "sigma_current": sigma,
            **{name: float(values[segment_id]) for name, values in fields.items()},
            "phi_sigma_alpha": sigma * segment_alpha,
            "phi_delta_alpha": delta * segment_alpha,
        }
        values = (float(utility[segment_id]), segment_alpha, *candidates.values())
        if not all(math.isfinite(value) for value in values):
            raise OracleContractError("segment evidence contains non-finite values")
        rows.append(
            SegmentEvidence(
                sample_id=plan.sample_id,
                dataset=dataset,
                segment_id=segment_id,
                utility=float(utility[segment_id]),
                alpha=segment_alpha,
                normalized_position=plan.segments[segment_id].normalized_position,
                candidates=candidates,
            )
        )
    return tuple(rows)


def analyze_discovery(
    evidence: Sequence[SegmentEvidence],
    k_by_sample: Mapping[str, int],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    by_sample: dict[str, list[SegmentEvidence]] = {}
    for row in evidence:
        by_sample.setdefault(row.sample_id, []).append(row)
    label_ranges = {
        sample_id: max(row.utility for row in rows)
        - min(row.utility for row in rows)
        for sample_id, rows in sorted(by_sample.items())
    }
    usable_ids = {
        sample_id for sample_id, value_range in label_ranges.items() if value_range > 1e-12
    }
    usable = tuple(row for row in evidence if row.sample_id in usable_ids)
    payload = {
        "candidate_names": list(EVALUATED_CANDIDATES),
        "sample_label_ranges": label_ranges,
        "excluded_zero_range_samples": sorted(set(by_sample) - usable_ids),
        "candidate_results": {},
        "selected_candidate": None,
        "direction": "inconclusive",
    }
    if len(usable_ids) < 2:
        return payload

    cv_results = []
    for candidate in EVALUATED_CANDIDATES:
        result = evaluate_candidate_grouped_cv(
            usable,
            candidate,
            k_by_sample={sample_id: k_by_sample[sample_id] for sample_id in usable_ids},
            folds=min(4, len(usable_ids)),
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        candidate_values = [float(row.candidates[candidate]) for row in usable]
        utilities = [row.utility for row in usable]
        alphas = [row.alpha for row in usable]
        positions = [row.normalized_position for row in usable]
        payload["candidate_results"][candidate] = {
            **asdict(result),
            "pooled_spearman": spearman_correlation(candidate_values, utilities),
            "residual_correlation": residual_correlation(
                candidate_values,
                utilities,
                alphas,
                positions,
            ),
        }
        cv_results.append(result)

    selected = select_discovery_candidate(cv_results)
    payload["selected_candidate"] = selected.candidate
    pairwise = selected.pairwise_improvement.mean
    ndcg = selected.ndcg_improvement.mean
    if pairwise > 0 and ndcg > 0:
        payload["direction"] = "positive"
    elif pairwise <= 0 and ndcg <= 0:
        payload["direction"] = "negative"
    else:
        payload["direction"] = "mixed"
    return payload


def _build_samples(tokenizer, args) -> list:
    samples = []
    for dataset_index, dataset in enumerate(args.datasets.split(",")):
        name = dataset.strip()
        dataset_seed = args.seed + dataset_index * 1000
        if name == "needle":
            built = make_needle_samples(
                tokenizer,
                n_samples=args.samples_per_dataset,
                context_length=args.context_length,
                seed=dataset_seed,
            )
        elif name == "longeval_lines":
            built = make_longeval_lines_samples(
                tokenizer,
                n_samples=args.samples_per_dataset,
                context_length=args.context_length,
                seed=dataset_seed,
            )
        else:
            raise ValueError(f"unsupported discovery dataset {name!r}")
        samples.extend(built)
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise OracleContractError("discovery sample IDs must be unique")
    return samples


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _capture_sample_signals(
    model,
    prompt,
    recurrent_layers: Sequence[int],
    *,
    segment_length: int,
):
    recurrent_hooks = Qwen35RecurrentCandidateHookManager(
        model,
        recurrent_layers,
        segment_length=segment_length,
    )
    legacy_hooks = DeltaNetHookManager(
        model,
        list(recurrent_layers),
        segment_length=segment_length,
    )
    recurrent_hooks.attach()
    legacy_hooks.attach()
    try:
        with torch.no_grad():
            outputs = model.model(
                prompt.context_ids.to(model.device),
                use_cache=True,
                return_dict=True,
            )
        recurrent = aggregate_recurrent_candidates(
            recurrent_hooks.finalize_context()
        )
        legacy_signals = dict(legacy_hooks.get_signals())
    finally:
        recurrent_hooks.detach()
        legacy_hooks.remove()
    sigma = compute_segment_saturation(
        legacy_signals,
        segment_length=segment_length,
        input_ids=prompt.context_ids,
    )
    return outputs, recurrent, sigma


def run_discovery(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("P1 discovery requires exactly one visible CUDA device")
    run_dir = Path(args.run_dir).resolve()
    model_path = Path(args.model_path).resolve()
    model_identity = model_provenance(
        model_path,
        args.model_id,
        revision=args.model_revision,
    )
    scientific_args = {
        "model_id": args.model_id,
        "datasets": args.datasets,
        "samples_per_dataset": args.samples_per_dataset,
        "context_length": args.context_length,
        "segment_length": args.segment_length,
        "middle_kv_fraction": args.middle_kv_fraction,
        "donors_per_segment": args.donors_per_segment,
        "backgrounds_per_pair": args.backgrounds_per_pair,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "recurrent_backend": REFERENCE_BACKEND,
        "secondary_generation": False,
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    manifest = ensure_run_manifest(
        run_dir,
        experiment="e3_v2_p1_discovery",
        args=scientific_args,
        selections={
            "scope": "discovery_only",
            "candidate_names": list(EVALUATED_CANDIDATES),
            "oracle_primary_only": True,
            "lightweight_oracle_override": {
                "donors_per_segment": args.donors_per_segment,
                "backgrounds_per_pair": args.backgrounds_per_pair,
            },
        },
        model=model_identity,
        project_root=PROJECT_ROOT,
        require_clean=True,
        environment=collect_environment(),
    )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("highest")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model, tokenizer = _load_model(model_path)
    attention_layers = tuple(get_full_attention_indices(model.config))
    recurrent_layers = tuple(get_linear_attention_indices(model.config))
    _force_torch_reference_backend(model, recurrent_layers)
    samples = _build_samples(tokenizer, args)
    observation_path = run_dir / OBSERVATIONS_FILENAME
    persisted = load_pair_observations(observation_path)
    persisted_by_key = {
        (row.sample_id, row.comparison_id): row for row in persisted
    }
    all_evidence = []
    k_by_sample = {}
    sample_summaries = []

    config = OracleConfig(
        segment_length=args.segment_length,
        middle_kv_fraction=args.middle_kv_fraction,
        donors_per_segment=args.donors_per_segment,
        backgrounds_per_pair=args.backgrounds_per_pair,
        seed=args.seed,
    )
    for sample_index, sample in enumerate(samples, start=1):
        sample_started = time.perf_counter()
        prompt = tokenize_sample_prompt(sample, tokenizer)
        context_outputs, recurrent, sigma = _capture_sample_signals(
            model,
            prompt,
            recurrent_layers,
            segment_length=args.segment_length,
        )
        segments = build_segment_catalog(
            context_outputs.past_key_values,
            attention_layers,
            context_tokens=prompt.context_tokens,
            config=config,
        )
        plan = build_oracle_plan(
            sample_id=sample.sample_id,
            context_tokens=prompt.context_tokens,
            attention_layer_indices=attention_layers,
            segments=segments,
            config=config,
        )
        sample_dir = run_dir / "samples" / sample.sample_id
        ensure_oracle_manifest(sample_dir / "oracle_manifest.json", plan)
        del context_outputs
        _cleanup_cuda()

        alpha_result = collect_isolated_query_alpha(
            model,
            prompt,
            attention_layer_indices=attention_layers,
            segments=segments,
        )
        alpha = alpha_result.as_dict()
        answer_ids = tokenize_answer_continuation(
            tokenizer,
            prompt,
            sample.answer,
        )
        observations = []
        total_comparisons = len(plan.comparisons)
        print(
            f"[{sample_index}/{len(samples)}] {sample.sample_id}: "
            f"{total_comparisons} comparisons",
            flush=True,
        )
        for comparison_index, comparison in enumerate(plan.comparisons, start=1):
            key = (sample.sample_id, comparison.comparison_id)
            if key in persisted_by_key:
                observation = persisted_by_key[key]
                if observation.oracle_manifest_id != plan.manifest_id:
                    raise OracleContractError(
                        f"persisted observation disagrees with {sample.sample_id} plan"
                    )
                observations.append(observation)
                continue

            target_state = run_post_intervention_prompt(
                model,
                prompt,
                attention_layer_indices=attention_layers,
                recurrent_layer_indices=recurrent_layers,
                intervention=make_oracle_intervention(
                    plan, comparison.comparison_id, "target"
                ),
            )
            target_score = score_gold_answer_logprob(
                model,
                target_state,
                answer_ids,
            )
            del target_state

            donor_state = run_post_intervention_prompt(
                model,
                prompt,
                attention_layer_indices=attention_layers,
                recurrent_layer_indices=recurrent_layers,
                intervention=make_oracle_intervention(
                    plan, comparison.comparison_id, "donor"
                ),
            )
            donor_score = score_gold_answer_logprob(
                model,
                donor_state,
                answer_ids,
            )
            del donor_state

            observation = build_pair_observation(
                plan,
                comparison.comparison_id,
                ArmQuality(mean_gold_logprob=target_score.mean_logprob),
                ArmQuality(mean_gold_logprob=donor_score.mean_logprob),
            )
            observations.append(observation)
            persisted_by_key[key] = observation
            _append_jsonl(
                observation_path,
                {
                    **asdict(observation),
                    "target_mean_gold_logprob": target_score.mean_logprob,
                    "donor_mean_gold_logprob": donor_score.mean_logprob,
                },
            )
            if comparison_index % args.progress_every == 0:
                print(
                    f"  {sample.sample_id}: {comparison_index}/{total_comparisons}",
                    flush=True,
                )
                _cleanup_cuda()

        pair_aggregates, utility = aggregate_pair_observations(plan, observations)
        sample_evidence = build_segment_evidence(
            plan=plan,
            dataset=sample.dataset,
            utility=utility,
            alpha=alpha,
            recurrent=recurrent,
            sigma_current=sigma,
        )
        all_evidence.extend(sample_evidence)
        k_by_sample[sample.sample_id] = plan.middle_budget_slots
        evidence_payload = [asdict(row) for row in sample_evidence]
        _atomic_json(sample_dir / "segment_evidence.json", {"rows": evidence_payload})
        sample_summary = {
            "sample_id": sample.sample_id,
            "dataset": sample.dataset,
            "context_tokens": prompt.context_tokens,
            "query_tokens": prompt.query_tokens,
            "answer_tokens": int(answer_ids.shape[1]),
            "oracle_manifest_id": plan.manifest_id,
            "segments": len(plan.segments),
            "eligible_segments": len(plan.eligible_segment_ids),
            "middle_budget_slots": plan.middle_budget_slots,
            "comparisons": total_comparisons,
            "pair_aggregates": len(pair_aggregates),
            "utility_min": min(utility.values()),
            "utility_max": max(utility.values()),
            "elapsed_seconds": time.perf_counter() - sample_started,
        }
        sample_summaries.append(sample_summary)
        _atomic_json(sample_dir / "sample_summary.json", sample_summary)
        print(
            f"  completed {sample.sample_id} in "
            f"{sample_summary['elapsed_seconds']:.1f}s",
            flush=True,
        )
        _cleanup_cuda()

    analysis = analyze_discovery(
        all_evidence,
        k_by_sample,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    payload = {
        "status": "complete",
        "scope": "discovery_only_not_confirmation",
        "manifest_id": manifest["manifest_id"],
        "model_id": args.model_id,
        "architecture": {
            "attention_layers": list(attention_layers),
            "recurrent_layers": list(recurrent_layers),
            "recurrent_backend": REFERENCE_BACKEND,
        },
        "sample_summaries": sample_summaries,
        "analysis": analysis,
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "gpu_name": torch.cuda.get_device_name(0),
            "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
    }
    _atomic_json(run_dir / SUMMARY_FILENAME, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--model-revision")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--datasets", default="needle,longeval_lines")
    parser.add_argument("--samples-per-dataset", type=int, default=2)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--segment-length", type=int, default=256)
    parser.add_argument("--middle-kv-fraction", type=float, default=0.10)
    parser.add_argument("--donors-per-segment", type=int, default=2)
    parser.add_argument("--backgrounds-per-pair", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    if (
        args.samples_per_dataset <= 0
        or args.context_length < 1024
        or args.segment_length <= 0
        or args.donors_per_segment <= 0
        or args.backgrounds_per_pair <= 0
        or args.bootstrap_samples <= 0
        or args.progress_every <= 0
    ):
        parser.error("all count/length arguments must be positive")
    if not 0 < args.middle_kv_fraction < 1:
        parser.error("middle-kv-fraction must lie in (0, 1)")
    return args


def main() -> int:
    payload = run_discovery(parse_args())
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
