"""Real-model integrity preflight for the E3-v2 scientific runner."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.phase2.e3_v2.alpha_probe import collect_isolated_query_alpha
from experiments.phase2.e3_v2.context_query import (
    InterventionResult,
    P0B_EXECUTION_EVENTS,
    full_kv_intervention,
    generate_greedy,
    reference_full_prompt_logits,
    run_post_intervention_prompt,
    tokenize_sample_prompt,
)
from experiments.phase2.e3_v2.integrity import (
    REQUIRED_INTEGRITY_CHECKS,
    IntegrityCheck,
    require_integrity_gate,
)
from experiments.phase2.e3_v2.oracle import (
    OracleConfig,
    OracleContractError,
    audit_equal_byte_pair,
    build_oracle_plan,
    build_segment_catalog,
    ensure_oracle_manifest,
    load_oracle_manifest,
    make_oracle_intervention,
)
from experiments.phase2.e3_v2.recurrent_signals import sequential_gated_delta_trace
from experiments.utils.cache_access import get_cache_layer
from experiments.utils.dataset_utils import make_needle_samples
from experiments.utils.model_loader import (
    get_full_attention_indices,
    get_linear_attention_indices,
)
from experiments.utils.run_manifest import collect_environment, ensure_run_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVIDENCE_FILENAME = "preflight_evidence.json"
ORACLE_MANIFEST_FILENAME = "oracle_manifest.json"


@dataclass(frozen=True)
class PreflightThresholds:
    full_kv_max_abs_max: float = 0.5
    full_kv_mean_abs_max: float = 0.1
    full_kv_js_divergence_max: float = 0.001
    full_kv_top10_overlap_min: float = 0.8
    needle_max_abs_min: float = 1e-4
    needle_mean_abs_min: float = 1e-7


REFERENCE_BACKEND = "transformers_torch_reference"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_provenance(model_path: Path, model_id: str) -> dict:
    model_path = model_path.resolve()
    config_path = model_path / "config.json"
    index_path = model_path / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise FileNotFoundError("model snapshot lacks config or safetensors index")
    revision = model_path.name if model_path.parent.name == "snapshots" else None
    weight_files = sorted(model_path.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError("model snapshot contains no safetensors weights")
    return {
        "model_id": model_id,
        "revision": revision,
        "config_sha256": _sha256(config_path),
        "weight_index_sha256": _sha256(index_path),
        "weight_files": [
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "blob_id": path.resolve().name,
            }
            for path in weight_files
        ],
    }


def _logit_difference(left: torch.Tensor, right: torch.Tensor) -> dict[str, float | bool | int]:
    left = left.detach().to(torch.float32).cpu()
    right = right.detach().to(torch.float32).cpu()
    if left.shape != right.shape or left.ndim != 2 or left.shape[0] != 1:
        raise OracleContractError("logit comparisons require aligned [1, vocab] tensors")
    difference = (left - right).abs()
    left_log_probability = torch.log_softmax(left, dim=-1)
    right_log_probability = torch.log_softmax(right, dim=-1)
    left_probability = left_log_probability.exp()
    right_probability = right_log_probability.exp()
    mixture = 0.5 * (left_probability + right_probability)
    log_mixture = mixture.clamp_min(torch.finfo(mixture.dtype).tiny).log()
    js_divergence = 0.5 * (
        torch.sum(left_probability * (left_log_probability - log_mixture))
        + torch.sum(right_probability * (right_log_probability - log_mixture))
    )
    top_k = min(10, left.shape[-1])
    left_top = set(torch.topk(left, top_k, dim=-1).indices.reshape(-1).tolist())
    right_top = set(torch.topk(right, top_k, dim=-1).indices.reshape(-1).tolist())
    return {
        "max_abs": float(difference.max().item()),
        "mean_abs": float(difference.mean().item()),
        "argmax_left": int(left.argmax(dim=-1).item()),
        "argmax_right": int(right.argmax(dim=-1).item()),
        "argmax_equal": bool(torch.equal(left.argmax(dim=-1), right.argmax(dim=-1))),
        "exact_equal": bool(torch.equal(left, right)),
        "js_divergence": float(js_divergence.item()),
        "max_probability_abs": float(
            (left_probability - right_probability).abs().max().item()
        ),
        "top10_overlap": len(left_top & right_top) / top_k,
    }


def _needle_token_indices(tokenizer, prompt, needle: str) -> tuple[int, ...]:
    start = prompt.text.memory_context.find(needle)
    if start < 0 or prompt.text.memory_context.find(needle, start + 1) >= 0:
        raise OracleContractError("controlled needle must occur exactly once in memory context")
    end = start + len(needle)
    encoded = tokenizer(
        prompt.text.full_prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    if not torch.equal(encoded["input_ids"], prompt.full_ids):
        raise OracleContractError("needle offset tokenization disagrees with prompt tokenization")
    offsets = encoded["offset_mapping"][0].tolist()
    indices = tuple(
        index
        for index, (token_start, token_end) in enumerate(offsets[: prompt.context_tokens])
        if token_end > start and token_start < end and token_end > token_start
    )
    if not indices:
        raise OracleContractError("controlled needle has no context token positions")
    return indices


def _make_drop_segments_intervention(plan, segment_ids: tuple[int, ...]):
    segment_set = set(segment_ids)
    if not segment_set or not segment_set <= set(plan.eligible_segment_ids):
        raise OracleContractError("controlled needle must occupy eligible middle segments")
    active_segments = [
        segment for segment in plan.segments if segment.segment_id not in segment_set
    ]

    def intervene(cache, context_ids: torch.Tensor) -> InterventionResult:
        positions = torch.cat(
            [
                torch.arange(segment.start, segment.end, device=context_ids.device)
                for segment in active_segments
            ]
        ).to(torch.long)
        for layer_index in plan.attention_layer_indices:
            layer = get_cache_layer(cache, layer_index)
            if not layer.has_kv() or layer.keys.shape[-2] != plan.context_tokens:
                raise OracleContractError("needle intervention did not receive Full-KV context")
            layer.keys = layer.keys.index_select(-2, positions)
            layer.values = layer.values.index_select(-2, positions)
        return InterventionResult(
            name="drop_controlled_needle",
            active_context_positions=positions,
            metadata={"dropped_segment_ids": tuple(sorted(segment_set))},
        )

    return intervene


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _evidence(metrics: Mapping) -> str:
    return json.dumps(dict(metrics), ensure_ascii=True, sort_keys=True)


def _capture_check(
    checks: dict[str, IntegrityCheck],
    name: str,
    function: Callable[[], tuple[bool, Mapping]],
) -> None:
    started = time.perf_counter()
    try:
        passed, metrics = function()
        payload = {**dict(metrics), "elapsed_seconds": time.perf_counter() - started}
        checks[name] = IntegrityCheck(bool(passed), _evidence(payload))
    except Exception as exc:
        checks[name] = IntegrityCheck(
            False,
            _evidence(
                {
                    "elapsed_seconds": time.perf_counter() - started,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            ),
        )
    finally:
        _cleanup_cuda()


def _synthetic_gate_direction() -> tuple[bool, Mapping]:
    key = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]]])
    value = torch.tensor([[[[2.0]], [[0.0]], [[0.0]]]])
    beta = torch.tensor([[[1.0], [0.0], [0.0]]])
    weak = torch.tensor([[[0.0], [-0.1], [-0.1]]])
    strong = torch.tensor([[[0.0], [-1.0], [-1.0]]])
    weak_trace = sequential_gated_delta_trace(key, value, beta, weak)
    strong_trace = sequential_gated_delta_trace(key, value, beta, strong)
    weak_norm = float(torch.linalg.vector_norm(weak_trace.final_state).item())
    strong_norm = float(torch.linalg.vector_norm(strong_trace.final_state).item())
    expected_ratio = math.exp(1.8)
    observed_ratio = weak_norm / strong_norm
    passed = weak_norm > strong_norm and math.isclose(
        observed_ratio, expected_ratio, rel_tol=1e-5
    )
    return passed, {
        "weak_log_survival": -0.2,
        "strong_log_survival": -2.0,
        "weak_final_state_norm": weak_norm,
        "strong_final_state_norm": strong_norm,
        "observed_ratio": observed_ratio,
        "expected_ratio": expected_ratio,
    }


def _load_model(model_path: Path):
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
    )
    model.eval()
    return model, tokenizer


def _force_torch_reference_backend(model, recurrent_layers) -> None:
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        torch_causal_conv1d_update,
        torch_chunk_gated_delta_rule,
        torch_recurrent_gated_delta_rule,
    )

    for layer_index in recurrent_layers:
        linear_attention = model.model.layers[layer_index].linear_attn
        linear_attention.causal_conv1d_fn = None
        linear_attention.causal_conv1d_update = torch_causal_conv1d_update
        linear_attention.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
        linear_attention.recurrent_gated_delta_rule = torch_recurrent_gated_delta_rule


def run_preflight(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("preflight requires exactly one visible CUDA device")
    thresholds = PreflightThresholds()
    run_dir = Path(args.run_dir).resolve()
    model_path = Path(args.model_path).resolve()
    model_identity = model_provenance(model_path, args.model_id)
    scientific_args = {
        "model_id": args.model_id,
        "context_length": args.context_length,
        "segment_length": args.segment_length,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "recurrent_backend": REFERENCE_BACKEND,
        "thresholds": asdict(thresholds),
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    ensure_run_manifest(
        run_dir,
        experiment="e3_v2_real_model_preflight",
        args=scientific_args,
        selections={"sample_count": 1, "arm_execution": "sequential"},
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
    if not attention_layers or not recurrent_layers:
        raise OracleContractError("model is not a hybrid attention/recurrent architecture")
    _force_torch_reference_backend(model, recurrent_layers)

    sample = make_needle_samples(
        tokenizer,
        n_samples=1,
        context_length=args.context_length,
        seed=args.seed,
    )[0]
    prompt = tokenize_sample_prompt(sample, tokenizer)
    context_outputs = model.model(
        prompt.context_ids.to(model.device), use_cache=True, return_dict=True
    )
    config = OracleConfig(segment_length=args.segment_length)
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
    del context_outputs
    _cleanup_cuda()
    oracle_path = run_dir / ORACLE_MANIFEST_FILENAME
    ensure_oracle_manifest(oracle_path, plan)
    needle_tokens = _needle_token_indices(tokenizer, prompt, sample.answer)
    needle_segments = tuple(
        sorted(
            {
                segment.segment_id
                for segment in segments
                if any(segment.start <= index < segment.end for index in needle_tokens)
            }
        )
    )

    checks: dict[str, IntegrityCheck] = {}
    shared: dict[str, object] = {}

    def query_order_check():
        state = run_post_intervention_prompt(
            model,
            prompt,
            attention_layer_indices=attention_layers,
            recurrent_layer_indices=recurrent_layers,
        )
        metrics = {
            "events": list(state.events),
            "logical_position": state.logical_position,
            "context_tokens": prompt.context_tokens,
            "query_tokens": prompt.query_tokens,
        }
        shared["full_split_logits"] = state.first_answer_logits.detach().cpu().clone()
        passed = state.events == P0B_EXECUTION_EVENTS and state.logical_position == (
            prompt.context_tokens + prompt.query_tokens
        )
        del state
        return passed, metrics

    _capture_check(checks, "query_after_intervention", query_order_check)

    def equivalence_check():
        reference = reference_full_prompt_logits(model, prompt.full_ids).cpu()
        split = shared.get("full_split_logits")
        if not isinstance(split, torch.Tensor):
            state = run_post_intervention_prompt(
                model,
                prompt,
                attention_layer_indices=attention_layers,
                recurrent_layer_indices=recurrent_layers,
            )
            split = state.first_answer_logits.detach().cpu().clone()
            del state
        metrics = _logit_difference(reference, split)
        passed = (
            bool(metrics["argmax_equal"])
            and float(metrics["max_abs"]) <= thresholds.full_kv_max_abs_max
            and float(metrics["mean_abs"]) <= thresholds.full_kv_mean_abs_max
            and float(metrics["js_divergence"])
            <= thresholds.full_kv_js_divergence_max
            and float(metrics["top10_overlap"])
            >= thresholds.full_kv_top10_overlap_min
        )
        metrics.update(asdict(thresholds))
        return passed, metrics

    _capture_check(checks, "full_kv_equivalence", equivalence_check)

    def repeat_check():
        results = []
        for _ in range(2):
            state = run_post_intervention_prompt(
                model,
                prompt,
                attention_layer_indices=attention_layers,
                recurrent_layer_indices=recurrent_layers,
            )
            logits = state.first_answer_logits.detach().cpu().clone()
            generated = generate_greedy(
                model, tokenizer, state, max_new_tokens=args.max_new_tokens
            ).token_ids.cpu()
            results.append((logits, generated))
            del state
            _cleanup_cuda()
        metrics = _logit_difference(results[0][0], results[1][0])
        metrics["generated_tokens_left"] = results[0][1].reshape(-1).tolist()
        metrics["generated_tokens_right"] = results[1][1].reshape(-1).tolist()
        passed = bool(metrics["exact_equal"]) and torch.equal(results[0][1], results[1][1])
        return passed, metrics

    _capture_check(checks, "repeated_arm_determinism", repeat_check)

    def equal_byte_check():
        comparison = plan.comparisons[0]
        target = run_post_intervention_prompt(
            model,
            prompt,
            attention_layer_indices=attention_layers,
            recurrent_layer_indices=recurrent_layers,
            intervention=make_oracle_intervention(
                plan, comparison.comparison_id, "target"
            ),
        )
        donor = run_post_intervention_prompt(
            model,
            prompt,
            attention_layer_indices=attention_layers,
            recurrent_layer_indices=recurrent_layers,
            intervention=make_oracle_intervention(
                plan, comparison.comparison_id, "donor"
            ),
        )
        audit = audit_equal_byte_pair(target, donor, attention_layers)
        metrics = asdict(audit)
        metrics["manifest_id"] = plan.manifest_id
        passed = (
            audit.middle_charged_bytes == comparison.middle_charged_bytes
            and audit.context_resident_bytes == comparison.context_resident_bytes
        )
        del target, donor
        return passed, metrics

    _capture_check(checks, "equal_byte_arms", equal_byte_check)
    _capture_check(checks, "recurrent_gate_direction", _synthetic_gate_direction)

    def needle_check():
        full = run_post_intervention_prompt(
            model,
            prompt,
            attention_layer_indices=attention_layers,
            recurrent_layer_indices=recurrent_layers,
            intervention=full_kv_intervention,
        )
        dropped = run_post_intervention_prompt(
            model,
            prompt,
            attention_layer_indices=attention_layers,
            recurrent_layer_indices=recurrent_layers,
            intervention=_make_drop_segments_intervention(plan, needle_segments),
        )
        metrics = _logit_difference(
            full.first_answer_logits, dropped.first_answer_logits
        )
        metrics["needle_token_indices"] = list(needle_tokens)
        metrics["dropped_segment_ids"] = list(needle_segments)
        metrics["needle_max_abs_min"] = thresholds.needle_max_abs_min
        metrics["needle_mean_abs_min"] = thresholds.needle_mean_abs_min
        passed = (
            float(metrics["max_abs"]) >= thresholds.needle_max_abs_min
            and float(metrics["mean_abs"]) >= thresholds.needle_mean_abs_min
        )
        del full, dropped
        return passed, metrics

    _capture_check(checks, "controlled_needle_logit_effect", needle_check)

    def alpha_check():
        comparison = plan.comparisons[0]

        def oracle_logits():
            state = run_post_intervention_prompt(
                model,
                prompt,
                attention_layer_indices=attention_layers,
                recurrent_layer_indices=recurrent_layers,
                intervention=make_oracle_intervention(
                    plan, comparison.comparison_id, "target"
                ),
            )
            logits = state.first_answer_logits.detach().cpu().clone()
            del state
            return logits

        before = oracle_logits()
        alpha = collect_isolated_query_alpha(
            model,
            prompt,
            attention_layer_indices=attention_layers,
            segments=segments,
        )
        after = oracle_logits()
        metrics = _logit_difference(before, after)
        metrics["alpha_segment_count"] = len(alpha.segment_ids)
        metrics["alpha_total_context_mass"] = float(sum(alpha.attention_mass))
        passed = (
            bool(metrics["exact_equal"])
            and len(alpha.segment_ids) == len(segments)
            and math.isfinite(float(metrics["alpha_total_context_mass"]))
            and float(metrics["alpha_total_context_mass"]) > 0
            and not hasattr(alpha, "cache")
        )
        return passed, metrics

    _capture_check(checks, "alpha_isolation", alpha_check)

    def manifest_check():
        loaded = load_oracle_manifest(oracle_path)
        exact = loaded.to_dict() == plan.to_dict()
        return exact, {
            "manifest_id": loaded.manifest_id,
            "segment_count": len(loaded.segments),
            "comparison_count": len(loaded.comparisons),
            "middle_budget_slots": loaded.middle_budget_slots,
            "exact_roundtrip": exact,
        }

    _capture_check(checks, "manifest_recoverability", manifest_check)

    missing = set(REQUIRED_INTEGRITY_CHECKS) - set(checks)
    if missing:
        raise OracleContractError(f"preflight failed to execute checks: {sorted(missing)}")
    try:
        require_integrity_gate(checks)
        status = "pass"
        gate_error = None
    except OracleContractError as exc:
        status = "block"
        gate_error = str(exc)

    payload = {
        "status": status,
        "gate_error": gate_error,
        "model": model_identity,
        "sample": {
            "sample_id": sample.sample_id,
            "context_tokens": prompt.context_tokens,
            "query_tokens": prompt.query_tokens,
            "answer": sample.answer,
            "needle_segments": list(needle_segments),
        },
        "architecture": {
            "attention_layers": list(attention_layers),
            "recurrent_layers": list(recurrent_layers),
            "recurrent_backend": REFERENCE_BACKEND,
        },
        "oracle": {
            "manifest_id": plan.manifest_id,
            "segment_length": args.segment_length,
            "middle_budget_slots": plan.middle_budget_slots,
            "comparison_count": len(plan.comparisons),
        },
        "checks": {
            name: {"passed": checks[name].passed, "evidence": checks[name].evidence}
            for name in REQUIRED_INTEGRITY_CHECKS
        },
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "gpu_name": torch.cuda.get_device_name(0),
            "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
    }
    evidence_path = run_dir / EVIDENCE_FILENAME
    temporary = evidence_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(evidence_path)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--segment-length", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    if args.context_length < 1024 or args.segment_length <= 0:
        parser.error("context-length must be >=1024 and segment-length positive")
    return args


def main() -> int:
    payload = run_preflight(parse_args())
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if payload["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
