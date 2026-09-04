"""Frozen Full-KV solvability evaluation on transparently augmented 32K HotpotQA."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import zipfile
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

from experiments.phase2.e3_v2.context_query import (
    full_kv_intervention,
    generate_greedy,
    run_post_intervention_prompt,
    tokenize_sample_prompt_aligned,
)
from experiments.phase2.e3_v2.oracle import OracleContractError
from experiments.phase2.e3_v2.real_model_preflight import (
    REFERENCE_BACKEND,
    _force_torch_reference_backend,
    _load_model,
    model_provenance,
)
from experiments.phase2.e3_v2.run_coverage_fidelity import (
    _append_jsonl,
    _atomic_json,
    _cleanup_cuda,
)
from experiments.utils.dataset_utils import EvalSample
from experiments.utils.eval_harness import get_ground_truths, score_prediction
from experiments.utils.memory_accounting import get_active_kv_bytes
from experiments.utils.model_loader import (
    get_full_attention_indices,
    get_linear_attention_indices,
)
from experiments.utils.run_manifest import collect_environment, ensure_run_manifest
from experiments.vendor.longbench_metrics import LONG_BENCH_REVISION, normalize_answer


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_SCHEMA = "hmo.hotpotqa_32k_solvability_protocol.v1"
RESULT_SCHEMA = "hmo.hotpotqa_32k_solvability_result.v1"
RESULTS_FILENAME = "hotpot_solvability_results.jsonl"
SUMMARY_FILENAME = "hotpot_solvability_summary.json"
STAGE_CASE_COUNTS = {"smoke": 1, "formal": 4}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol(path: Path) -> tuple[dict, str]:
    try:
        encoded = path.read_bytes()
        payload = json.loads(encoded)
    except (OSError, json.JSONDecodeError) as exc:
        raise OracleContractError(f"cannot read Hotpot solvability protocol: {path}") from exc

    construction = payload.get("construction", {})
    execution = payload.get("execution", {})
    generation = payload.get("generation", {})
    dataset = payload.get("dataset", {})
    cases = payload.get("cases", [])
    if (
        payload.get("schema_version") != PROTOCOL_SCHEMA
        or payload.get("status") != "frozen_before_outcomes"
        or payload.get("purpose")
        != "full_kv_solvability_only_before_any_compressed_hotpotqa_arm"
        or payload.get("benchmark_variant") != "longbench_hotpotqa_32k_aug"
        or payload.get("system") != "full_kv_reference"
        or payload.get("model_id") != "Qwen/Qwen3.5-0.8B"
        or not payload.get("model_revision")
        or dataset.get("member") != "data/hotpotqa.jsonl"
        or dataset.get("metric") != "qa_f1_score"
        or dataset.get("metric_revision") != LONG_BENCH_REVISION
        or len(dataset.get("archive_sha256", "")) != 64
        or construction.get("max_memory_context_tokens") != 32768
        or construction.get("max_shortfall_tokens") not in (0, 1, 2)
        or not construction.get("base_context_unchanged")
        or construction.get("boundary_alignment")
        != "if_one_token_crosses_the_semantic_context_boundary_move_that_complete_token_into_memory_context"
        or construction.get("outcome_conditioned_selection") is not False
        or not construction.get("delimiter")
        or generation.get("decoding") != "greedy"
        or generation.get("max_new_tokens") != 32
        or int(generation.get("inference_seed", 0)) <= 0
        or execution.get("formal_case_count") != len(cases)
        or execution.get("formal_case_count") != STAGE_CASE_COUNTS["formal"]
        or execution.get("smoke_case_count") != STAGE_CASE_COUNTS["smoke"]
        or execution.get("automatic_compressed_continuation") is not False
        or any(
            not isinstance(case.get(key), expected)
            for case in cases
            for key, expected in (
                ("base_index", int),
                ("base_record_sha256", str),
                ("donor_index", int),
                ("donor_record_sha256", str),
            )
        )
    ):
        raise OracleContractError("Hotpot solvability protocol mismatch")
    identities = [(case["base_index"], case["donor_index"]) for case in cases]
    if len(set(identities)) != len(identities):
        raise OracleContractError("Hotpot solvability cases are not unique")
    return payload, _sha256_bytes(encoded)


def choose_max_fitting_prefix(
    item_count: int,
    target: int,
    measure: Callable[[int], int],
) -> tuple[int, int]:
    """Find the longest prefix whose measured serialized prompt fits the target."""
    if item_count < 0 or target <= 0:
        raise ValueError("prefix search requires nonnegative items and positive target")
    if measure(0) > target:
        raise OracleContractError("base Hotpot prompt already exceeds the 32K target")

    low, high = 0, item_count
    while low < high:
        middle = (low + high + 1) // 2
        if measure(middle) <= target:
            low = middle
        else:
            high = middle - 1
    measured = measure(low)
    while low < item_count and measure(low + 1) <= target:
        low += 1
        measured = measure(low)
    return low, measured


def _load_records(archive: Path, protocol: Mapping) -> tuple[list[dict], list[bytes]]:
    expected_sha = protocol["dataset"]["archive_sha256"]
    observed_sha = _sha256_file(archive)
    if observed_sha != expected_sha:
        raise OracleContractError(
            f"LongBench archive SHA mismatch: {observed_sha} != {expected_sha}"
        )
    member = protocol["dataset"]["member"]
    try:
        with zipfile.ZipFile(archive) as handle:
            encoded = handle.read(member)
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise OracleContractError(f"cannot read {member} from {archive}") from exc

    raw_lines = encoded.splitlines()
    try:
        records = [json.loads(line) for line in raw_lines if line.strip()]
    except json.JSONDecodeError as exc:
        raise OracleContractError("HotpotQA JSONL contains an invalid record") from exc
    if len(records) != len(raw_lines):
        raise OracleContractError("HotpotQA JSONL contains blank records")
    return records, raw_lines


def _validate_case_records(
    records: Sequence[Mapping], raw_lines: Sequence[bytes], cases: Sequence[Mapping]
) -> None:
    for case in cases:
        for side in ("base", "donor"):
            index = case[f"{side}_index"]
            if index < 0 or index >= len(records):
                raise OracleContractError(f"{side} record index is out of range: {index}")
            observed = _sha256_bytes(raw_lines[index])
            if observed != case[f"{side}_record_sha256"]:
                raise OracleContractError(f"{side} record SHA changed at index {index}")

        base = records[case["base_index"]]
        donor = records[case["donor_index"]]
        answers = [str(value) for value in base.get("answers", []) if str(value)]
        if not answers:
            raise OracleContractError("HotpotQA base record has no answers")
        donor_lower = str(donor.get("context", "")).lower()
        if any(answer.lower() in donor_lower for answer in answers):
            raise OracleContractError("a frozen distractor contains a base gold answer")


def _make_augmented_sample(base: Mapping, context: str, base_index: int, donor_index: int):
    answers = [str(value) for value in base["answers"]]
    return EvalSample(
        dataset="longbench_hotpotqa",
        sample_id=f"hotpotqa_32k_aug_b{base_index:04d}_d{donor_index:04d}",
        context=context,
        question=str(base["input"]),
        answer=answers[0],
        answers=answers,
        context_length=32768,
    )


def validate_longest_base_selection(
    records: Sequence[Mapping], cases: Sequence[Mapping], tokenizer
) -> list[dict]:
    lengths = [
        len(tokenizer.encode(str(record["context"]), add_special_tokens=False))
        for record in records
    ]
    expected = sorted(range(len(records)), key=lambda index: (-lengths[index], index))[
        : len(cases)
    ]
    observed = [int(case["base_index"]) for case in cases]
    if observed != expected:
        raise OracleContractError(
            f"frozen Hotpot bases are not the longest records: {observed} != {expected}"
        )
    return [{"base_index": index, "source_context_tokens": lengths[index]} for index in observed]


def tokenize_hotpot_prompt(sample: EvalSample, tokenizer):
    """Align the semantic context boundary to the next exact tokenizer boundary."""
    return tokenize_sample_prompt_aligned(sample, tokenizer)

def build_augmented_sample(
    base: Mapping,
    donor: Mapping,
    case: Mapping,
    tokenizer,
    construction: Mapping,
) -> tuple[EvalSample, object, dict]:
    base_context = str(base["context"])
    delimiter = construction["delimiter"]
    donor_ids = tokenizer.encode(str(donor["context"]), add_special_tokens=False)
    target = int(construction["max_memory_context_tokens"])
    cache: dict[int, tuple[EvalSample, object, int]] = {}

    def materialize(prefix_tokens: int):
        if prefix_tokens not in cache:
            donor_prefix = tokenizer.decode(
                donor_ids[:prefix_tokens], skip_special_tokens=True
            )
            sample = _make_augmented_sample(
                base,
                base_context + delimiter + donor_prefix,
                case["base_index"],
                case["donor_index"],
            )
            prompt, boundary_shift = tokenize_hotpot_prompt(sample, tokenizer)
            cache[prefix_tokens] = (sample, prompt, boundary_shift)
        return cache[prefix_tokens]

    donor_prefix_tokens, measured = choose_max_fitting_prefix(
        len(donor_ids), target, lambda count: materialize(count)[1].context_tokens
    )
    sample, prompt, boundary_shift = materialize(donor_prefix_tokens)
    shortfall = target - measured
    if shortfall < 0 or shortfall > int(construction["max_shortfall_tokens"]):
        raise OracleContractError(
            f"augmented prompt misses 32K target by {shortfall} tokens"
        )
    if not sample.context.startswith(base_context + delimiter):
        raise OracleContractError("augmentation changed the base context")
    return sample, prompt, {
        "base_source_context_tokens": len(
            tokenizer.encode(base_context, add_special_tokens=False)
        ),
        "donor_source_tokens": len(donor_ids),
        "donor_prefix_tokens": donor_prefix_tokens,
        "memory_context_tokens": prompt.context_tokens,
        "query_tokens": prompt.query_tokens,
        "target_tokens": target,
        "shortfall_tokens": shortfall,
        "boundary_shift_characters": boundary_shift,
    }


def _load_completed(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise OracleContractError(
                    f"invalid Hotpot result at line {line_number}"
                ) from exc
    sample_ids = [row.get("sample_id") for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise OracleContractError("Hotpot results contain duplicate cases")
    return rows


def summarize_results(rows: Sequence[Mapping]) -> dict:
    if not rows:
        raise OracleContractError("Hotpot solvability summary requires results")
    f1_values = [float(row["official_qa_f1"]) for row in rows]
    return {
        "case_count": len(rows),
        "mean_official_qa_f1": float(np.mean(f1_values)),
        "median_official_qa_f1": float(np.median(f1_values)),
        "nonzero_f1_cases": sum(value > 0.0 for value in f1_values),
        "normalized_exact_match_cases": sum(
            float(row["normalized_exact_match"]) for row in rows
        ),
        "normalized_answer_contains_cases": sum(
            float(row["normalized_answer_contains"]) for row in rows
        ),
        "mean_memory_context_tokens": float(
            np.mean([row["construction"]["memory_context_tokens"] for row in rows])
        ),
        "mean_post_query_resident_kv_bytes": float(
            np.mean([row["post_query_resident_kv_bytes"] for row in rows])
        ),
        "initial_solvability_signal": any(value > 0.0 for value in f1_values),
        "stronger_compressed_pilot_signal": sum(value > 0.0 for value in f1_values) >= 2,
    }


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Hotpot solvability evaluation requires one visible GPU")
    protocol, protocol_sha = load_protocol(Path(args.protocol).resolve())
    if (
        args.model_id != protocol["model_id"]
        or args.model_revision != protocol["model_revision"]
    ):
        raise OracleContractError("model identity disagrees with Hotpot protocol")

    case_count = STAGE_CASE_COUNTS[args.stage_set]
    selected_cases = protocol["cases"][:case_count]
    archive = Path(args.archive).resolve()
    records, raw_lines = _load_records(archive, protocol)
    _validate_case_records(records, raw_lines, selected_cases)
    run_dir = Path(args.run_dir).resolve()
    model_path = Path(args.model_path).resolve()
    model_identity = model_provenance(
        model_path, args.model_id, revision=args.model_revision
    )
    manifest = ensure_run_manifest(
        run_dir,
        experiment="hotpotqa_32k_aug_full_kv_solvability",
        args={
            "archive": str(archive),
            "archive_sha256": protocol["dataset"]["archive_sha256"],
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "protocol_sha256": protocol_sha,
            "stage_set": args.stage_set,
            "max_new_tokens": protocol["generation"]["max_new_tokens"],
            "inference_seed": protocol["generation"]["inference_seed"],
            "recurrent_backend": REFERENCE_BACKEND,
            "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        selections={
            "scope": (
                "operational_smoke_excluded_from_claims"
                if args.stage_set == "smoke"
                else "full_kv_solvability_routing_evidence"
            ),
            "benchmark_variant": protocol["benchmark_variant"],
            "system": protocol["system"],
            "case_count": case_count,
            "cases": selected_cases,
            "candidate_search": False,
            "automatic_compressed_continuation": False,
        },
        model=model_identity,
        project_root=PROJECT_ROOT,
        require_clean=True,
        environment=collect_environment(),
    )

    results_path = run_dir / RESULTS_FILENAME
    completed = _load_completed(results_path)
    if completed and not args.resume:
        raise OracleContractError("Hotpot results exist; pass --resume to continue")
    rows = list(completed)
    completed_ids = {str(row["sample_id"]) for row in completed}

    seed = int(protocol["generation"]["inference_seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("highest")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model, tokenizer = _load_model(model_path)
    attention_layers = tuple(get_full_attention_indices(model.config))
    recurrent_layers = tuple(get_linear_attention_indices(model.config))
    if not attention_layers or not recurrent_layers:
        raise OracleContractError("Hotpot model is not a Hybrid architecture")
    _force_torch_reference_backend(model, recurrent_layers)
    longest_base_audit = validate_longest_base_selection(
        records, protocol["cases"], tokenizer
    )

    for case_index, case in enumerate(selected_cases, start=1):
        base = records[case["base_index"]]
        donor = records[case["donor_index"]]
        sample, prompt, construction = build_augmented_sample(
            base, donor, case, tokenizer, protocol["construction"]
        )
        if sample.sample_id in completed_ids:
            continue
        case_started = time.perf_counter()
        state = run_post_intervention_prompt(
            model,
            prompt,
            attention_layer_indices=attention_layers,
            recurrent_layer_indices=recurrent_layers,
            intervention=full_kv_intervention,
        )
        resident_bytes = get_active_kv_bytes(state.cache, list(attention_layers))
        generated = generate_greedy(
            model,
            tokenizer,
            state,
            max_new_tokens=protocol["generation"]["max_new_tokens"],
        )
        scores = score_prediction(generated.text, sample)
        truths = get_ground_truths(sample)
        prediction = normalize_answer(generated.text)
        normalized_truths = [normalize_answer(value) for value in truths]
        row = {
            "stage": args.stage_set,
            "sample_id": sample.sample_id,
            "dataset": sample.dataset,
            "base_index": case["base_index"],
            "donor_index": case["donor_index"],
            "question": sample.question,
            "answers": truths,
            "construction": construction,
            "system": "full_kv_reference",
            "official_metric": scores.primary_metric,
            "official_qa_f1": scores.primary_score,
            "normalized_exact_match": float(prediction in normalized_truths),
            "normalized_answer_contains": float(
                any(value and value in prediction for value in normalized_truths)
            ),
            "generated_text": generated.text,
            "generated_token_ids": [
                int(value) for value in generated.token_ids[0].tolist()
            ],
            "post_query_resident_kv_bytes": int(resident_bytes),
            "elapsed_seconds": time.perf_counter() - case_started,
        }
        rows.append(row)
        _append_jsonl(results_path, row)
        print(
            f"[{case_index}/{case_count}] {sample.sample_id}: "
            f"tokens={prompt.context_tokens} f1={scores.primary_score:.4f} "
            f"prediction={generated.text!r}",
            flush=True,
        )
        del state
        _cleanup_cuda()

    expected_ids = {
        f"hotpotqa_32k_aug_b{case['base_index']:04d}_d{case['donor_index']:04d}"
        for case in selected_cases
    }
    if {str(row["sample_id"]) for row in rows} != expected_ids:
        raise OracleContractError("Hotpot run did not complete the selected package")
    rows.sort(key=lambda row: (row["base_index"], row["donor_index"]))
    summary = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "scope": (
            "operational_smoke_excluded_from_claims"
            if args.stage_set == "smoke"
            else "full_kv_solvability_routing_evidence"
        ),
        "manifest_id": manifest["manifest_id"],
        "protocol_sha256": protocol_sha,
        "analysis": summarize_results(rows),
        "routing_interpretation": protocol["execution"]["routing_interpretation"],
        "automatic_compressed_continuation": False,
        "longest_base_selection_audit": longest_base_audit,
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
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--stage-set", choices=tuple(STAGE_CASE_COUNTS), default="formal")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    payload = run(parse_args())
    print(json.dumps({key: value for key, value in payload.items() if key != "samples"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
