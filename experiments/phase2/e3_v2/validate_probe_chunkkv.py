"""Real-model equivalence and executability check for probe v2 and ChunkKV."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.phase2.e3_v2.attention_probe import collect_attention_token_probe
from experiments.phase2.e3_v2.chunkkv_adapter import (
    CHUNKKV_CHUNK_SIZE,
    build_chunkkv_plan,
    make_chunkkv_intervention,
)
from experiments.phase2.e3_v2.context_query import (
    generate_greedy,
    run_post_intervention_prompt,
    tokenize_sample_prompt,
)
from experiments.phase2.e3_v2.coverage_fidelity import allocate_coverage_fidelity
from experiments.phase2.e3_v2.coverage_fidelity_cache import (
    build_retained_position_plan,
    make_coverage_fidelity_intervention,
)
from experiments.phase2.e3_v2.oracle import (
    OracleConfig,
    OracleContractError,
    build_segment_catalog,
)
from experiments.phase2.e3_v2.query_accessibility import (
    collect_hybrid_query_token_probe,
)
from experiments.phase2.e3_v2.query_probe_cache import retained_positions_sha256
from experiments.phase2.e3_v2.real_model_preflight import (
    _force_torch_reference_backend,
    _load_model,
    model_provenance,
)
from experiments.phase2.e3_v2.run_coverage_fidelity import _cleanup_cuda
from experiments.utils.dataset_utils import make_needle_samples
from experiments.utils.memory_accounting import get_active_kv_bytes
from experiments.utils.model_loader import (
    get_full_attention_indices,
    get_linear_attention_indices,
)


SCHEMA_VERSION = "hmo.probe_chunkkv_validation.v1"


def _generate(model, tokenizer, prompt, attention_layers, recurrent_layers, intervention):
    state = run_post_intervention_prompt(
        model,
        prompt,
        attention_layer_indices=attention_layers,
        recurrent_layer_indices=recurrent_layers,
        intervention=intervention,
    )
    resident_bytes = get_active_kv_bytes(state.cache, list(attention_layers))
    metadata = dict(state.intervention.metadata)
    generated = generate_greedy(model, tokenizer, state, max_new_tokens=4)
    result = {
        "token_ids": [int(value) for value in generated.token_ids[0].tolist()],
        "text": generated.text,
        "post_query_resident_kv_bytes": int(resident_bytes),
        "intervention_metadata": metadata,
    }
    del state
    _cleanup_cuda()
    return result


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("validation requires exactly one visible GPU")
    model_path = Path(args.model_path).resolve()
    identity = model_provenance(
        model_path, args.model_id, revision=args.model_revision
    )
    model, tokenizer = _load_model(model_path)
    attention_layers = tuple(get_full_attention_indices(model.config))
    recurrent_layers = tuple(get_linear_attention_indices(model.config))
    if not attention_layers or not recurrent_layers:
        raise OracleContractError("validation model must be hybrid")
    _force_torch_reference_backend(model, recurrent_layers)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sample = make_needle_samples(
        tokenizer, n_samples=1, context_length=args.context_length, seed=args.seed
    )[0]
    prompt = tokenize_sample_prompt(sample, tokenizer)
    with torch.no_grad():
        context_outputs = model.model(
            prompt.context_ids.to(model.device), use_cache=True, return_dict=True
        )
    segments = build_segment_catalog(
        context_outputs.past_key_values,
        attention_layers,
        context_tokens=prompt.context_tokens,
        config=OracleConfig(
            segment_length=args.segment_length,
            middle_kv_fraction=args.middle_kv_fraction,
        ),
    )
    del context_outputs
    _cleanup_cuda()

    legacy = collect_hybrid_query_token_probe(
        model,
        prompt,
        attention_layer_indices=attention_layers,
        recurrent_layer_indices=recurrent_layers,
        segments=segments,
        segment_length=args.segment_length,
    )
    _cleanup_cuda()
    attention_only = collect_attention_token_probe(
        model,
        prompt,
        attention_layer_indices=attention_layers,
        segments=segments,
    )
    old_scores = np.asarray(legacy.token_attention_mass, dtype=np.float32)
    new_scores = np.asarray(attention_only.token_attention_mass, dtype=np.float32)
    max_abs = float(np.max(np.abs(old_scores - new_scores)))
    bitwise_equal = bool(np.array_equal(old_scores, new_scores))
    if not np.allclose(old_scores, new_scores, rtol=0.0, atol=1e-7):
        raise OracleContractError("attention-only probe changed aggregate token scores")

    eligible_ids = tuple(segment.segment_id for segment in segments if segment.eligible)
    old_attention = legacy.alpha.as_dict()
    new_attention = attention_only.alpha.as_dict()
    old_plan = allocate_coverage_fidelity(
        {index: old_attention[index] for index in eligible_ids},
        None,
        segments,
        middle_kv_fraction=args.middle_kv_fraction,
        sparse_width=args.sparse_width,
        use_accessibility=False,
    )
    new_plan = allocate_coverage_fidelity(
        {index: new_attention[index] for index in eligible_ids},
        None,
        segments,
        middle_kv_fraction=args.middle_kv_fraction,
        sparse_width=args.sparse_width,
        use_accessibility=False,
    )
    old_positions = build_retained_position_plan(
        old_plan,
        segments,
        legacy.token_attention_mass,
        context_tokens=prompt.context_tokens,
        sparse_selector="max_mass_window",
    )
    new_positions = build_retained_position_plan(
        new_plan,
        segments,
        attention_only.token_attention_mass,
        context_tokens=prompt.context_tokens,
        sparse_selector="max_mass_window",
    )
    if old_positions.active_positions != new_positions.active_positions:
        raise OracleContractError("attention-only probe changed HMO retained positions")

    eligible = [segment for segment in segments if segment.eligible]
    context_token_kv_bytes = eligible[0].kv_bytes // eligible[0].token_count
    chunkkv = build_chunkkv_plan(
        segments,
        attention_only.layer_scores(),
        context_tokens=prompt.context_tokens,
        target_context_charged_bytes=new_plan.total_charged_bytes,
        context_token_kv_bytes=context_token_kv_bytes,
        observation_query_tokens=prompt.query_tokens,
        chunk_size=CHUNKKV_CHUNK_SIZE,
    )
    old_generated = _generate(
        model,
        tokenizer,
        prompt,
        attention_layers,
        recurrent_layers,
        make_coverage_fidelity_intervention(old_positions, attention_layers, name="old"),
    )
    new_generated = _generate(
        model,
        tokenizer,
        prompt,
        attention_layers,
        recurrent_layers,
        make_coverage_fidelity_intervention(new_positions, attention_layers, name="new"),
    )
    if old_generated["token_ids"] != new_generated["token_ids"]:
        raise OracleContractError("probe migration changed generated HMO tokens")
    chunkkv_generated = _generate(
        model,
        tokenizer,
        prompt,
        attention_layers,
        recurrent_layers,
        make_chunkkv_intervention(chunkkv),
    )
    expected_bytes = new_plan.total_charged_bytes + (
        prompt.query_tokens * context_token_kv_bytes
    )
    if chunkkv_generated["post_query_resident_kv_bytes"] != expected_bytes:
        raise OracleContractError("ChunkKV post-query bytes miss the HMO target")

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "model": identity,
        "sample_id": sample.sample_id,
        "context_tokens": prompt.context_tokens,
        "query_tokens": prompt.query_tokens,
        "attention_layers": list(attention_layers),
        "recurrent_layers": list(recurrent_layers),
        "probe_equivalence": {
            "aggregate_token_scores_bitwise_equal": bitwise_equal,
            "aggregate_token_scores_max_abs": max_abs,
            "retained_positions_equal": True,
            "retained_positions_sha256": retained_positions_sha256(
                new_positions.active_positions
            ),
            "generated_token_ids_equal": True,
        },
        "equal_byte_target": {
            "context_resident_kv_bytes": new_plan.total_charged_bytes,
            "post_query_resident_kv_bytes": expected_bytes,
        },
        "legacy_hmo": old_generated,
        "attention_only_hmo": new_generated,
        "chunkkv": {
            "plan": chunkkv.to_dict(),
            "generation": chunkkv_generated,
        },
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--segment-length", type=int, default=256)
    parser.add_argument("--middle-kv-fraction", type=float, default=0.5)
    parser.add_argument("--sparse-width", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260905)
    return parser.parse_args()


def main() -> int:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
