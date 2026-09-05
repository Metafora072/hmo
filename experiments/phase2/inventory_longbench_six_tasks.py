"""Inventory and deterministically select six LongBench QA task prefixes."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from experiments.phase2.e3_v2.context_query import tokenize_sample_prompt_aligned
from experiments.phase2.e3_v2.run_native_tasks import _make_sample, select_longest_candidates


TASKS = (
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(args: argparse.Namespace) -> dict:
    archive = Path(args.archive).resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        Path(args.model_path).resolve(), local_files_only=True, trust_remote_code=True
    )
    datasets = {}
    with zipfile.ZipFile(archive) as handle:
        for task in TASKS:
            member = f"data/{task}.jsonl"
            raw_lines = handle.read(member).splitlines()
            records = [json.loads(line) for line in raw_lines]
            metadata = []
            for index, (record, raw) in enumerate(zip(records, raw_lines)):
                sample = _make_sample(task, index, record)
                prompt, shift = tokenize_sample_prompt_aligned(sample, tokenizer)
                metadata.append(
                    {
                        "index": index,
                        "record_sha256": hashlib.sha256(raw).hexdigest(),
                        "context_tokens": prompt.context_tokens,
                        "query_tokens": prompt.query_tokens,
                        "boundary_shift_characters": shift,
                    }
                )
            selected = select_longest_candidates(
                metadata, args.minimum_tokens, args.maximum_tokens, args.count
            )
            lengths = np.asarray(
                [item["context_tokens"] for item in metadata], dtype=np.int64
            )
            datasets[task] = {
                "member": member,
                "record_count": len(records),
                "official_metric": "qa_f1_score",
                "eligible_count": int(
                    sum(
                        args.minimum_tokens <= int(value) <= args.maximum_tokens
                        for value in lengths
                    )
                ),
                "context_token_quantiles": {
                    str(percentile): float(np.percentile(lengths, percentile))
                    for percentile in (0, 25, 50, 75, 100)
                },
                "selected": selected,
            }
            print(
                f"{task}: eligible={datasets[task]['eligible_count']} "
                f"selected={len(selected)} max={int(lengths.max())}",
                flush=True,
            )
    return {
        "schema_version": "hmo.longbench_six_task_inventory.v1",
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "archive_sha256": _sha256(archive),
        "selection": {
            "rule": "longest_exact_serialized_memory_context_within_inclusive_token_band_then_record_index",
            "minimum_tokens": args.minimum_tokens,
            "maximum_tokens": args.maximum_tokens,
            "requested_per_task": args.count,
            "outcome_conditioned": False,
            "augmentation": False,
            "truncation": False,
        },
        "datasets": datasets,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--minimum-tokens", type=int, default=8192)
    parser.add_argument("--maximum-tokens", type=int, default=16384)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = inventory(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
