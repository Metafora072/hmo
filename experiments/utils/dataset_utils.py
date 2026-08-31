"""
HMO Research — Dataset Utilities
Load and prepare datasets for experiments:
  - Needle-in-a-Haystack (synthetic)
  - LongBench subsets (HotpotQA, NarrativeQA, GovReport, LCC)
  - LongEval-Lines (passkey retrieval)
"""
import json
import os
import random
import zipfile
from pathlib import Path
from dataclasses import dataclass
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(
    os.environ.get("HMO_RESULTS_ROOT", str(PROJECT_ROOT / "experiments/results"))
)
LONG_BENCH_REPO = "THUDM/LongBench"


@dataclass
class EvalSample:
    """A single evaluation sample."""
    dataset: str
    sample_id: str
    context: str          # the long context
    question: str         # the question/prompt
    answer: str           # ground truth answer
    context_length: int   # target context length in tokens
    answers: list[str] | None = None  # optional list of all valid answers


def make_needle_samples(
    tokenizer,
    n_samples: int = 50,
    context_length: int = 32768,
    seed: int = 42,
) -> list[EvalSample]:
    """
    Generate Needle-in-a-Haystack samples.
    Insert a fact at a random depth in repeated filler text, then ask about it.
    """
    rng = random.Random(seed)

    # Filler text (Paul Graham essays style)
    filler = (
        "The most important thing in life is to learn how to give out love, "
        "and to let it come in. Technology is nothing. What's important is that "
        "you have a faith in people, that they're basically good and smart, "
        "and if you give them tools, they'll do wonderful things with them. "
        "Innovation distinguishes between a leader and a follower. "
        "Stay hungry, stay foolish. The people who are crazy enough to think "
        "they can change the world are the ones who do. "
    )

    # (needle_sentence, question, answer_extractor)
    needles = [
        ("The secret code is: ALPHA-{rid}.", "What is the secret code?", "ALPHA-{rid}"),
        ("The meeting is scheduled for {rid} o'clock.", "What time is the meeting scheduled for?", "{rid} o'clock"),
        ("The password to the vault is {rid}-BRAVO.", "What is the password to the vault?", "{rid}-BRAVO"),
        ("The project deadline is March {rid}th.", "What is the project deadline?", "March {rid}th"),
        ("The special ingredient is {rid} grams of saffron.", "What is the special ingredient amount?", "{rid} grams"),
    ]

    filler_tokens = tokenizer.encode(filler, add_special_tokens=False)
    samples = []

    for i in range(n_samples):
        rid = rng.randint(100, 999)
        needle_template, question, answer_template = rng.choice(needles)
        needle_text = needle_template.format(rid=rid)
        answer = answer_template.format(rid=rid)  # short extractive answer
        needle_tokens = tokenizer.encode(needle_text, add_special_tokens=False)

        # Calculate how many filler repetitions we need
        available = context_length - len(needle_tokens) - 50  # margin for question
        n_reps = available // len(filler_tokens) + 1

        # Build haystack
        haystack_tokens = (filler_tokens * n_reps)[:available]

        # Insert needle at random depth (10%-90%)
        depth = rng.uniform(0.1, 0.9)
        insert_pos = int(len(haystack_tokens) * depth)

        full_tokens = haystack_tokens[:insert_pos] + needle_tokens + haystack_tokens[insert_pos:]
        full_tokens = full_tokens[:context_length - 50]

        context = tokenizer.decode(full_tokens, skip_special_tokens=True)

        samples.append(EvalSample(
            dataset="needle",
            sample_id=f"needle_{i:04d}",
            context=context,
            question=question,
            answer=answer,
            answers=[answer],
            context_length=context_length,
        ))

    logger.info(f"Generated {len(samples)} Needle samples @ {context_length} tokens")
    return samples


def make_longeval_lines_samples(
    tokenizer,
    n_samples: int = 80,
    context_length: int = 32768,
    n_lines: int = 100,
    seed: int = 42,
) -> list[EvalSample]:
    """
    Generate LongEval-Lines (passkey retrieval) samples.
    Each sample has numbered lines with random register values, padded with
    filler to reach context_length. The question asks for a specific line's value.
    """
    import string
    rng = random.Random(seed)
    chars = string.ascii_uppercase + string.digits

    filler = (
        "The most important thing in life is to learn how to give out love, "
        "and to let it come in. Technology is nothing. What's important is that "
        "you have a faith in people, that they're basically good and smart, "
        "and if you give them tools, they'll do wonderful things with them. "
    )
    filler_tokens = tokenizer.encode(filler, add_special_tokens=False)

    samples = []
    for i in range(n_samples):
        # Generate lines with random 6-char register values
        values = {}
        lines_text = []
        for line_num in range(1, n_lines + 1):
            val = "".join(rng.choices(chars, k=6))
            values[line_num] = val
            lines_text.append(f"line {line_num}: REGISTER_CONTENT is {val}.")

        lines_block = "\n".join(lines_text)
        lines_tokens = tokenizer.encode(lines_block, add_special_tokens=False)

        # Pad with filler to reach context_length
        available = context_length - len(lines_tokens) - 50
        if available > 0:
            n_reps = available // len(filler_tokens) + 1
            pad_tokens = (filler_tokens * n_reps)[:available]
        else:
            pad_tokens = []

        # Insert lines block at random depth (20%-80%) within filler
        depth = rng.uniform(0.2, 0.8)
        insert_pos = int(len(pad_tokens) * depth)
        full_tokens = pad_tokens[:insert_pos] + lines_tokens + pad_tokens[insert_pos:]
        full_tokens = full_tokens[:context_length - 50]

        context = tokenizer.decode(full_tokens, skip_special_tokens=True)

        # Ask about a random line
        target_line = rng.randint(1, n_lines)
        question = f"What is the REGISTER_CONTENT in line {target_line}?"
        answer = values[target_line]

        samples.append(EvalSample(
            dataset="longeval_lines",
            sample_id=f"longeval_{i:04d}",
            context=context,
            question=question,
            answer=answer,
            answers=[answer],
            context_length=context_length,
        ))

    logger.info(f"Generated {len(samples)} LongEval-Lines samples @ {context_length} tokens")
    return samples


def _load_longbench_records_via_hub_zip(subset: str) -> list[dict]:
    """Fallback loader for LongBench when remote dataset scripts are unsupported."""
    from huggingface_hub import snapshot_download

    snapshot_path = snapshot_download(
        repo_id=LONG_BENCH_REPO,
        repo_type="dataset",
        allow_patterns=["data.zip"],
    )
    zip_path = Path(snapshot_path) / "data.zip"
    inner_path = f"data/{subset}.jsonl"

    records = []
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(inner_path) as handle:
            for line in handle:
                records.append(json.loads(line.decode("utf-8")))
    return records


def load_longbench_subset(
    subset: str,
    tokenizer,
    n_samples: int = 80,
    context_length: int = 32768,
    seed: int = 42,
) -> list[EvalSample]:
    """
    Load a LongBench subset via HuggingFace datasets.

    Args:
        subset: one of "hotpotqa", "narrativeqa", "gov_report", "lcc"
        n_samples: max samples to load
        context_length: target context length (truncate/filter)
    """
    subset_map = {
        "hotpotqa": "hotpotqa",
        "narrativeqa": "narrativeqa",
        "gov_report": "gov_report",
        "lcc": "lcc",
    }

    if subset not in subset_map:
        raise ValueError(f"Unknown subset: {subset}. Available: {list(subset_map.keys())}")

    records = None
    try:
        from datasets import load_dataset

        ds = load_dataset(LONG_BENCH_REPO, subset_map[subset], split="test")
        records = [ds[idx] for idx in range(len(ds))]
        logger.info(f"Loaded LongBench-{subset} through datasets.load_dataset")
    except Exception as exc:
        logger.warning(
            f"datasets.load_dataset failed for LongBench-{subset}: {exc}. "
            "Falling back to local data.zip loader."
        )
        records = _load_longbench_records_via_hub_zip(subset_map[subset])

    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)

    samples = []
    for idx in indices:
        if len(samples) >= n_samples:
            break
        item = records[idx]
        context = item["context"]
        ctx_len = len(tokenizer.encode(context, add_special_tokens=False))

        # Skip if too short (< 50% of target) or too long (> 150%)
        if ctx_len < context_length * 0.3 or ctx_len > context_length * 1.5:
            continue

        # Truncate if needed
        if ctx_len > context_length:
            tokens = tokenizer.encode(context, add_special_tokens=False)[:context_length]
            context = tokenizer.decode(tokens, skip_special_tokens=True)

        answers = item["answers"] if isinstance(item["answers"], list) else [item["answers"]]

        samples.append(EvalSample(
            dataset=f"longbench_{subset}",
            sample_id=f"{subset}_{idx:04d}",
            context=context,
            question=item["input"],
            answer=answers[0] if answers else "",
            answers=answers,
            context_length=context_length,
        ))

    logger.info(f"Loaded {len(samples)} LongBench-{subset} samples (target {context_length} tokens)")
    return samples
