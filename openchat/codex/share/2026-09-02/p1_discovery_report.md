# HMO E3-v2 P1 Discovery Report

Date: 2026-09-02  
Branch: `dev/e3-v2-p1-discovery`  
Implementation commits: `3bf58ee`, `271256a`, `947e3fc`

## Decision

P1 supports a **partial scientific claim**, but does not yet support the current HMO controller formula.

- Recurrent-state measurements contain incremental information beyond attention mass and position.
- The useful signal is objective- and task-dependent: historical `sigma_current` helps broad pairwise ordering, while `phi_delta_alpha` and `surviving_write_norm` mainly help top-budget NDCG on LongEval.
- None of the tested training-free fusion formulas consistently improves over raw attention score `alpha`.
- Therefore the next step is a minimal controller redesign, not a larger confirmation run. Running 16K or a larger model now would amplify an unresolved scoring rule.

This is a discovery result, not a confirmation result.

## Lightweight ARIS Execution

The run followed the current ARIS experiment-bridge logic with deliberately lightweight settings:

- local single-GPU deployment on physical GPU1;
- no duplicate code-review gate because the implementation and focused/full CPU regression had already passed;
- no duplicate sanity-only run because the real-model preflight had already passed 8/8 checks;
- primary oracle quality only, no secondary generation per arm;
- append-and-fsync pair observations with resume support;
- sample-grouped cross-validation and sample-level bootstrap;
- large artifacts kept under `/mnt/nvme0/hmo`.

These choices removed process overhead without weakening the experiment's provenance, intervention semantics, or uncertainty unit.

## Experiment Configuration

| Item | Value |
|---|---|
| Model | `Qwen/Qwen3.5-0.8B` |
| Revision | `2fc06364715b967f1860aea9cf38778875588b17` |
| Context / segment | 8192 / 256 tokens |
| KV budget | 10% |
| Tasks | Needle, LongEval Lines |
| Samples | 12 total: 6 per task |
| Oracle comparisons | 686 |
| Segment evidence rows | 360 |
| Donors / backgrounds | 2 / 1 |
| Recurrent backend | `transformers_torch_reference` |
| Bootstrap | 5000, grouped by sample in combined analysis |

Source runs:

| Seed | Samples | Pairs | Runtime | Peak reserved VRAM | Manifest |
|---:|---:|---:|---:|---:|---|
| 20260902 | 4 | 230 | 756.48 s | 2.74 GiB | `ed2262672a08...6390bc` |
| 20260903 | 8 | 456 | 1294.60 s | 2.74 GiB | `6b60a7df4caa...f296a` |

Raw artifacts:

- `/mnt/nvme0/hmo/runs/p1_discovery_qwen08b_8k_20260902_192652`
- `/mnt/nvme0/hmo/runs/p1_discovery_qwen08b_8k_seed20260903_20260902_194216`
- `/mnt/nvme0/hmo/runs/p1_discovery_combined_12samples_20260902.json`
- `/mnt/nvme0/hmo/runs/p1_surviving_enrichment_12samples_20260902_203045.json`

All 12 samples had nonzero oracle-label range. No sample was dropped for a degenerate target.

## Learned Incremental Diagnostics

Each candidate was added to the `alpha + normalized_position` baseline using sample-grouped ridge diagnostics. Values below are mean improvement with 95% sample-bootstrap intervals.

| Candidate | Pairwise delta | NDCG delta | Interpretation |
|---|---:|---:|---|
| `sigma_current` | **+0.0257** `[+0.0021,+0.0494]` | -0.0009 `[-0.0965,+0.0907]` | Best broad ordering signal; positive pairwise delta on both tasks |
| `phi_delta_alpha` | +0.0178 `[-0.0076,+0.0473]` | **+0.0881** `[+0.0272,+0.1544]` | Best top-budget diagnostic; NDCG gain is LongEval-led |
| `suffix_interference` | +0.0190 `[-0.0184,+0.0554]` | +0.0189 `[-0.0235,+0.0810]` | Suggestive but inconclusive |
| `delta_update` | +0.0109 `[-0.0105,+0.0427]` | -0.0542 `[-0.1302,+0.0047]` | Does not improve the target consistently |
| `survival_retention` | +0.0015 `[-0.0025,+0.0059]` | 0.0000 `[0,0]` | No useful evidence |
| `phi_sigma_alpha` | -0.0071 `[-0.0157,+0.0008]` | -0.0259 `[-0.0847,+0.0128]` | Unsupported |

The `phi_delta_alpha` NDCG improvement splits into `+0.1771` on LongEval and `-0.0010` on Needle. This task asymmetry is material and should not be hidden by the pooled mean.

## Direct Controller Checks

The learned diagnostics show whether information exists; they do not prove that a hand-written product is a usable controller. Direct, training-free formulas were therefore compared against raw `alpha` using the same samples.

| Formula | Pairwise delta | NDCG delta | Result |
|---|---:|---:|---|
| `alpha * sigma_current` | +0.0013 `[-0.0040,+0.0061]` | +0.0155 `[-0.0052,+0.0517]` | Indistinguishable from baseline |
| `alpha * (1-rank01(delta_update))` | -0.1035 `[-0.1544,-0.0475]` | -0.3312 `[-0.5033,-0.1680]` | Clearly harmful |
| `alpha * (1-rank01(surviving_write_norm))` | -0.0900 `[-0.1448,-0.0391]` | -0.1602 `[-0.4262,+0.1202]` | Harmful overall and task-unstable |

The last formula improves LongEval NDCG by `+0.3007` but decreases Needle NDCG by `-0.6212`; pairwise accuracy decreases on both tasks. This is evidence of task-conditioned structure, not evidence for the formula.

## P1.5 Signal Enrichment

P0-C already computed the final surviving recurrent contribution but P1 did not aggregate or persist it. Commit `947e3fc` adds `surviving_write_norm` to the recurrent-candidate contract and provides a signal-only enrichment runner. It reused all existing oracle labels and did not repeat the 686 pair interventions.

The enrichment covered all 12 samples in 17.54 seconds, with 2.35 GiB allocated and 2.62 GiB reserved at peak. Its learned diagnostic produced:

- pairwise delta: `-0.0140` `[-0.0588,+0.0268]`;
- NDCG delta: `+0.0671` `[-0.0060,+0.1527]`;
- task NDCG: LongEval `+0.1555`, Needle `-0.0214`.

This closes the missing-signal check cheaply. It does not rescue the direct controller.

## Scientific Interpretation

### Supported

1. Attention-only importance is incomplete for hybrid attention/recurrent models.
2. Recurrent-state evidence has measurable incremental association with oracle segment utility.
3. The relevant recurrent statistic changes with the ranking objective and task family.
4. Naive multiplicative coupling is not a reliable way to turn that evidence into a controller.

### Not supported

1. The existing `alpha * sigma` HMO score beats raw `alpha`.
2. Actual DeltaNet update size or surviving-write magnitude is monotonically equivalent to discard risk.
3. A single global, training-free recurrent penalty transfers across Needle and LongEval.
4. The discovery effects will persist at 16K, larger models, or full downstream generation quality.

## Paper Framing

A defensible paper story is: **hybrid memory compression needs state-aware scoring, but recurrent importance is conditional rather than a universal scalar penalty**. The oracle and diagnostics expose where attention-only scoring fails, and the negative direct-fusion result motivates a calibrated or conditional controller.

The paper should not present the current multiplication rule as validated. A stronger method section needs one frozen deployable scorer that is chosen before confirmation and evaluated against `alpha`, recency, random, and the original `alpha * sigma` baseline.

## Next Action

Stop formula fishing on these 12 samples. Use them only to choose one minimal redesign:

- retain `alpha` as the base score;
- add a bounded recurrent correction rather than multiplication;
- allow the correction to depend on query/task-observable features, but not oracle labels at inference;
- freeze its form and hyperparameters before any new confirmation data;
- then run one modest held-out 8K confirmation before considering 16K or a larger model.

No additional GPU experiment was launched after this report. Physical GPU1 returned to 15 MiB and 0% utilization.
