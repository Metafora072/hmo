# Experiment Tracker

| Run ID | Purpose | Split | Status |
|---|---|---|---|
| R000-R004 | Probe repair and retrospective accessibility exploration | Historical oracle labels | DONE; development evidence only |
| P2-8K-O | Fresh prospective oracle acquisition | seed 20260921, 6+6 at 8K | DONE, 686 comparisons |
| P2-8K-Q | Frozen V2 evaluation | P2-8K-O | DONE, gate passed |
| P2-16K-O | Fresh prospective oracle acquisition | seed 20260922, 4+4 at 16K | DONE, 961 comparisons |
| P2-16K-Q | Frozen V2 evaluation | P2-16K-O | DONE, positive scoped result |
| P3-S | End-task smoke | fresh 1+1 at amended 4K | DONE, exact-byte contract passed |
| P3-8K | Equal-byte end-task quality | fresh 12+12 at 8K | DONE, +4.17 pp; continuation gate passed |
| P3-16K | Equal-byte end-task transfer | fresh 12+12 at 16K | DONE, -8.33 pp; final claim gate failed |
| PB-S | Structured Pareto smoke | 1 Needle at 8K, 5/10/20% | DONE; 3/3 budget cases, exact-byte contract passed |
| PB-P | Structured-baseline Pareto | matched 48-case 8K/16K suite, 5/10/20% | DONE; 144/144 budget cases, verdict partial/supplement |
| P5-S | Stratified Fixed-Chunk smoke | first frozen Needle at 16K/10% | DONE; parent plan exact, equal bytes, 56/60 Sparse windows changed |
| P5-P | Free-start mechanism isolation | frozen 12+12 suite at 16K/10% | DONE; HMO 18/24 vs aligned 17/24, verdict partial/supplement |
| P6-S | HotpotQA-32K-Aug Full-KV smoke | first frozen real-task augmented case | DONE; 32K exact, official F1 0.3333, exit 0 |
| P6-F | HotpotQA-32K-Aug solvability | four frozen Full-KV cases | DONE; mean official F1 0.2315, nonzero 2/4, GPU1 released |
| P7-S | HotpotQA-32K-Aug paired smoke | first frozen case, four compressed arms at 10% | DONE; 4/4 compressed arms exact equal bytes at 11.57% of Full |
| P7-F | HotpotQA-32K-Aug paired pilot | four frozen cases, four equal-byte arms plus Full KV | DONE; HMO F1 0.3357 at 11.556% Full, partial/supplement |
| P8-R1/R2 | persistent FP32 probe reproducibility | same first P7 case, two fresh run dirs | DONE; exact probe/position/output reuse, all invariants pass, GPU1 released |
| C2-P | final persistent-probe Pareto rerun | 0.8B, 48 cases x 3 budgets | DONE; 144/144 budget rows, exact-byte contract passed |
| C2-S | final scale transfer | 9B, 24 frozen mechanism cases at 10% | DONE; 24/24, HMO 23 vs Scattered 19, exact bytes |
| C2-N | native LongBench external validity | 12 HotpotQA + 12 NarrativeQA at 10% | DONE; 24/24, HMO competitive overall and best on NarrativeQA slice |
| C2-N6-9B | six-task native LongBench main table | Qwen3.5-9B, 506 native <=16K QA records at 10% | DONE; 506/506, exact equal bytes; HMO 0.4642 vs ChunkKV 0.4793, verdict partial |
| C3-PF | former separate paid preflight | 27B BF16, one exact-32K Needle, HMO + Full | SUPERSEDED; first frozen formal sample now carries runtime acceptance |
| C3-S | 27B mechanism and Pareto core | 12+12 exact-32K cases, 5/10/20% | FROZEN/HOLD; 312 generation cells, no paid action before new evidence and PZ confirmation |
| C3-N | 27B native QA core | same frozen C2 native cases at 10% | FROZEN/HOLD; 120 generation cells, no paid action before new evidence and PZ confirmation |
