# Experiment Tracker

| Run ID | Purpose | Split | Status |
|---|---|---|---|
| R000-R004 | Probe repair and retrospective accessibility exploration | Historical oracle labels | DONE; development evidence only |
| P2-8K-O | Fresh prospective oracle acquisition | seed 20260921, 6+6 at 8K | DONE, 686 comparisons |
| P2-8K-Q | Frozen V2 evaluation | P2-8K-O | DONE, gate passed |
| P2-16K-O | Fresh prospective oracle acquisition | seed 20260922, 4+4 at 16K | DONE, 961 comparisons |
| P2-16K-Q | Frozen V2 evaluation | P2-16K-O | DONE, positive scoped result |
| P3-S | End-task smoke | fresh 1+1 at 2K | READY |
| P3-8K | Equal-byte end-task quality | fresh 12+12 at 8K | BLOCKED on P3-S |
| P3-16K | Equal-byte end-task transfer | fresh 12+12 at 16K | BLOCKED on P3-8K gate |
