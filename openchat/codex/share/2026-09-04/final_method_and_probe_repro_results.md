# Final Method Contract And Persistent-Probe Result

Date: 2026-09-04

## Completed Scope

Approved Packages C0 and C1 are complete on main.

C0 unified the working title and paper story as HMO: Stratified KV Overlays for
Hybrid-Attention Language Models. The final method/theory contract is
docs/design/HMO_FINAL_METHOD_AND_THEORY_ZH.md. It fixes the two-level
macro-coverage plus free-start overlay, optional Exact fidelity, span-survival
proposition, max-mass corollary, approximate coverage floor, and an explicitly
stylized separable-concave allocation-regime theorem.

C1 adds one persistent, identity-bound FP32 query probe for the Hotpot, Pareto,
and scale-transfer runners. Every result row now records probe provenance and
per-arm retained-position hashes. Corrupt, incomplete, or mismatched artifacts
fail closed. The full CPU suite passes 155 tests, including seven focused cache
tests.

## GPU1 Reproducibility Result

P8 repeated the first frozen 32K Hotpot case twice from clean commit
348dff2632ccff6b4db9057ec6b33de7f8b5ae5c.

- Probe ID identical: fad4f015cbea80fd88ebdd2c808ef8b2a2713433a82e6025e9bfe54b90dd9339
- Score file SHA identical and verified:
  68bf022c3ea4f23b371da7c0f7a60c6f72c7ff93c54a29dff295a9a2b180b745
- Cache hits: R1 false, R2 true
- Retained-position hashes: identical for all four compressed arms
- Generated token IDs and F1: identical for all five systems
- Resident KV: exactly 46,657,536 bytes for every compressed arm
- Runtime: 42.01 s create, 37.92 s reuse
- GPU1 released to 15 MiB

This resolves the P7 cross-run ranking drift by exact artifact reuse and does
not introduce tolerance bucketing.

## Provenance And Next Decision

Full report: ../experiments/results/QUERY_PROBE_REPRO_20260904.md
Protocol: ../refine-logs/query_probe_repro_protocol.json
R1: /mnt/nvme0/hmo/runs/p8_probe_repro_348dff2_r1b/
R2: /mnt/nvme0/hmo/runs/p8_probe_repro_348dff2_r2b/

Package C2 has not started. The proposed next sequence remains final 0.8B
Pareto, final 9B central point, then a small native non-augmented real-task
package. Because these are broader GPU actions, PZ should confirm C2 scope
before launch.
