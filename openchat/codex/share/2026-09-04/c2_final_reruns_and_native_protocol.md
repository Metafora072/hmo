# C2 Final 0.8B Pareto And 9B Transfer Reruns

Date: 2026-09-04

## Scope

The first two parts of the final 5090 convergence package were rerun from clean
`main@54b0290578eb59b7c9435eb8242bee6323a2413d`. Both use the persistent,
identity-bound FP32 query probe and the final HMO method contract. GPU1 was the
only visible GPU; GPU0 was not touched.

## Qwen3.5-0.8B Pareto

The formal run completed all 144 budget cases: 48 frozen 8K/16K
Needle+LongEval samples at each 5%, 10%, and 20% middle-context cap. Counts are
normalized-answer-containment successes out of 48.

| Cap | Footprint | HMO | Fixed | Raw+Slack | Scattered | Sparse | Full |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5% | 8.57% | 30 | **36** | 30 | 21 | 30 | 35 |
| 10% | 13.38% | 34 | **36** | 32 | 27 | 32 | 35 |
| 20% | 23.01% | 35 | 35 | **36** | **36** | 33 | 35 |

HMO versus Scattered is +18.75 pp at 5% (9 wins, 39 ties, 0 losses) and
+14.58 pp at 10% (7 wins, 41 ties, 0 losses). At 20% the methods saturate and
HMO trails Scattered by one case. All five compressed systems match measured
resident KV bytes in 48/48 cases at every budget.

- Run: `/mnt/nvme0/hmo/runs/c2_final_pareto_08b_54b0290/`
- Manifest ID: `2c8f59123dfb08f12fac4b26d75d4fdaaf2c331218850f1795397e56f5212676`
- Results SHA256: `7daf5799c5b330f3b88ec4158fff91e9840084aafb4a67b75fcbe8634ee2d69c`
- Summary SHA256: `b2f8bb40970ac95f9bd580be09098ee76501748406139b5577df68e63507271f`
- Runtime: 1740.16 s; peak 4.34 GiB allocated / 4.61 GiB reserved

The final FP32-probe result supersedes the old exploratory Package-B artifact
where Raw+Slack had 35 rather than 36 successes at 20%. The HMO, Fixed,
Scattered, Sparse, and Full headline counts are unchanged.

## Qwen3.5-9B Central Transfer

The formal run completed 24/24 frozen 8K/16K Needle+LongEval cases without
retuning. HMO, Raw+Slack, Sparse-only, and Full each obtain 23/24; Scattered
obtains 19/24. HMO therefore improves over the strictly equal-byte Scattered
arm by +16.67 pp (4 wins, 20 ties, 0 losses) while matching Full on the primary
metric. The mean per-case footprint is 13.38% of Full KV, and all compressed
systems subject to the equal-byte contract match in 24/24 cases.

- Run: `/mnt/nvme0/hmo/runs/c2_final_scale9b_54b0290/`
- Manifest ID: `fd84240bb771e2efc123bbe47645789459af49d8185f0c8183247d89f7aee286`
- Results SHA256: `eb7eeddd6934a56dace8eb5c370805b16e8817c27ccd9266aef20b91af22a677`
- Summary SHA256: `7b5d3915128f6341456b6182f3c41a2d9c2e496b437429b21bad853342dc2fad`
- Runtime: 744.81 s; peak 23.61 GiB allocated / 24.00 GiB reserved

## Native Package Freeze

The remaining local C2 package is frozen before outcomes in
`refine-logs/native_longbench_protocol.json`. It uses unmodified, unaugmented,
untruncated LongBench HotpotQA and NarrativeQA records, official QA F1, a 10%
cap, and the five final systems. For each dataset it deterministically chooses
the 12 longest exact serialized memory contexts within the inclusive
8,192--16,384 token band, breaking ties by source record index. A full tokenizer
audit reproduced 108 eligible HotpotQA records, 61 eligible NarrativeQA
records, and every frozen case. Protocol SHA256:
`86ebfa5cfdff0613e559780811887b7537d0485cbd00534193c0aac433b49e2a`.

No native-task outcome existed when this protocol and runner were frozen.
