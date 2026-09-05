# OpenChat Conversation: 2026-09-05

**Codex(00:59:41)**:
C3 零 GPU 收敛包已实现：支持 GPT 的 27B/32K 主目标，但把首次付费必跑范围从约 1,080 generation cells 缩为 432 cells（312 个 exact-32K 合成机制 cells + 120 个复用 C2 冻结样本的原生 QA cells）；另设 HMO+Full 两-cell preflight，只校准可运行性、显存、耗时和费用，不按答案结果设 Gate。C2 的最终 Pareto/native runner 已向后兼容地接入同一 C3 主协议，新增逐阶段计时、费用估算器、clean-commit 单卡 launcher 和唯一 C3 runbook；旧 V6.1 入口已显式锁定。聚焦 C2/C3 CPU tests 12/12 通过。尚未下载 27B 权重、启动 GPU 或租卡；下一外部动作仍需 PZ 确认单张 80GB GPU 与 preflight 成本，preflight 后再按实测确认核心费用。详见 `codex/share/2026-09-05/c3_27b_execution_package.md`。

**Codex(11:12:15)**:
根据 PZ“不在昂贵大卡上单独 preflight”的要求，已完成 HMO 的 5090 证据、理论、A 会实验形态与 A100 预算审计。结论为 conditionally ready：方法、理论边界、0.8B/9B 等字节机制证据和复现工程已经成形，不应重开设计；但 24 条/2 数据集的 native 证据、缺少 faithful public baseline、以及 resident KV 与 peak/latency 的效率口径仍需在 5090 上补齐。建议先完成 6 个 LongBench QA 数据集各 100 条的 9B 主表、一个验证过的 SnapKV 对照、以及 8K/16K 三次重复效率测量，预计共 7--13 个 5090 GPU-hours。随后在一次 A100-80GB 付费窗口直接跑 27B/32K 正式矩阵：首个冻结正式样本同时承担在线运行验收并计入结果，不设独立 preflight 或质量 Gate；中央 10% 先跑，再补 5%/20% 和 native，共 432 cells，预期 5.75--11.5 h，建议预留 12 GPU-hours、14 h 硬上限。当前冻结协议暂未修改，待 PZ 与 GPT 审阅。详见 `codex/share/2026-09-05/hmo_pre_a100_readiness_and_budget_review.md`。
