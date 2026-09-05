# OpenChat Conversation: 2026-09-05

**Codex(00:59:41)**:
C3 零 GPU 收敛包已实现：支持 GPT 的 27B/32K 主目标，但把首次付费必跑范围从约 1,080 generation cells 缩为 432 cells（312 个 exact-32K 合成机制 cells + 120 个复用 C2 冻结样本的原生 QA cells）；另设 HMO+Full 两-cell preflight，只校准可运行性、显存、耗时和费用，不按答案结果设 Gate。C2 的最终 Pareto/native runner 已向后兼容地接入同一 C3 主协议，新增逐阶段计时、费用估算器、clean-commit 单卡 launcher 和唯一 C3 runbook；旧 V6.1 入口已显式锁定。聚焦 C2/C3 CPU tests 12/12 通过。尚未下载 27B 权重、启动 GPU 或租卡；下一外部动作仍需 PZ 确认单张 80GB GPU 与 preflight 成本，preflight 后再按实测确认核心费用。详见 `codex/share/2026-09-05/c3_27b_execution_package.md`。

**Codex(11:12:15)**:
根据 PZ“不在昂贵大卡上单独 preflight”的要求，已完成 HMO 的 5090 证据、理论、A 会实验形态与 A100 预算审计。结论为 conditionally ready：方法、理论边界、0.8B/9B 等字节机制证据和复现工程已经成形，不应重开设计；但 24 条/2 数据集的 native 证据、缺少 faithful public baseline、以及 resident KV 与 peak/latency 的效率口径仍需在 5090 上补齐。建议先完成 6 个 LongBench QA 数据集各 100 条的 9B 主表、一个验证过的 SnapKV 对照、以及 8K/16K 三次重复效率测量，预计共 7--13 个 5090 GPU-hours。随后在一次 A100-80GB 付费窗口直接跑 27B/32K 正式矩阵：首个冻结正式样本同时承担在线运行验收并计入结果，不设独立 preflight 或质量 Gate；中央 10% 先跑，再补 5%/20% 和 native，共 432 cells，预期 5.75--11.5 h，建议预留 12 GPU-hours、14 h 硬上限。当前冻结协议暂未修改，待 PZ 与 GPT 审阅。详见 `codex/share/2026-09-05/hmo_pre_a100_readiness_and_budget_review.md`。

**Gpt(13:33:15):**
已审阅 main@bb23fbf 的 pre-A100 readiness/budget 与实际 runner。支持 5090 承担 9B 广覆盖主表、A100 只做 27B 正式规模验证，不设独立付费 preflight。首个冻结正式样本计入结果、执行常规字节/有限值/落盘检查后自动继续；只在租卡前由 PZ 一次确认费用。C0–C2 与 persistent FP32 probe 已完成，不再追加 tolerance bucket 或重复已有 smoke。

建议合并补齐三项：9B 原生任务与最接近公开基线（优先核对 ChunkKV，SnapKV 为强对照）；明确 span/凹效用定理与实际异成本 attention allocator 的对应，6.25% 是覆盖成本门槛而非已证明准确率相变；将取消 preflight 真正落实到协议、launcher 和估算器。代码仍在冷 probe 中计算主方法已不使用的 recurrent accessibility，且各臂重跑完整 prompt，须先在 5090 审查等价简化并按真实路径预算。

432 cells 算术正确，建议顺序改为合成中央10%→原生中央10%→合成5/20%；原生QA并非全部32K。为保持额度，可用已验证公开基线替换27B Scattered，不扩大矩阵。8.6h目前是外推草案；12h预留、14h拟议上限需用本地分阶段计时和目标配置假设复算，不作完成保证。保持现有设计，不以每个任务显著为门槛；同时用对照验证已有工作之上的真实增量。详细审阅、小时表和执行改动见 gpt/share/2026-09-05/gpt_pre_a100_readiness_budget_assessment_20260905.md。

**Codex(13:44:34)**:
支持 GPT 的总体结论与执行修正，并核实其 runner 判断：当前正式路径确实在 `use_accessibility=false` 时仍采集 recurrent contribution、每个实验臂重跑完整 prompt、成本估算器仍绑定旧两-cell preflight。还需加重一个风险：ChunkKV 已是 NeurIPS 2025 正式论文，其核心就是用 attention-ranked contiguous chunks 避免离散 token 破坏语义，并有 NVIDIA `kvpress` 实现；因此 HMO 不能再把“连续性有益”本身作为主要新颖性，贡献必须收紧到 stratified macro-region coverage、区域内 free-start、混合粒度 Exact upgrade 与 Hybrid residual-KV 组合。下一包应先审计/适配 ChunkKV，并用它验证真实增量；SnapKV 为第二公开基线。理论需修正 tie-break 与 `O(T+n log n+N_keep)` 口径；SAGE-KV 应标为 ICLR 2025 workshop，而非 main paper。建议 5090 上先做 attention-only probe 等价验证、ChunkKV adapter 验证、六任务冻结 50+50 前缀主表和分阶段计时；A100 仍保持 432 cells，优先顺序为 synthetic 10%→native 10%→synthetic 5/20%，由验证后的 ChunkKV 替换 Scattered，不设独立 preflight。未修改代码/协议或启动 GPU，具体实施仍待 PZ 确认。详见 `codex/share/2026-09-05/gpt_pre_a100_review_followup_assessment.md`。

**Codex(14:45:33)**:
PZ 确认后的 pre-A100 合并包已实现。理论 tie/复杂度已修正；正式 runner 改为 attention-only FP32 probe v2，旧 hybrid probe 保留；新增逐 Full 层、10-token chunk、严格等 resident-byte 的 ChunkKV hybrid adapter，并以 NeurIPS 2025 外部 baseline 身份替换新正式表中的 Scattered。GPU1 实模验证通过：新旧聚合分数 bitwise equal、最大误差 0、HMO 保留位置和生成 token 完全相同，ChunkKV 与 HMO post-query KV 均为 15,249,408 bytes，进程结束后 GPU1 已释放。9B 六任务官方 prompt/QA-F1/max-gen 已补齐，按不截断且 exact memory context <=16K 冻结实际 506 条，预定 prefix50=295、prefix100=506，同 manifest 续跑且无结果 Gate。C3 升级为 v2，执行顺序固定为 synthetic 10%→native 10%→synthetic 5/20%，首样本即正式结果，无独立 preflight，总量仍 432 cells；计时拆分 prompt/decode。完整 177 项测试通过，历史 Scattered control 由 byte-exact legacy SHA 保留。未下载 27B、未租 A100。详细实现、协议哈希与小时预算见 `codex/share/2026-09-05/hmo_pre_a100_package_implementation_report.md`。
