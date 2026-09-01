# OpenChat Conversation: 2026-09-01

**PZ(22:48:39)**:
请参考 GPT 对 HMO 当前设计的评估，判断它是否合理，并决定下一阶段应补设计还是补实验。随后在 HMO 中初始化 OpenChat，沉淀结论并推送到 GitHub。

**Gpt(22:48:39)**:
建议冻结 V6.1，不再堆叠 policy heuristic；先修正 benchmark metric 和 baseline fidelity，再把 E3 机制实验作为第一优先级，验证 `phi = sigma * alpha` 是否比 attention-only signal 更能预测显式 KV 需求。完整评估见 `gpt/share/2026-09-01/hmo_design_review.md`。

**Codex(22:48:39)**:
同意“停止扩展 action policy、以机制证据为下一主线”，但现有 E3 不能直接开跑。当前 `sigma` 与文档声称的 recurrent-memory reliability 存在公式和时间语义差距；E3 oracle 不是严格同预算对照，且首 token 仍来自 Full-KV prefill。决策是冻结动作策略，先修 metric、`sigma` 和 E3 protocol，再做小规模 pilot；只有 `phi` 在控制 `alpha` 后仍稳定提供增量预测力，才投入 27B、跨模型和系统实验。完整核查、路线与 stop conditions 见 `codex/share/2026-09-01/hmo_review_assessment.md`。

**Gpt(22:58:46)**:
接受 Codex 的 gate，并核对确认当前 `sigma` 存在 actual delta residual 缺失、`1/(-g)` 与 `exp(g)` retention 语义方向冲突、以及只看局部 segment 而未建模 suffix survival 三个核心问题；现有 E3 也不是严格固定预算且首 token 复用 Full-KV prefill logits。下一步不直接跑现有 E3，也不先拍定新版 `sigma`：冻结 V6.1 action，先修 official metric / manifests / fixed-budget post-intervention E3-v2，并把 current sigma、actual-delta、cumulative-retention、suffix-interference 作为候选在小模型近 exhaustive pilot 中比较。当前 eager Refresh 暂从核心 oracle 和必要组件 claim 降级为 ablation，待机制 gate 通过后再决定是否重构。详细回应见 `gpt/share/2026-09-01/codex_followup_response.md`。

**Codex(23:20:32)**:
接受 GPT 的收敛意见，并将 E3-v2 固化为预注册协议。新增三项硬约束：context KV 必须在 query suffix 处理前完成干预，signal discovery 与 held-out confirmation 必须分离，segment utility 必须跨多个等字节 donor/background 聚合。协议同时冻结主要标签、统计单位、完整性测试和 PASS/HOLD/KILL 门槛；现有 E1/E3 与 GPU 运行继续 BLOCK，下一实现包从 official metrics + immutable manifest 开始。详见 `codex/share/2026-09-01/e3_v2_preregister.md`。
