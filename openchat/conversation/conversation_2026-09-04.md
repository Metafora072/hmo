# OpenChat Conversation: 2026-09-04

**Codex(09:20:35)**:
冻结的 contiguous CF fresh confirmation 已完整完成：48 个全新、未筛选的 8K/16K Needle+LongEval 样本中，contiguous CF 为 34/48，raw Exact 32/48，scattered CF 27/48，Sparse-only 32/48，Full KV 35/48。最干净的结论是连续窗口相对同分配器、严格等字节的离散 Top-token 提升 +14.58 pp，7 wins/41 ties/0 losses，bootstrap 95% CI [+6.25,+25.00]，exact p=0.0156，且 8K/16K 均正。相对 raw 的 +4.17 pp 只有 3 wins/1 loss，CI [-4.17,+12.50]、p=0.625，且方法多用 1.22% resident bytes；相对 Sparse-only 的 fidelity 增益也不显著。独立 internal result-to-claim verdict 为 partial/medium：支持当前模型和合成任务内的 contiguous-local-coverage 机制，不支持强 raw 优势、recurrent-aware allocation 或跨模型泛化。GPU1 已释放至 15 MiB，不自动追加 GPU 实验。详见 `codex/share/2026-09-04/contiguous_cf_fresh_confirmation_report.md`。
