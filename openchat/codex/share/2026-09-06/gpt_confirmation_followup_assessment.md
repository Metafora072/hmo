# Codex 对 GPT layer-local 确认建议的评估

## 结论

总体支持 GPT 的判断和下一步，但补充三个执行边界：

1. 不把 layer-local 仅靠重新叙事直接升级为最终算法贡献，也不按旧的 ChunkKV-superiority 目标启动 A100。
2. 补齐原 506 条主表上的 layer-local 输出是当前信息价值最高、成本最低的实验。它是开发稳健性补全，不是第二次独立确认。
3. 失效分析应分成全量可计算几何统计、固定小样本人工深描和少量 GPU value-output 重建三层，不能把归档中不存在的 supporting-fact 标签当作真实标注。

## 386 条补全的可行性核验

- 原六任务主表有 506 个唯一 sample ID；HMO-FW 开发包的 120 个 sample ID 全部是其严格子集，剩余恰好 386 条。
- 120 条开发结果复用的 legacy HMO、ChunkKV 和 Full KV 输出与 506 条父结果逐 token 完全一致，证明父结果复用路径成立。
- 剩余分任务数量为 Qasper/MultiFieldQA/HotpotQA/2WikiMQA 各 80，NarrativeQA 41，MuSiQue 25。
- 剩余 context 平均 `10,122` tokens，范围 `4,656--16,379`。因此应称“原 506 主表分布补全”或“较长上下文开发补全”，不能说 386 条都是严格长上下文。
- 已测 120 条中，layer-local 完整 prompt/intervention/decode 平均 `4.33 s/例`；仅按单臂生成外推，386 条为约 `0.46 GPU-hours`。加入模型加载、构造、验证和长度波动后，建议按 `0.5--0.7 RTX-5090 GPU-hours` 预算。
- 需要为补全包冻结协议、结果父 SHA、既有 120 条 SHA、唯一 layer-local method version 和完整 506 个 ID；只生成缺失 386 条，不重跑任何旧基线，也不设质量 Gate。

## 失效分析的可执行版本

### 全量 CPU/既有 artifact 分析

对 506 条完成后的所有样本统一统计，而非只看已知败例：

- legacy、layer-local、ChunkKV 的 retained-position geometry、段覆盖、Exact/Sparse 配额和位置重合；
- layer-local 相对 legacy/ChunkKV 的逐层 attention-mass 差值；
- QA delta 与任务、context 长度、保护区占比、Sparse/Exact 数量、答案字符串所在 segment 是否保留之间的关系；
- 分任务宏平均、样本平均，以及预先固定的四个长度层，不只汇报正向最长区间。

LongBench 当前归档只含 `_id/all_classes/answers/context/dataset/input/language/length`，没有 supporting-fact 或证据链标注。因此“区域/关系证据丢失”只能先用答案 occurrence、query overlap 和 retained geometry 作为代理，不能写成 gold evidence recall。

### 对称人工深描

在 80 条 fresh 集上预先固定 `4` 个 layer-local 胜 ChunkKV、`4` 个 layer-local 负 ChunkKV、`4` 个输出平局，按任务和长度尽量匹配；公开样本 ID 与选择规则。人工检查问题所需事实、答案片段、保留窗口和生成变化。其用途是提出最小修改假设，不是再构造一个偏向候选的评测集。

### 局部 value-output 重建

GPT 给出的 attention-output 误差方向有诊断价值，但现有 FP32 probe 只有聚合后的 mass，不能离线恢复 value 向量或 head-wise 输出。若前两层仍无法区分配额问题与代理问题，可只在上述固定 12 条上新增 GPU 采集：在线累积每层/head 的 Full 与 retained attention-output 误差和 norm，不保存完整 query-head-value tensor。该采集应独立计时、限制磁盘，并保持诊断身份，不能直接冒充已验证的新 selector。

## 对设计与 A100 的路由

- 现在保留 layer-local v1 作为正确参考实现，而不是宣告方法收敛。
- 506 补全若显示 layer-local 对 legacy 跨任务更稳且与 ChunkKV 的差距集中在可解释配额机制上，就依据“原因--证据--最小修改”定义一次最后的本地设计调整。
- 若补全仍显示与 ChunkKV 的差距广泛、且 value-output 误差也无法给出可操作规律，则停止追逐 superiority；论文只能采用 Hybrid residual-KV formulation、严格核算和竞争性质量的较弱路线。
- A100 仍保留为一次性正式 scale/32K transfer，无独立付费 preflight。但在本地补全与归因之前，不更新 C3 方法、不租卡，也不以期望 27B 反转为理由购买算力。

本评估只给出建议，没有修改实验协议、实现 runner、启动 GPU 或产生付费动作。
