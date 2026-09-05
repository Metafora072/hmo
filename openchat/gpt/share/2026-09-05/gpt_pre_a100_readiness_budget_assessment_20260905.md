# GPT 审阅：5090 收敛、算法贡献与无独立 Preflight 的 A100 正式实验

日期：2026-09-05  
作者：GPT  
审阅对象：Codex《HMO: 5090 Evidence Closure and One-Shot A100 Budget Review》  
仓库基准：`main@bb23fbf`  
工作性质：设计、理论和实验就绪审阅；未启动 GPU、未租卡、未修改远端仓库。

## 1. 结论与当前决策

支持保留 HMO 方向，支持“5090 承担方法验证与广覆盖主实验，A100 承担 27B 规模确认”。不另设付费 preflight，不在租卡后等待第二轮费用批准；第一条正式样本就进入结果表，同时执行正常的有限值、字节、落盘和显存检查。

当前建议是完成一个合并的本地收尾包，再由 PZ 一次确认正式矩阵和费用上限。不是因为局部结果不显著而暂停，而是因为还有三个具体事项未闭合：公开强基线、理论与实际算法的对应、正式执行成本。不要继续重跑已完成的 C0–C2、重复 smoke 或要求每个任务置信区间排除零。

A100 是必做的最终规模证据，不是探索设计的环境。理论和 5090 实验可以降低放大失败的风险，但不能提前保证未测过的 27B 任务效果或硬件组合必定成功。

## 2. 最新进度：已经完成的事情不要再列为待办

C0/C1 已统一最终方法与理论文档，并实现按模型 revision、token/query identity、聚合版本绑定的持久化 FP32 query probe。两次 P8 复现中，保留位置、输出和字节一致。默认不需要再增加 tolerance bucket 或量化阈值；那会引入新的方法参数。

C2 已完成最终 0.8B Pareto 和 9B 中央设置复跑。连续相对散点的主要效应仍在。

C2-native 已完成 24 条未增广、未裁剪的真实 QA：**模型是 0.8B，不是 9B**。每任务 12 条，实际序列化上下文为 13,701–16,302 token。HMO / Global Fixed / Raw+Slack / Scattered / Full 的总体 F1 为 0.3086 / 0.3251 / 0.2873 / 0.3211 / 0.2602。HMO 在 NarrativeQA 上最高，但 HotpotQA 由 Fixed、Scattered 领先。该结果支持进一步做 9B 任务覆盖，不支持“原生任务验证已经全面完成”。[R2]

因此，下一轮真正新增的证据应是 9B 的较广原生任务与公开基线，而不是再次证明既有 0.8B/9B 合成结果。

## 3. 算法 A 会定位：支持增量贡献，但需要验证增量本身

已有工作 plus 是可行定位。HMO 无需声称 chunk、连续窗口、attention score 或分层预算全部首次提出。应把贡献明确到：区域覆盖的组织方式、区域内自由起点窗口，以及它们在已训练 Hybrid 模型剩余 KV 上的组合价值。

有两类证据要分开：

- HMO 比同分配器的 Scattered 更好，验证了连续保留的结构价值；这是很好的机制消融，但不能独自证明相对 ChunkKV 等已有结构化方法的新贡献。
- HMO 相对 Global Fixed / Stratified Fixed / 公开结构化方法的表现，才直接支撑“已有工作 plus”的增量价值。目前这部分是局部、任务相关的正向证据，值得补强，而不是应被判死。

保留“Hybrid residual-KV organization”的问题定义。recurrent state 未被删除不等于证明它已经完整承担全局语义，更不等于证明这种分工对其他结构没有用。现阶段把它写成架构出发点，算法优势交给对照实验说明。第二个 Hybrid 家族是扩展项，不是本轮必须先满足的通行证。

不建议再使用“必须达到六任务各 100 条才够 A 会”这种固定标准。任务数和样本数是证据设计选择，不是会议门槛。真正要回答的是：增量设计是否有可复现价值，最接近方法是否公平比较，优势和适用范围是否能被解释。

## 4. 理论审阅：哪些已经成立，哪些仍是解释模型

### 4.1 连续证据存活：成立

固定保留 k 个位置，证据为同一 segment 内长度 ell 的连续区间，2≤ell≤k。若保留集合的连续 run 长度为 r_j，则：

`N_ell(R) = sum_j max(r_j - ell + 1, 0) <= k - ell + 1`。

单个连续窗口达到上界。该结论很好地解释固定 token 数下连续结构的优势。uniform-start 假设用于把计数转换成概率；它不是所有自然任务的事实。[R3]

### 4.2 最大 attention-mass 窗口：成立，但目标有限

free-start 搜索能在给定长度的连续窗口集合中找到 attention mass 最大者。这里优化的是 attention mass，不是已知的任务正确率。可以称为窗口选址的精确求解，不要据此声称整个 HMO 是下游最优分配。

### 4.3 区域凹效用模型：定理与具体实现要分开

文档中的最优 greedy 证明依赖：等成本单位、可分效用、单调离散凹性、已知真实边际收益。该理论切片成立，但实际 HMO 使用 attention demand、整段 Exact 的异成本动作、固定 coverage-first 与 slack 分配，不是直接观测并排序真实任务边际收益。

另一个容易混淆的点是，完整 span 的收益本身可能有互补性。例如保留一个 token 没用、两个一起才有用，就不是 token 级凹效用。因此局部 span 完整性与宏观区域凹效用应在不同抽象层次建模；不能无说明地把后者假设施加到每一个 token。

等号情况下，若要保证“先覆盖再升级”，理论 greedy 的 tie-break 应优先尚未覆盖的区域；否则应表述为“存在一个 coverage-first 的最优分配”。这只需修订证明措辞，不需要重做算法。

### 4.4 6.25% 是覆盖成本门槛，不是已证明的准确率相变

`w/L = 16/256 = 6.25%` 说明何时预算能够为所有区域提供一个 base window。它不能推出所有任务在该点改变优劣，也不能推出长度增加必然有利于 HMO。

当前 8K/16K 与三个预算点是预算—长度交互的经验线索，应保持完整报告。不要为“相变”另造控制器，也不要把单个有利切片命名为普适定律。

### 4.5 时间复杂度更正

最终理论文档已有 `O(T)` 聚合/滑窗与 `O(n log n)` 区域排序；审计报告的整体 `O(T)` selection 太宽。对当前排序实现，写：

`O(T + n log n + N_keep), n ≈ T/L`，另计模型 probe 与推理成本。

固定 L 时，不能把区域排序项在渐近意义上自动抹掉。retained KV 仍是较小系数的线性空间。[R3]

结论：理论足以支撑结构动机和受限子问题，尚不是具体 HMO 任务最优性的证明。把这个映射写清楚即可，不要求补一套与实际实现脱节的大定理。

## 5. 公开基线：优先最接近工作，而不是只看名字知名度

优先核对 ChunkKV 或最接近的公开 chunk-retention 实现；它直接回应当前新颖性问题。SnapKV 很值得加入，作为成熟且有竞争力的公开基线，但不能替代最接近结构化工作的位置。PyramidKV/CAKE 的跨层分配可以放扩展，不要一轮实现全部。

建议本地做一次代码和成本比较，确定一个核心公开基线；若实现成本可控，再加 SnapKV 作为第二个公开基线。用已有小样本验证适配，随后纳入同一批 9B 主实验，不为每个方法另开审批。

公平性需要区分两类实验：

1. 机制控制：共享 query score、anchors、窗口数量和真实 bytes，单独改变结构。
2. 算法比较：尊重公开算法自己的 observation window、pooling、per-head/per-layer 选择，核算相同真实 KV 预算。不要强迫所有方法共享一个已经跨头平均的 score，以免把公开算法改成弱化控制。

Hybrid 适配应说明具体改动；Global Fixed-Chunk 是项目内控制，不应直接改名为官方 ChunkKV。外部方法在 5090 上确认后才能进入付费包。[R1,R10]

## 6. 一个合并的 5090 收尾包

建议两条线并行，完成后只做一次付费就绪决策。

### 6.1 9B 任务覆盖与基线

候选任务为 NarrativeQA、Qasper、MultiFieldQA-en、HotpotQA、2WikiMultihopQA、MuSiQue。先做 CPU 长度普查：当前安全长度范围不保证每个任务恰好有 100 条未裁剪样本。冻结合法的选择方式、上下文长度策略和样本 ID，不因模型是否答对而筛选。

默认中央 10% 预算。先完成六任务各 50 条的完整同口径表，再按预定队列扩展到每任务 100 条；它们是同一实验包的样本前缀，不是达到正结果就停止的顺序检验。样本数不足时报告实际数量或明确统一截断协议，不能为了“32K”拼接无关文本再当原生任务。

主表优先 HMO、最接近公开结构化基线、Raw+Slack、Full；Global Fixed 与 Scattered 可在预先固定的子集用于机制解释。资源足够时将成熟 SnapKV 纳入整表。不要将扩大所有内部 ablation 的算力优先于加入公开基线。

Codex 原案的 7–13 个 5090 GPU-hours 可作为暂定资源范围，但应利用已有阶段计时更新。六任务各 100 条、五系统就是 3,000 次生成，追加一个公开基线再增加 600 次。5–8 h 主表意味着每条样本的五臂总耗时平均约 30–48 s；现有依据不足以将这个速度视为 9B 长答案 QA 的保证。

### 6.2 正式成本与显存路径

本轮静态阅读发现：

- `run_pareto.py` 对每个比较臂调用 `run_post_intervention_prompt`，每臂会重新执行完整 prompt，而不只是共享 KV 后短解码。[R6]
- 同一 runner 仍调用 `collect_hybrid_query_token_probe`，后者挂接 `Qwen35QueryAccessibilityHookManager`，计算每层、每段 recurrent contribution；主方法却已设 `use_accessibility=false`。[R6,R7]
- runner 显式选择 reference recurrent backend。不能用未采用的融合 kernel 的性能预期估算预算。

优先审查能否将 recurrent contribution 采集改为独立诊断选项，让正式 HMO 使用同一 attention 定义的轻量 probe。先在 5090 验证 score 定义、保留位置和生成一致性；这是移除已退出方法的采集开销，不是更换算法。不应通过伪造零 accessibility 值来掩盖旧接口。若改动不经济，保留原路径并如实计入成本。

前缀 cache 共享可省计算，但也会增加并存缓存与状态隔离复杂度。本周不将全面缓存重构设为新前置任务。重复 prefill 的现状可以保留，只要预算按实际路径计算。

效率表区分：冷请求的 probe/压缩开销、持久化 probe 的评测复用收益、decode resident KV、peak VRAM、TTFT、decode ms/token。持久化 probe 不能让真实部署的冷启动成本从表里消失；9B 的 score 也不能复用为 27B score。

## 7. A100 不单独 preflight：需要修改真实入口

支持 Codex 最新文字方案：首个正式样本承担正常运行检查，其全部方法输出计入最终表，不额外创建测试样本，不因准确率设置中途 Gate，也不暂停等待第二次报价批准。

但当前代码仍未实现这一方案：

- `experiments/C3_27B_ONE_SHOT_RUNBOOK.md` 仍有单独两-cell preflight 与后置费用批准；
- `estimate_c3_cost.py` 只接受该两-cell preflight 结果；
- C3 launcher/protocol 仍区分 preflight 与 core。[R4,R5]

需在租卡前一次修订协议、launcher、runbook、估算器，并在 5090 用同一正式调度流程演练。首个正式样本完成后自动进入后续样本。非有限输出、错误字节、OOM 或落盘失败只按常规执行错误处理。

27B 的模型配置、权重校验、tokenizer、依赖、attention/recurrent backend、原始数据、样本列表、resume 和总预算要提前确定。若供应商允许，先在不计 GPU 的持久卷准备权重和依赖。

旧 V6.1 的 59–63 GiB 峰值只能作为粗参考：当前 probe、reference backend、临时张量与旧实现不同。按最终代码计算 `权重 + 全量 KV + recurrent state + probe 临时张量 + workspace + 余量`；必要时在 5090 测试目标模型单层的真实形状/算子，不能将这个算子测试称为 27B 质量验证。

无独立 preflight 不意味着可以保证目标硬件全部行为已经本地测过。用首个正式任务检查不可避免的运行差异，是正常实验执行，不是重新引入研究审批。

## 8. 正式矩阵与更有价值的顺序

Codex 的原矩阵计数正确：

| 实验 | 样本/长度 | 方法与预算 | generation cells |
|---|---|---|---:|
| 合成中央点 | 12 Needle + 12 LongEval，精确 32K | 四个压缩臂 10% + Full | 120 |
| 合成两侧预算 | 同 24 条 | 四个压缩臂 5%/20%，复用 Full | 192 |
| 原生 QA 中央点 | 12 HotpotQA + 12 NarrativeQA，复用 C2 内容 | 四个压缩臂 10% + Full | 120 |
| 合计 | 48 个不同任务样本 | 无额外 preflight | 432 |

原生 QA 不是 exact-32K；须用 27B tokenizer 重新核对长度，不能将整个矩阵称为“全部 27B/32K”。这 24 个原生样本已看过小模型结果，合理标签是预先固定的跨模型迁移样本，而非从未接触的最终测试集。

推荐顺序为：**合成中央 10% → 原生中央 10% → 合成 5%/20%**。这样租卡被意外打断时，也已优先完成规模机制与真实任务两类结果，而不是只留下合成预算曲线。

建议保持 432-cell 框架，把四个压缩臂中的 Scattered 换成已在 5090 验证的最接近公开基线。Scattered 已有 0.8B/9B 强机制证据，不必在 27B 再同密度重复。最终替换需在 outcomes 前写进协议，不能边跑边选。

若坚持保留四个内部压缩臂并新增一个外部臂，矩阵变为 528 cells：合成 384 + 原生 144。增加 96 cells，约 22.2% 的生成次数，不等于必然只增加 22.2% 时间；公开基线自己的计算开销也必须预算。不要不改小时预算就默认添加。

## 9. 小时预算：认可作为额度草案，尚不认可为校准估计

Codex 给出的原始规划为：[R1]

| 付费阶段 | 点估计 h | 范围 h |
|---|---:|---:|
| 启动、加载、第一条正式样本的初始输出 | 0.35 | 0.25–0.50 |
| 合成中央 10% | 2.25 | 1.50–3.00 |
| 合成 5%/20% | 3.00 | 2.00–4.00 |
| 原生中央 10% | 2.25 | 1.50–3.00 |
| 汇总、备份、重试余量 | 0.75 | 0.50–1.00 |
| 总计 | 8.60 | 5.75–11.50 |

可以把 **12 A100-80GB GPU-hours** 作为拟议预留额度，14 h 作为供 PZ 审议的绝对费用上限；不能承诺 8.6 h 完成。单卡时 GPU-hours 与运行 wall-clock hours 相同；多卡并非如此。

需要的不是付费 preflight，而是一个本地可解释估算器：

`H = [模型加载 + Σ样本准备/冷probe + Σ(各臂prefill + query + 选址/压缩 + 输出token数×decode时间) + I/O与余量] / 3600`。

每项列出：5090 实测值、长度/模型形状放大依据、A100 硬件迁移假设、低/中/高情景。预算使用冻结后端，不按“更大卡一定更快”或单一参数比外推。

现估算器把 Narrative 的完整 per-arm 耗时乘 4 来处理最大输出 128 对 32，连与输出长度无关的 prefill 也被乘了 4。它是粗保守代理，不是分阶段测量；改成输入长度成本与实际/保守输出长度成本分开。

若供应商单卡总价为 P/小时，则 12P 是预留 GPU 费用、14P 是拟议上限，另列持久盘和带宽等费用。当前没有实时供应商报价，不填造货币数值。首次正式样本计入中央块，不能在启动项与中央块重复核算。

## 10. 相似论文应该借鉴什么

Codex 归纳的 H2O、SnapKV、Quest、DuoAttention、HeadKV、CAKE 提供了合理实验组织参考：成熟的 7–9B 模型做较广任务与消融，大参数模型作规模锚点；理论子问题、正式基线、预算曲线与真实任务共同构成论证，不是靠单个巨大模型取代其他证据。

该报告目前主要列举 2023–2025 的工作，不是完整“最新 AAAI/ICLR/NeurIPS”审计。尤其应补最接近的 ChunkKV 实验配置，而不是只增加更多不直接相关的会议名。对于只有 arXiv 链接却标具体录用会议的条目，例如表中的 SAGE-KV，正式引用前需核对实际会议身份。也不要将某篇论文的 16 个任务或 200 样本/任务变成所有论文的必需条件。

本轮网页搜索禁用；上述文献比较来自仓库中 Codex 的整理，我未独立打开原论文核验逐项配置或最新录用状态。

## 11. 给 Codex 的合并推进要求

在 9 月 9 日设计和结果收敛目标下，执行三条相互配合的工作线：

1. **证据线**：9B 原生 QA 主表、最接近公开基线、少量预定机制和效率测量。
2. **理论与方法线**：明确已有证明的范围，修正整体复杂度和 coverage-threshold 表述；不再发明 scorer 或追加大而空的定理。
3. **运行线**：取消独立 preflight，审计无用 recurrent probe、实际 prefill 次数、峰值和恢复粒度；给出有来源的小时预算与一个正式命令。

只有租卡、下载等真实资源开销由 PZ 一次确认。本地已授权范围内的实现和实验按合并包推进，不每个微小步骤回来申请。已有结果不好的 cell 保留，不据此临时换任务、改变主指标或重新打开设计空间。

完成的标志不是“所有指标显著”，而是“增量贡献有实际证据、理论对象明确、付费任务没有已知未修的实施缺口、预算和代码一致”。届时 A100 直接运行正式验证。

## 参考：本轮核查的仓库文件

- [R1] [openchat/codex/share/2026-09-05/hmo_pre_a100_readiness_and_budget_review.md](https://github.com/Metafora072/hmo/blob/bb23fbff2ee319ef05912a6f7547884765b5e55e/openchat/codex/share/2026-09-05/hmo_pre_a100_readiness_and_budget_review.md)
- [R2] [openchat/codex/share/2026-09-04/native_longbench_c2_results.md](https://github.com/Metafora072/hmo/blob/bb23fbff2ee319ef05912a6f7547884765b5e55e/openchat/codex/share/2026-09-04/native_longbench_c2_results.md)
- [R3] [docs/design/HMO_FINAL_METHOD_AND_THEORY_ZH.md](https://github.com/Metafora072/hmo/blob/bb23fbff2ee319ef05912a6f7547884765b5e55e/docs/design/HMO_FINAL_METHOD_AND_THEORY_ZH.md)
- [R4] [experiments/C3_27B_ONE_SHOT_RUNBOOK.md](https://github.com/Metafora072/hmo/blob/bb23fbff2ee319ef05912a6f7547884765b5e55e/experiments/C3_27B_ONE_SHOT_RUNBOOK.md)
- [R5] [experiments/phase2/estimate_c3_cost.py](https://github.com/Metafora072/hmo/blob/bb23fbff2ee319ef05912a6f7547884765b5e55e/experiments/phase2/estimate_c3_cost.py)
- [R6] [experiments/phase2/e3_v2/run_pareto.py](https://github.com/Metafora072/hmo/blob/bb23fbff2ee319ef05912a6f7547884765b5e55e/experiments/phase2/e3_v2/run_pareto.py)
- [R7] [experiments/phase2/e3_v2/query_accessibility.py](https://github.com/Metafora072/hmo/blob/bb23fbff2ee319ef05912a6f7547884765b5e55e/experiments/phase2/e3_v2/query_accessibility.py)
- [R8] [openchat/conversation/conversation_2026-09-05.md](https://github.com/Metafora072/hmo/blob/bb23fbff2ee319ef05912a6f7547884765b5e55e/openchat/conversation/conversation_2026-09-05.md)
- [R9] [openchat/conversation/conversation_2026-09-04.md](https://github.com/Metafora072/hmo/blob/bb23fbff2ee319ef05912a6f7547884765b5e55e/openchat/conversation/conversation_2026-09-04.md)
- [R10] [docs/design/HMO_METHOD_AND_NOVELTY_DOSSIER_ZH.md](https://github.com/Metafora072/hmo/blob/bb23fbff2ee319ef05912a6f7547884765b5e55e/docs/design/HMO_METHOD_AND_NOVELTY_DOSSIER_ZH.md)
