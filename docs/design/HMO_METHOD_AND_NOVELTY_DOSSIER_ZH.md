# HMO 方法与新颖性边界

## 结论先行

HMO 当前可以继续，但论文不能再把“连续 chunk 比离散 token 更能保护语义”
作为首创观察。ChunkKV、SentenceKV、ProtoKV 和 Kara 已经从固定 chunk、句子、
语义 cluster 和动态 chunk 等角度覆盖了这一空间。

更准确也更有辨识度的定位是：

> HMO 面向 Hybrid-Attention LLM，把固定容量 recurrent state 视为全局压缩
> 基础，把仍然增长的 Full-Attention KV 组织成 stratified local overlay：先
> 在不同上下文区域建立局部可读性，再把剩余预算用于高需求区域的 fidelity。

其中真正需要共同成立的不是某一个新打分公式，而是四件事：

1. 研究对象是 Hybrid 模型中 residual Full-Attention KV，而非假设全部历史
   只存在于 KV cache。
2. 预算语义是 coverage-first：优先让更多区域拥有一个完整、可直接寻址的
   局部窗口，再做 Exact upgrade。
3. 窗口不是固定 chunk 边界；它在每个 macro-segment 内按 query attention
   自由滑动并取最大质量区间。
4. 结论由真实 cache intervention、逐样本 resident bytes 和等字节结构对照
   支撑，而不是只比较名义 token ratio。

这是一条可包装、可解释、也与现有代码一致的主线；但它仍需要在下一轮实验
加入 structured-chunk baseline，才能把“优于 scattered”升级成“优于已有
structured retention”。

## Closest-work 审计

审计时间为 2026-09-04。表中只写从论文或官方实现能够核对的性质。

| 工作 | 单位与选择方式 | Query 条件 | Coverage 语义 | Hybrid memory 角色 | 与 HMO 的关键差异 |
|---|---|---|---|---|---|
| H2O, NeurIPS 2023 | token；累计 attention heavy hitters + recent | 历史 query 间接决定 | 无区域覆盖 | 无 | 在线 eviction，核心是 token heavy hitter |
| SnapKV, NeurIPS 2024 | per-head token positions，并用 pooling 扩展邻域；保留 observation window | prompt 尾部 observation queries | 无跨区域保证 | 无 | 已具有邻域聚合，不能笼统称为纯 scattered baseline |
| Quest, ICML 2024 | KV page；用 page 的 min/max key 估算当前 query 重要性 | 强，逐 decode query | 只加载 Top-K pages | 无；完整 KV 仍被存储 | 优化 decode 访存而非 resident KV eviction |
| PyramidKV, ICLR 2025 submission | token；跨层非均匀预算与 attention selection | observation attention | 无 | 无 | 贡献在 layer budget funneling |
| ChunkKV, 2025 preprint | 固定边界 chunk；按 observation attention 的 chunk sum 做全局 Top-K | 是 | Top-K chunk，无分层区域覆盖 | 无 | 与 locality 动机最接近；HMO 是 macro-segment 内自由滑窗并 coverage-first |
| SentenceKV, COLM 2025 | 句子；GPU 语义向量 + CPU KV，decode 时检索句子 | 是，多 query 聚合 | sentence retrieval | 无 | 语义边界、offload 与动态加载，不是 resident overlay |
| ProtoKV, ICLR 2026 | key-space semantic prototypes/clusters | pre-query 组织 | semantic clusters | 无 | 更强的语义单元建模；HMO 不做聚类，强调 Hybrid 分工和区域覆盖 |
| Kara, 2026 preprint | 最近生成窗口中的 token candidates 扩为可变 chunk | 双向窗口 attention | 可在任意位置形成 chunk | 无 | 服务 reasoning decode 与周期压缩；HMO 压缩长 prompt 的 residual KV |
| Zoology, 2024 | 训练期 hybrid architecture；学习/规则选择 attention positions | input dependent | 非 KV eviction | 解释 attention 补足 recurrent recall | 为双记忆动机提供机制背景，不是部署期 cache policy |
| Hypic, 2026 preprint | Hybrid PIC 的 recurrent transition + boundary seam windows | 非本问题的 query retention | segment cache composition | 显式处理两种状态 | 目标是跨请求 position-independent cache reuse，不是 KV compression |

### 最高风险

ChunkKV 已明确提出“离散 token 会破坏 token dependency，连续 chunk 保留完整
语义”，且也使用 observation-query attention 给 chunk 打分。ProtoKV 与
SentenceKV 进一步扩大了 semantic-level compression 的已有覆盖。因此以下
说法不再安全：

- 首次发现离散 KV 会破坏局部语义；
- 首次用连续 chunk 做 KV compression；
- query-guided locality 本身就是完整新颖性；
- span-survival 命题本身足以构成理论贡献。

### 可守住的差异

HMO 不应与 ChunkKV 比“谁更 chunk”，而应比 memory organization：ChunkKV
从全局候选中保留若干固定边界 chunk；HMO 先把 Hybrid residual KV 划成
macro-segments，在可负担时给每个 eligible segment 一个自由起点的连续
micro-window，再把剩余预算用于窗口扩展或 Exact upgrade。这个 stratified
coverage 由 recurrent global base 提供架构动机。

当前 0.8B/9B 结果只直接证明 HMO 的连续实现优于同 allocator、同字节的
scattered 实现。它尚未证明 HMO 优于 ChunkKV/ProtoKV。因此下一轮最有价值的
baseline 不是再发明 scorer，而是加入同 probe、同 resident bytes 的 global
fixed-chunk Top-K；若工程允许，再对接官方 ChunkKV。

## 冻结方法定义

### System model

设 Hybrid 模型的 recurrent/linear-attention layer 集合为
`L_rec`，Full-Attention layer 集合为 `L_full`。HMO 始终保留
`L_rec` 的 recurrent state，只干预 `L_full` 的 prompt KV：

$$
M_{hybrid}=M_{rec}+M_{KV}^{full},\qquad
\widetilde M_{hybrid}=M_{rec}+\widetilde M_{KV}^{full}.
$$

因此论文报告的是 residual Full-Attention KV footprint，不能把它写成总模型
显存的同倍缩减。

### Budgeted action hierarchy

上下文按长度 `L` 切成 macro-segments。首尾 protected segments 保留 Exact，
中间 eligible segments 可取三种动作：

$$
A_i\in\{\text{recurrent-only},\text{local-window}(k_i),\text{Exact}\}.
$$

`recurrent-only` 不是删除 recurrent memory，而是该 segment 不再保留显式
Full-Attention KV。实验中 base width `w=16`、`L=256`。

“coverage 是 mandatory core”的精确定义是：一旦对某个 segment 执行
coverage，其保留位置必须构成连续窗口；它不表示任意预算下所有 segment
都有窗口。当 middle cap 小于 `w/L=6.25%` 时，全 segment coverage 在算术上
不可行，allocator 按 query demand 优先覆盖可负担的 segments。

### Query-guided free-start window

令 `a_t >= 0` 为 query suffix 对 token `t` 的 attention mass。对于 segment
`S_i` 和实际分配宽度 `k_i`：

$$
W_i^*=\arg\max_{W\subseteq S_i,\,W\text{ contiguous},\,|W|=k_i}
\sum_{t\in W}a_t.
$$

tie 时选择最早起点，保证确定性。实现先分配 base width，再将不足以完成
Exact upgrade 的逐 token slack 按 segment demand 用于扩展 `k_i`，随后重新
求对应宽度的最大质量窗口。因此“width 16”应写成 base width，而不是每个
Sparse segment 最终都恰好保留 16 token。

### Frozen allocator pseudocode

```text
Input: Full-layer KV, unchanged recurrent state, query probe a,
       segments S, base width w, middle byte cap B

1  Keep protected prefix/suffix segments as Exact; charge separately.
2  For each eligible segment i, compute demand q_i = sum_{t in S_i} a_t.
3  If B covers one w-token window in every segment:
       coverage_order = context order
   else:
       coverage_order = descending rank(q_i) / byte_cost(window_i)
4  In coverage_order, assign local-window(w) whenever it fits in B.
5  If Exact fidelity is enabled, visit covered segments by descending
   query-demand-per-upgrade-byte and upgrade those that fit.
6  Spend remaining per-token slack by extending non-Exact local windows
   in descending query demand.
7  For each local-window(k_i), choose the free-start max-mass window W_i*.
8  Apply the same retained context positions to every Full-Attention KV layer;
   leave all recurrent states unchanged; assert measured resident bytes.
```

当前 frozen main variant 设置 `use_accessibility=false`。任何 DeltaNet
accessibility/saturation 信号都不是主方法输入。

## 理论解释与边界

### Proposition 1: local span survival

在一个 segment 内固定保留 `k` 个位置。保留集合由长度
`r_1,...,r_m` 的 maximal consecutive runs 构成。对长度
`2 <= ell <= k` 的连续证据 span，能够完整存活的起点数量为：

$$
N_\ell(R)=\sum_j\max(r_j-\ell+1,0)\le k-\ell+1.
$$

单个 `k` 长度连续窗口达到上界。成立条件必须同时写出：证据是单一连续
span、span 完全位于同一 segment、比较固定相同 `k`、起点按 uniform prior
计数。该命题解释结构先验，不保证 end-task accuracy，也不是 locality 的首创
理论。

### Corollary 1: query placement within the locality class

滑动和求最大值使 `W_i*` 在全部长度 `k_i` 的连续窗口中保留最多 query
attention mass。它不声称超过 unconstrained global Top-token 的 singleton
mass。

### Retention coefficient

若 `c` 个 eligible segments 获得 base window、其中 `m` 个升级为 Exact，
另有 `s` 个 slack token，忽略不同尾段长度时：

$$
N_{keep,middle}=cw+m(L-w)+s,
\qquad
\rho_{middle}=\frac{cw+m(L-w)+s}{T_{middle}}.
$$

当所有约 `T_middle/L` 个 segment 均被覆盖且 `s` 很小时，才可近似写成
`w/L + m(L-w)/T_middle`。固定参数下仍为 `O(T)`，贡献是降低线性系数。

## 论文贡献边界

建议正文贡献写成以下三项：

1. **Hybrid residual-memory formulation**：把 recurrent state 与 residual
   Full-Attention KV 的能力和成本分开，提出 global compressed base + local
   addressable overlay 的部署期组织原则。
2. **Stratified locality-preserving overlay**：coverage-first 的区域预算语义、
   segment 内 query-guided free-start window，以及可选 fidelity action 组成
   一个确定、training-free 的 cache policy。
3. **Causal and byte-grounded evidence**：真实 cache intervention 下，用共享
   allocator 的 equal-byte contiguous/scattered 对照隔离结构效应，并在
   Qwen3.5-0.8B 与 9B 得到一致方向。

Exact upgrade 是框架动作，不单列成已验证的核心贡献。span proposition 是
解释性支撑，不包装成普适 accuracy guarantee。

## 对下一阶段的直接影响

Package B 的五个已有 arms 可以保留，但应增加一个 highest-priority
structured baseline：

```text
Global Fixed-Chunk Top-K
  score: same query-attention probe
  unit: non-overlapping 16-token chunks
  allocator: global Top-K under exact target resident bytes
  anchors/query KV: identical to HMO
```

这能回答 HMO 的增益来自“任何 chunk 都行”，还是来自 stratified coverage +
free-start windows。Package C 的真实任务同样至少保留 HMO、Global Chunk、
Raw+Slack 和 Full 四个 arms。以上是增强论文说服力的优先级，不是严格停止
Gate。

## 核对来源

- [H2O, NeurIPS 2023](https://arxiv.org/abs/2306.14048)
- [SnapKV, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/28ab418242603e0f7323e54185d19bde-Abstract-Conference.html)
- [Quest, ICML 2024](https://arxiv.org/abs/2406.10774)
- [PyramidKV](https://arxiv.org/abs/2406.02069)
- [ChunkKV](https://arxiv.org/abs/2502.00299)
- [SentenceKV, COLM 2025](https://openreview.net/pdf?id=HyPeYU9JR6)
- [ProtoKV, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/e73ad1f690542144ce354637bb913c35-Abstract-Conference.html)
- [Kara](https://arxiv.org/abs/2607.01237)
- [Zoology](https://arxiv.org/abs/2312.04927)
- [Hypic](https://arxiv.org/abs/2607.01299)
