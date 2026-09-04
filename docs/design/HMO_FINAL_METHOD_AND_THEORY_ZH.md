# HMO 最终方法与理论合同

## 1. 文档角色

本文冻结 HMO 在大卡实验前唯一的 paper-facing 方法定义与理论解释。它不是
开发历史，也不继续搜索新的 recurrent score。后续 5090 与 A100/H100 runner
只能改变模型、任务、样本和预算配置，不能在结果出现后改变本文件中的动作
空间、排序语义或字节口径。

工作标题：

> HMO: Stratified KV Overlays for Hybrid-Attention Language Models

一句话方法：

> HMO 保持 Hybrid LLM 的 recurrent global state 不变，将 residual
> Full-Attention KV 组织成两级 overlay：跨 macro-regions 分配连续局部覆盖，
> 再在每个区域内按 query demand 放置 free-start window，并用剩余字节执行
> 窗口扩展或可选 Exact fidelity。

## 2. 目标与不变量

### 2.1 要解释的现象

Hybrid 模型已经用 recurrent/linear-attention layers 提供固定容量的全局压缩
记忆，但周期性的 Full-Attention layers 仍保存随上下文线性增长的 KV。有限
residual KV 同时面临两种失败：

1. **fragmentation**：scattered Top-token 保留高 singleton importance，却
   破坏由多个连续 token 构成的关系证据；
2. **concentration**：global Top-chunk 保留完整局部结构，却可能把预算集中在
   少数区域，使长上下文中的 regional coverage 不均。

### 2.2 统一不变量

理论的统一对象是在 resident-byte 预算内，证据被 residual KV 完整、可寻址地
保留下来的期望效用：

$$
\max_{\mathcal A}\quad
\mathcal U(\mathcal A)
\qquad
\text{s.t.}\quad
C_{KV}(\mathcal A)\le B.
$$

其中 $\mathcal A$ 是各 macro-region 的 KV action 集合，$C_{KV}$ 是真实
Full-Attention resident KV bytes。span survival、query mass 和 region utility
都是该不变量的可分析代理，不等同于最终生成准确率。

## 3. 系统模型与记号

设：

- $\mathcal L_{rec}$：recurrent/linear-attention layers；
- $\mathcal L_{full}$：保留显式 KV 的 Full-Attention layers；
- $T$：可压缩的 middle-context token 数；
- $S_i$：第 $i$ 个 macro-segment，标准长度为 $L$；
- $a_t\ge 0$：冻结 query probe 对 context token $t$ 的 attention mass；
- $w$：每个 coverage action 的 base local width；
- $B$：middle-context resident-KV byte cap；
- $c$：获得 coverage 的 segment 数；
- $m$：从 local window 升级为 Exact 的 segment 数；
- $s$：用于扩展 local windows 的 slack token 数。

HMO 只改变 $\mathcal L_{full}$ 的 prompt KV：

$$
M_{hybrid}=M_{rec}+M_{KV}^{full},
\qquad
\widetilde M_{hybrid}=M_{rec}+\widetilde M_{KV}^{full}.
$$

$M_{rec}$ 在所有对照中保持相同。本文报告 residual Full-Attention KV
footprint，而不是总模型显存的同倍下降。

## 4. 最终动作空间

首尾 protected segments 始终保留 Exact。每个 eligible middle segment 取：

$$
A_i\in
\{\text{recurrent-only},\ \text{local-window}(k_i),\ \text{Exact}\}.
$$

- `recurrent-only`：该区域不保留显式 Full-Attention KV，但 recurrent state
  没有被删除或重置；
- `local-window(k_i)`：保留一个长度 $k_i\ge w$ 的连续窗口；
- `Exact`：保留该 segment 的全部 Full-Attention KV。

mandatory core 的含义是：一旦某个 segment 获得 coverage，其保留位置必须
连续；它不表示所有预算下每个 segment 都必须有窗口。Exact 是可选 fidelity
action，允许 $m=0$。

## 5. Query Probe 与确定性合同

所有 query-ranked methods 必须消费同一份持久化 token-score artifact：

1. 对固定模型 revision、完整 token IDs、query boundary 和 Full-Attention
   layer set 只计算一次 query probe；
2. token scores 以 FP32 `.npy` 保存，并记录 SHA256；
3. segment demand 从该 FP32 vector 以固定 CPU 求和语义导出；
4. 所有排序使用总序
   $(-\text{stored score},\ \text{position or chunk index})$；
5. 每个 result row 记录 probe ID、score hash、cache hit 和 aggregation version；
6. identity、长度、segment bounds 或哈希不匹配时 fail closed。

默认不对近似相等分数做 tolerance bucketing，因为 tolerance 会成为新的方法
超参数。跨运行与 resume 通过复用同一份 artifact 获得完全相同的 retained
positions；fresh recomputation 只作为数值敏感性诊断。

## 6. 最终算法

```text
Input: unchanged recurrent state, Full-layer prompt KV, persisted query scores a,
       macro-segments S, base width w, middle byte cap B

1  Keep protected prefix/suffix segments as Exact; charge them separately.
2  For every eligible segment i, compute q_i = sum_{t in S_i} a_t.
3  If B can buy one w-token window for every eligible segment:
       establish one coverage action in context order.
   Else:
       visit segments by descending q_i / coverage_byte_cost, tie by segment id.
4  Assign local-window(w) while the middle byte cap permits.
5  If fidelity is enabled, visit covered segments by descending
       q_i / exact_upgrade_byte_cost, tie by segment id,
   and upgrade affordable segments to Exact.
6  Spend remaining token-granularity slack by extending non-Exact windows in
   descending q_i order, tie by segment id.
7  For each local-window(k_i), choose the free-start window with maximum stored
   query mass; break ties by the earliest start.
8  Apply the same retained context positions to every Full-Attention KV layer.
9  Process the query at original logical positions and assert measured
   post-query resident KV bytes.
```

当前中央配置为 $L=256,w=16$，但它们是实验参数，不是 HMO 的定义本身。

## 7. 命题一：完整连续证据存活

### 假设

在一个 segment 内固定保留 $k$ 个不同位置。证据是完全位于该 segment 内的
单一连续 $\ell$-token span，其中 $2\le\ell\le k$；所有合法起点按 uniform
prior 计数。

### 命题

若保留集合 $R$ 的 maximal consecutive runs 长度为
$r_1,\ldots,r_J$，则完整存活的 $\ell$-span 起点数为：

$$
N_\ell(R)=\sum_{j=1}^{J}\max(r_j-\ell+1,0)
\le k-\ell+1.
$$

单个长度为 $k$ 的连续 run 达到上界。

### 证明骨架

每个长度 $r_j$ 的 run 内包含
$\max(r_j-\ell+1,0)$ 个完整 $\ell$-span。合并任意两个非空 runs 不减少
可容纳的完整 spans；反复合并后得到一个总长度
$\sum_j r_j=k$ 的 run，因此上界为 $k-\ell+1$。

### 解释与边界

这是固定 cardinality 下的组合命题，解释 contiguous 对 relational evidence
的结构优势。它不声称 evidence 必为 uniform，不声称跨 segment 证据被完整
覆盖，也不直接保证生成准确率。

## 8. 推论一：Locality class 内的 query-optimal placement

在 segment $S_i$ 的全部长度 $k_i$ 连续窗口中定义：

$$
W_i^*=\arg\max_{
W\subseteq S_i,\ W\text{ contiguous},\ |W|=k_i
}
\sum_{t\in W}a_t.
$$

该定义直接给出：$W_i^*$ 在固定 locality class 内保留最大的 query mass。
它不与 unconstrained scattered Top-token 的 singleton-mass optimum 混淆。

## 9. 恒等式：保留量与 coverage floor

忽略 partial tail 的异长影响时，middle context 的保留 token 数满足：

$$
N_{keep}=cw+m(L-w)+s,
\qquad
\rho_{middle}=\frac{cw+m(L-w)+s}{T}.
$$

这是给定 action counts 后的恒等式。若所有约 $T/L$ 个 regions 都获得 base
window，且 Exact/slack 暂时忽略，则：

$$
B_{cover}\approx\frac{w}{L}.
$$

对 $w=16,L=256$，coverage floor 约为 $6.25\%$ 的 middle Full KV。它是
规划近似，不是整体 resident ratio；protected anchors、query KV、partial
segments、Exact 与 slack 都会改变实际 footprint。

固定 $L,w$ 与预算比例时，retained KV 仍随 $T$ 线性增长，即 $O(T)$。HMO
降低的是 residual-KV 线性系数，不改变渐近阶数。

## 10. 命题二：Coverage 与 concentration 的工作区间

### 可分析切片

令每个 region 接收整数个等字节 action units $q_i\ge0$，并定义：

$$
U(\mathbf q)=\sum_i p_i u_i(q_i),
\qquad
\sum_i q_i\le Q.
$$

假设 $p_i\ge0$，且每个 $u_i$ 单调并具有离散凹性：

$$
\Delta u_i(r)=u_i(r+1)-u_i(r),
\qquad
\Delta u_i(r+1)\le\Delta u_i(r).
$$

### 命题

在上述等成本、可分、离散凹切片中，按
$p_i\Delta u_i(q_i)$ 的当前最大边际收益逐单位分配是最优的。进一步，若：

$$
\min_i p_i\Delta u_i(0)
\ge
\max_j p_j\Delta u_j(1),
$$

则在任意 region 获得第二个 action unit 之前，所有可负担 regions 都先获得
第一个 unit，即 coverage-first 是该阶段的最优结构。

### 证明骨架

每个 region 产生一条非增 marginal-gain 序列。任何可行分配都等价于从这些
序列中选择满足 prefix constraint 的 $Q$ 个 marginals。选取当前最大合法
marginal 的 greedy 与全局 top-$Q$ marginals 等价；若某解包含更小 marginal
却遗漏更大合法 marginal，可交换并不降低目标。附加不等式保证所有 first
marginals 排在所有 second marginals 之前。

### Regime 解释

- **coverage floor 以下**：$Q$ 不足以给每个 region 一个 unit，或 $p_i$ 高度
  集中，global concentration 可能更有效；
- **coverage regime**：first-coverage density 高于 repeated-upgrade density
  时，regional coverage 优先；
- **saturation regime**：预算增加后，各 structured methods 的关键 evidence
  已存活，质量趋于收敛。

该命题不证明 HMO 的 attention score 等于真实 $p_i\Delta u_i$，也不覆盖
Exact upgrade 的异成本 knapsack。它为实验观察提供统一解释模型，而不是
downstream accuracy guarantee。

## 11. 复杂度与系统口径

设 segment 数 $n\approx T/L$：

- query-score 聚合：$O(T)$；
- segment demand：$O(T)$；
- segment/action ordering：$O(n\log n)$；
- 全部 free-start sliding windows：$O(T)$；
- retained-position materialization：$O(N_{keep})$。

当前实现先产生 Full-KV prompt cache，再执行压缩，因此实验直接证明的是
post-query resident KV reduction，而不是同倍 peak-VRAM reduction。最终系统
表必须分别报告 resident KV bytes、controller/probe overhead、TTFT 和 peak
allocated/reserved memory。

## 12. 理论到实验的对应

| 理论对象 | 实验对照 | 当前证据 |
|---|---|---|
| span survival | HMO contiguous vs Scattered | 0.8B 5%/10% 为 +18.75/+14.58 pp；9B 10% 为 +16.67 pp |
| free-start locality optimum | HMO vs Stratified Fixed | 16K/10% 为 18/24 vs 17/24，分离集中在 LongEval |
| coverage floor 与 regime | Global Fixed/HMO 的 5%/10%/20% Pareto | 5% concentration 强，10%/16K HMO 反超，20% 饱和 |
| residual-KV coefficient | compressed vs Full measured bytes | 中央 synthetic suite 为 13.38% mean per-case footprint |
| real-task feasibility | HotpotQA-32K-Aug paired pilot | 11.556% footprint 保留相同 2/4 solvable set |

## 13. 最终贡献边界

正文使用三项贡献：

1. **Hybrid residual-memory formulation**：recurrent global substrate 与
   residual addressable KV overlay 的职责分工；
2. **Stratified KV overlay**：macro-region coverage、query-guided free-start
   windows、可选 fidelity 和真实字节预算组成的 training-free allocator；
3. **Mechanism and regime evidence**：跨 0.8B/9B 的 equal-byte geometry
   evidence、budget-by-length transition 与 near-Full-KV quality。

不声称首次使用 chunk/locality，不声称 recurrent accessibility 已验证，不
声称对 fixed chunk 普遍占优，不声称次线性 KV，也不把 residual-KV ratio
写成总 GPU memory 的等比例下降。
