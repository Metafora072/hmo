# HMO Figure 1 故事板

## 图的任务

Figure 1 不负责展示全部算法，也不宣称 HMO 首次使用 contiguous chunks。
它需要让读者在十秒内理解四个事实：Hybrid 模型已经有一条固定大小的全局
记忆通道；剩余 KV 应承担可直接寻址的局部高保真职责；HMO 同时组织区域
coverage 与区域内 placement；最合适的组织会随预算和上下文长度变化。

## 横向三面板

### Panel A: Hybrid memory anatomy

画一条重复的 `3 x Gated DeltaNet + 1 x Full Attention` layer stack。DeltaNet
层侧边汇聚成固定大小的 `recurrent global state`；Full-Attention 层侧边连接
到随 token 数增长的 `residual KV cache`。只在后者上画压缩箭头。

主标签：

```text
Compressed global base        Addressable local overlay
fixed-size recurrent state    length-growing Full-Attention KV
```

禁止画成两个独立模型，也不要暗示 recurrent state 能按 token 精确读取。

### Panel B: Two-level residual-KV organization

使用同一条带有目标 evidence span 的长上下文，画上下两行严格等数量 token。

上行 `Global concentration / scattered importance`：预算集中在少数全局
高分单元，或把目标关系拆成离散 token。下行 `HMO stratified overlay`：
macro coverage 先分布到不同区域，目标区域内的 free-start window 再滑动到
evidence span，并完整覆盖。

角落加入一个很小的 `Global fixed chunks` 轮廓作为 related-work 提示，配文
`structured, but globally ranked / fixed boundaries`。它不是结果 bar，也不应
在没有实验前画成被 HMO 击败。

Panel B 的视觉重点是：

```text
same resident KV bytes
same query probe
different retention geometry
```

### Panel C: Cross-scale mechanism evidence

画两组简洁的 paired bars，不画复杂坐标系：

| Model | HMO contiguous | Equal-byte scattered | Delta |
|---|---:|---:|---:|
| Qwen3.5-0.8B | 70.83 | 56.25 | +14.58 pp |
| Qwen3.5-9B | 95.83 | 79.17 | +16.67 pp |

下方单独标注 `13.38% mean per-case Full-KV footprint`。Full KV 可用细横线
表示 0.8B 的 72.92 和 9B 的 95.83，但不要形成第三组粗 bar 抢走结构对照。

在 paired bars 下增加一条很窄的 regime strip：

```text
below coverage floor       long-context coverage regime       saturation
global concentration       stratified coverage                convergence
5%                         10% / 16K                           20%
```

## Caption V1

> HMO organizes the residual Full-Attention cache of a Hybrid-Attention LLM
> as a two-level stratified overlay on an unchanged recurrent global state.
> Macro-region coverage distributes scarce exact memory across long contexts,
> while query-guided free-start windows preserve complete local evidence.
> Exact-byte experiments expose a budget-length transition and consistent
> gains over scattered retention at Qwen3.5-0.8B and 9B.

## 视觉编码

- Recurrent state：中性深绿，表示持续但不可逐 token 寻址的压缩状态。
- Full KV：石墨灰底，保留位置用蓝色；不要让整图变成单一蓝紫色。
- Evidence span：高对比红色细边框，只用于同一目标 span。
- Scattered selection：灰蓝小方块；HMO window：连续蓝色带。
- Exact upgrade：可用实心边框作为次要图例，不在 hero panel 中占主角。

## 与后续图的分工

Figure 1 只讲问题、方法直觉和已经完成的跨规模证据。5/10/20% Pareto 属于
Figure 2；ChunkKV/ProtoKV 等差异属于 Related Work 表；完整 allocator、
slack extension 和 Exact upgrade 放 Algorithm 1。这样可以让 hero 图漂亮，
同时不靠省略 closest work 制造虚假新颖性。
