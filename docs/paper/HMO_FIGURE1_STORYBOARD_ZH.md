# HMO Figure 1 故事板

## 图的任务

Figure 1 不负责展示全部算法，也不宣称 HMO 首次使用 contiguous chunks。
它需要让读者在十秒内理解三个事实：Hybrid 模型已经有一条固定大小的全局
记忆通道；剩余 KV 应承担可直接寻址的局部高保真职责；相同 KV 字节下，
retention structure 会改变证据是否完整存活。

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

### Panel B: Equal bytes, different organization

使用同一条带有目标 evidence span 的长上下文，画上下两行严格等数量 token。

上行 `Scattered importance`：高分位置被离散保留，但目标短语中的连接词、数字
或实体边界缺失。下行 `HMO stratified local overlay`：不同 macro-segments
各有一个小窗口，目标 segment 内的窗口可滑动到 evidence span，并完整覆盖。

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

## Caption V1

> HMO organizes the residual Full-Attention cache of a Hybrid-Attention LLM
> as a locality-preserving overlay on an unchanged recurrent global state.
> Under the same query probe and measured resident KV bytes, stratified
> contiguous windows preserve complete local evidence that scattered token
> retention can fragment, yielding consistent gains at Qwen3.5-0.8B and 9B.

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
