# Layer-local HMO v1 方法契约

## 1. 要解决的问题

Qwen3.5 Hybrid 模型只有部分层保存随上下文增长的 Full-Attention KV，DeltaNet 层维护固定形状递归状态。现有 HMO 在所有 Full-Attention 层共享同一组 Sparse token 位置，但不同层对同一段上下文的查询注意力并不一致。共享位置因此可能把有限 KV 字节交给某一层并不需要的 token。

## 2. 固定部分

Layer-local HMO v1 沿用 legacy HMO 的全部全局决策：256-token 分段、首尾保护段、每段 `Recurrent-only/Sparse/Exact` 动作、每段保留 token 数、10% 中段 KV 预算及其整数 slack。DeltaNet 递归状态保持不变，所有 Full-Attention 层的 KV tensor 形状和总驻留字节保持一致。

## 3. 唯一改动

对每个被分配为 Sparse 的段 `s`、每个 Full-Attention 层 `l`，在该段内独立选择固定宽度连续窗口：

```text
W*_{l,s} = argmax_{W subset s, |W| = k_s, W continuous}
           sum_{i in W} a_{l,i}
```

其中 `a_{l,i}` 是真实 query suffix 在层 `l` 对 context token `i` 的平均注意力，`k_s` 完全继承 legacy HMO。位置在 KV heads 内共享，但不在 Full-Attention 层间共享。

## 4. 可证明边界

在固定段级动作和保留数、并以逐层 query-attention mass 为代理目标时，逐层最优窗口包含 legacy 共享窗口这一可行解，因此每层 Sparse 保留质量不下降，总代理质量也不下降。该结论不等价于 QA 指标必然提升，也不声称优于 ChunkKV；后两项必须由真实生成实验验证。

## 5. 确认实验

冻结协议为 `refine-logs/layer_local_confirmation_protocol.json`。它从原始 LongBench 归档排除既有 506 条记录及其输入身份，在 Qasper、MultiFieldQA-en、HotpotQA、2WikiMQA 各固定 20 条、四个长度层各 5 条。比较 legacy HMO、layer-local HMO、ChunkKV 和 Full KV；前三者要求 post-query 总字节及逐层字节完全相等。由于剩余样本系统性更短，该实验只证明 shorter-context transfer，不包装为原 506 条同分布独立复现。
