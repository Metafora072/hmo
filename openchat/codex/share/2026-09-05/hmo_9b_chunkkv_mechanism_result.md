# HMO 9B ChunkKV 机制补充实验

## 判定

- 针对预设命题“在受控 8K/16K 机制集上，HMO 相对 ChunkKV 存在增量”：
  `claim_supported = no`。
- 针对收紧命题“二者在严格等 resident bytes 的中央预算下行为等价，并以
  小比例 residual KV 保持 near-Full”：支持。

独立内部 result-to-claim reviewer 与 Codex 的本地判读一致。该 reviewer 是
Codex 调用的内部评估，不代表 GPT 或 Opus 的发言。

## 决定性证据

Qwen3.5-9B、冻结 24 条 8K/16K Needle 与 LongEval-Lines、中央 10% 预算：

| System | Contains | Exact | Token F1 | Mean resident KV | Full ratio |
|---|---:|---:|---:|---:|---:|
| HMO | 0.958333 | 0.916667 | 0.972222 | 51,757,056 | 13.3849% |
| ChunkKV | 0.958333 | 0.916667 | 0.972222 | 51,757,056 | 13.3849% |
| Full KV | 0.958333 | 0.875000 | 0.958333 | 397,164,544 | 100% |

HMO 对 ChunkKV 在三项指标上均为 `0W/24T/0L`；8K、16K 分组均值差都是
0。两者 24/24 严格等 resident bytes，生成 token 为 23/24 完全相同。唯一
不同样本只是 `335 grams of saffron.` 是否带末尾句号，分数不变。

本次 HMO 与 Full 的输出分别对历史冻结 scale-transfer 达成 24/24 token 级
复现，说明结论不是新旧运行漂移造成。正式运行耗时 418.40 秒，GPU1 已正常
释放。

## 对论文故事的影响

可以保留：

- Hybrid 模型 residual Full-Attention KV 的问题定义与 memory-role 分工；
- 可复现、严格等字节的真实 cache 干预与核算；
- HMO/ChunkKV 均在 Full residual-KV 的 13.3849% 下保持 near-Full；
- HMO 相对 scattered singleton retention 的跨规模 locality 机制证据；
- stratified coverage、free-start 和 Exact/Sparse action space 作为 HMO 的设计
  结构与分析对象。

不能再写：

- 连续 locality 本身是 HMO 相对 ChunkKV 的独有贡献；
- 当前 HMO 在公开 structured retention baseline 上有质量优势；
- synthetic mechanism suite 已证明 HMO-over-ChunkKV 的 working regime。

## 路线建议

不建议为挽救 superiority 继续盲跑 5%/20%，也不建议把当前命题直接放大到
A100/27B。5% 可能更偏好集中保留，20% 又容易饱和；此 sweep 只有作为诚实的
equivalence/negative curve 时才有价值。

下一步应先做零 GPU 的 policy geometry 与失败分析，明确 HMO 除“连续”之外
还能否形成可检验增量；同时整理 peak VRAM、TTFT、decode throughput 等真实
效率口径。若提出新 policy，现有 506 条与 24 条结果只能作为开发诊断，必须在
新冻结样本上确认。

完整报告：`../../../../experiments/results/CHUNKKV_MECHANISM_TRANSFER_9B_20260905.md`。
原始结果：`/mnt/nvme0/hmo/runs/chunkkv_mechanism_9b_f465657/`。
