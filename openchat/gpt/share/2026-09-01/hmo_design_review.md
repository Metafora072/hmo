可以。现在你已经具备读 HMO 的前置知识了，下面我直接把这个工作按“论文视角”讲清楚。

以下概况以 **当前 GitHub `main` 分支的 V6.1 stable 主线和已经提交到仓库的正式 E1 结果**为准，不沿用早期 V1～V6 的历史方案。最后一部分我还专门检查了 ICLR 2024–2026 的相关录用工作，以及 2026 年刚出现的几篇非常接近 HMO 问题空间的工作。

---

# 1. 问题背景：HMO 到底在解决什么问题

你刚才已经理解 Hybrid Attention：

```text
Qwen3.5 这类 Hybrid LLM

DeltaNet
DeltaNet
DeltaNet
Full Attention
DeltaNet
DeltaNet
DeltaNet
Full Attention
...
```

因此模型内部实际上同时存在两套性质不同的历史记忆。

Full Attention 层保存：

$$
\{K_1,V_1,\cdots,K_T,V_T\}
$$

它是一种**显式、逐 token、可以精确寻址的记忆**，但其空间随上下文长度 \(T\) 线性增长。

DeltaNet 层保存：

$$
S_t
$$

它是一种**固定容量、隐式、压缩的 recurrent memory**。它几乎不随 \(T\) 增长，但历史信息经过不断的 decay、write 和 delta correction 后，可能发生干扰、覆盖和遗忘。

所以 Hybrid LLM 的历史并不是传统 Transformer 中的：

$$
\boxed{\text{Memory}=\text{KV Cache}}
$$

而是：

$$
\boxed{
\text{Memory}
=
\text{Explicit KV Memory}
+
\text{Implicit Recurrent Memory}
}
$$

这就是 HMO 整个工作的出发点。

---

## 1.1 为什么 Hybrid 模型仍然有 KV Cache 问题

假设一个模型 3/4 是 DeltaNet，1/4 是 Full Attention。

相比纯 Transformer，它已经天然减少了约 75% 需要保存 KV 的层。

但是剩下 1/4 Full Attention 层的：

$$
KV_1,\ldots,KV_T
$$

仍然随 \(T\) 线性增长。

32K、64K、128K 上下文继续增大以后，这部分 KV 依然会成为重要的推理显存和带宽开销。

于是自然会想到使用：

* H2O；
* SnapKV；
* Quest；
* CAKE；
* SAGE-KV；

这一类 KV Cache compression。

但这里就产生了 HMO 认为的新问题。

---

## 1.2 传统 KV Compression 只看了一半的记忆

假设上下文中有一个 segment：

```text
Segment 17

"The security code is 573914."
```

它同时经历了：

```text
                 Segment 17
                     │
          ┌──────────┴──────────┐
          ↓                     ↓
     DeltaNet layers       Full Attention
          │                     │
          ↓                     ↓
      写入 state S          生成 explicit KV
```

现在你准备压缩 Full Attention 的 KV。

传统 KV compression 会问：

> Segment 17 的 KV 重要吗？

比如根据：

$$
\text{attention score}
$$

或者：

$$
\|K\|,\|V\|
$$

判断。

但 HMO 认为还应该问另一个问题：

> **Segment 17 的信息有没有被 DeltaNet recurrent state 可靠保存下来？**

如果 DeltaNet 已经很好地记住了这一段，那么 Full Attention KV 可能可以更大胆地压缩。

反过来，如果：

```text
DeltaNet state
      ↓
强烈覆盖 / decay / collision
      ↓
Segment 17 的信息很可能已经不可靠
```

这时候再把 Full Attention 中 Segment 17 的 KV 删除，等于：

```text
隐式记忆：已经不可靠
+
显式记忆：又被删掉
=
信息真正丢失
```

因此 HMO 针对的问题可以浓缩成：

$$
\boxed{\text{Hybrid Memory Mismatch}}
$$

也就是：

> **Hybrid Attention 模型拥有 explicit KV memory 和 implicit recurrent memory 两条记忆通道，但现有推理时压缩策略主要只管理前者，没有利用后者的可靠性状态来决定 KV 资源应该如何分配。**

这才是 HMO 的核心 research question。

---

# 2. 现有研究的问题：为什么值得做 HMO

过去几年的 KV Cache 工作其实已经非常多。

StreamingLLM 发现 Attention Sink，只保留初始 sink token 和 recent tokens，就可以支持稳定的流式推理；该工作被 ICLR 2024 接收。([OpenReview][1])

DuoAttention 进一步发现，不同 Attention heads 的长距离检索作用差异很大，于是把它们分成 Retrieval Heads 和 Streaming Heads，只给少数 Retrieval Heads 保留完整 KV，被 ICLR 2025 接收。([OpenReview][2])

CAKE 则发现不同层不应该获得相同 KV budget，通过 layer preference 和 temporal dynamics 做全局预算分配，在仅保留约 3.2% KV 的情况下维持性能，并报告了 128K 下超过 10× 的 decoding latency speedup，同样被 ICLR 2025 接收。([OpenReview][3])

这些工作的共同演进路线其实是：

```text
最开始：
哪些 token 可以删？

↓

进一步：
不同 token 重要性不同

↓

进一步：
不同 head 的需求不同

↓

进一步：
不同 layer 的需求不同

↓

进一步：
当前 Query 不同，KV 需求也不同
```

但是这些工作观察的主体仍然基本是：

$$
\boxed{\text{Attention / KV channel}}
$$

HMO 想增加另外一个维度：

$$
\boxed{\text{Recurrent-memory reliability}}
$$

从这个角度看，HMO 的问题是成立的，而且确实对应 Hybrid LLM 普及以后才明显出现的新场景。

---

## 2.1 HMO 最有价值的地方并不是“又发明一种 KV pruning score”

如果 HMO 最后只是：

$$
score_i
=
0.4x_i+0.3y_i+0.3z_i
$$

然后：

```text
score 高 → Keep
score 低 → Drop
```

它的论文价值其实不会很强。

HMO 真正有潜力的 scientific insight 应该是：

> **在 Hybrid LLM 中，是否需要显式保存某段 KV，取决于该信息在 recurrent memory 中是否可靠。**

也就是说：

$$
\boxed{
\text{explicit KV demand}
\leftrightarrow
\text{implicit memory reliability}
}
$$

如果这个关系能够被实验明确证明，它比某一个具体 pruning heuristic 更重要。

这也是后面我为什么会说 **E3 是 HMO 目前最关键的实验**。

---

## 2.2 但 2026 年的文献环境已经让这个 novelty 变得更紧张

这里必须特别提醒你。

2026 年 3 月出现了 **Hybrid Associative Memories（HAM）**。它的核心思想非常接近 HMO 的高层 intuition：

> RNN 负责压缩整个历史，KV Cache 只显式存储 RNN 难以预测的信息。

HAM 明确把 RNN memory 和 KV memory 看成互补记忆，并根据 RNN prediction error 决定哪些信息值得进入 KV Cache。([arXiv][4])

这意味着：

> **“RNN 记得住的不用 KV，RNN 记不住的需要 KV”这个最高层思想，HMO 已经不能宣称完全首次提出。**

但 HMO 仍然有一个非常重要的区别。

HAM 是在**模型结构和训练阶段**设计 Hybrid Associative Memory layer。

HMO 当前目标则是：

$$
\boxed{\text{training-free inference-time controller}}
$$

即：

> 对已经训练好的 Qwen3.5 / Kimi Linear 这类 Hybrid LLM，不修改模型参数，根据它们运行时 recurrent state 的状态动态管理现有 Full-Attention KV。

这个定位必须强化。

而且就在 **2026 年 8 月 31 日**，也就是非常近的时间，又出现了 DASC。DASC 分析 Gated DeltaNet/KDA 的 decay 行为，通过 retention horizon 压缩 recurrent-state checkpoints，并提供 suffix refresh；其目标主要是 prefix reuse / serving，而不是 HMO 当前的 active long-context KV allocation，但说明 **“利用 recurrent decay 信息做 memory management”已经成为一个活跃方向**。([arXiv][5])

另外 HYPIC 也已经开始专门研究 Hybrid Attention 的 caching primitive 与 recurrent state composition。([arXiv][6])

所以 HMO 现在真正需要守住的创新点应该收窄成：

$$
\boxed{
\text{Inference-time recurrent-state reliability}
\rightarrow
\text{explicit KV allocation}
}
$$

而不能宽泛写成：

> “首次联合管理 RNN memory 和 KV memory。”

---

# 3. HMO 当前具体是怎么设计的

现在进入真正的 HMO。

当前 V6.1 大致可以分成：

```text
长 Context
    ↓
Segment
    ↓
检测 DeltaNet 状态风险 σ
    +
检测 Attention 依赖 α
    ↓
φ = σ × α
    ↓
固定 KV budget 下分配
    ↓
KV / Refresh / RTS / Drop
```

---

## 3.1 第一步：按 segment 管理，而不是逐 token 管理

默认：

$$
\text{segment length}=512
$$

例如 32K context：

$$
32768/512=64
$$

所以整个 Prompt 变成：

```text
C0 C1 C2 C3 ... C63
```

HMO 的基本决策对象是：

$$
C_j
$$

而不是每一个 token。

这样一方面降低决策开销，另一方面 HMO 想表达的是：

> “这一段上下文当前处于什么记忆状态？”

---

# 3.2 第二步：检测 DeltaNet 的 recurrent-state risk —— \(\sigma\)

这是 HMO 中目前最有研究味道的一部分。

Gated DeltaNet：

$$
S_t
=
e^{g_t}S_{t-1}
+
\beta_t k_t
(v_t-S_{t-1}^{T}k_t)
$$

HMO 从这里抽取三个信号。

### \(\rho\)：写入压力

大体为：

$$
\rho
=
\frac{\|\beta u\|}
{\tau+\epsilon},
\qquad
\tau=-g
$$

直觉：

```text
新的东西写得很猛
        +
旧状态保持能力较弱
        ↓
recurrent state 更新压力大
```

---

### \(c\)：写入方向 collision

比较当前 write direction 和最近若干 write directions：

$$
c_t
=
\max_j \cos(u_t,u_{t-j})
$$

如果连续很多信息往相似方向写：

```text
        ↑
       ↗
        ↑
       ↖
        ↑
```

意味着固定状态中的一部分空间反复竞争。

HMO 将它解释成一种 effective-rank / collision pressure。

---

### \(p\)：Decay Pressure

大致：

$$
p=
\frac{1}{\tau+\epsilon}
$$

旧信息保持越弱，历史信息越容易快速衰减。

最后组合：

$$
\sigma
\approx
0.4\rho+0.3c+0.3p
$$

再经过归一化和跨 layer / token 聚合。

因此：

$$
\boxed{
\sigma_j
=
\text{segment }j\text{ 对 recurrent memory 的风险估计}
}
$$

高 \(\sigma\) 的意思不是：

> 这个 segment 一定重要。

而是：

> **仅依靠 recurrent state 承载这一段信息可能风险较高。**

这点很重要。

---

# 3.3 第三步：Attention 侧依赖 \(\alpha\)

假设一个 segment：

```text
Segment 21
```

recurrent state 很不可靠：

$$
\sigma_{21}=0.95
$$

但用户的问题压根和 Segment 21 没关系。

那就没有必要花很多 KV budget 救它。

所以还需要另外一个信号：

$$
\alpha_j
$$

表示：

> 当前生成过程在 Full Attention 侧有多依赖这一段。

当前原型会进行额外的 attention probe，对 Prompt segment 的 attention dependence 做聚合。

于是：

```text
σ 高：DeltaNet 记得不可靠
α 高：生成又确实需要这里
```

真正危险的 segment 是两者同时高。

---

# 3.4 双通道优先级 \(\phi\)

因此当前采用：

$$
\boxed{
\phi_j=\sigma_j\alpha_j
}
$$

直观理解：

| \(\sigma\) | \(\alpha\) | 含义                               |
| ---------- | ---------- | -------------------------------- |
| 低          | 低          | recurrent 没明显风险，而且当前不依赖          |
| 高          | 低          | recurrent 有压力，但当前任务不太依赖          |
| 低          | 高          | Attention 很需要，但 recurrent 可能仍较可靠 |
| 高          | 高          | **最危险，需要显式记忆保护**                 |

不过这里我要提醒你：

**当前代码实际上把二维关系最终压成了一个乘积 \(\phi\)，并没有真正实现四象限式 policy。**

所以现在它更准确地叫：

> dual-channel priority score

而不是一个真正复杂的 dual-channel decision model。

---

# 3.5 第四步：四种动作

HMO 不只是 Keep / Drop。

它现在有四种动作。

### KV

完整保留这一 segment 的 Full-Attention KV。

当前默认：

```text
第一个 segment → KV
最后一个 segment → KV
```

类似 sink + recent protection。

---

### Refresh

高 \(\phi\) segment 可以分配 Refresh。

直觉是：

> 这一段很重要，但 recurrent memory 可能已经无法可靠承担它，所以重新获得它的高质量 exact memory。

不过**当前 prototype 的 Refresh 有一个非常重要的工程局限**：

它并不是轻量地只重算那 512 个 token。

当前实现会重新 replay 整个输入，然后从 replay 产生的 cache 中拿目标 segment 的 exact K/V 插回 active cache。

所以当前：

$$
1\text{ Refresh}
$$

实际上可能意味着：

$$
1\text{ additional full Prefill}
$$

这会给 TTFT 带来很大压力。

这个问题之后必须解决，否则 Refresh 很难成为 ICLR 论文里的“效率设计”。

---

### RTS

RTS 当前也要特别澄清。

早期代码里留有 SVD low-rank RTS，但 **当前 V6.1 主路径已经不是 SVD**。

现在是 token skeleton：

> 在一个 segment 内，根据跨 Full-Attention layers / heads 的 \(K/V\) norm 给 token 打分，保留少量 top tokens，其余删除。

所以：

```text
512 tokens

↓ RTS

可能只保留
token 7
token 41
token 115
token 231
...
```

这些 token 保留原位置和原 KV。

因此 RTS 本质上是：

$$
\boxed{\text{sparse exact KV coverage}}
$$

---

### Drop

预算不足：

$$
n_{\text{keep}}=0
$$

则整个 segment 从 active KV 中删除。

---

# 3.6 最后是预算分配

主实验：

$$
keep\_ratio=0.10
$$

预算定义大致为：

$$
B
=
B_{\text{protected}}
+
0.1 B_{\text{middle-full-KV}}
$$

于是 V6.1 policy 可以压成：

```text
First / Last
     ↓
exact KV

Middle segments
     ↓
计算 σ
     ↓
计算 α
     ↓
φ = σα
     ↓
高 φ → 少量 Refresh
     ↓
剩余 budget → RTS
     ↓
完全没有 budget → Drop
```

这就是当前 HMO 的主体。

---

# 4. HMO 目前效果怎么样

目前最正式的一组结果是：

$$
\text{Qwen3.5-27B}
$$

$$
T=32768
$$

$$
keep\_ratio=0.10
$$

每个主要任务约 50 samples，LCC 当前只有 18。

| Dataset           |    Full KV |        H2O | SnapKV | Quest-lite | SAGE-KV-lite |    **HMO** |
| ----------------- | ---------: | ---------: | -----: | ---------: | -----------: | ---------: |
| Needle Acc        |     1.0000 |     0.7800 | 1.0000 |     1.0000 |       1.0000 | **1.0000** |
| LongEval Acc      |     1.0000 |     0.1200 | 1.0000 |     1.0000 |       1.0000 | **1.0000** |
| HotpotQA F1       |     0.6795 | **0.6871** | 0.6787 |     0.6787 |       0.6268 |     0.6741 |
| NarrativeQA F1    |     0.3140 |     0.2899 | 0.3105 |     0.3118 |       0.3085 | **0.3162** |
| GovReport ROUGE-L | **0.1901** |     0.1842 | 0.1803 |     0.1850 |       0.1755 |     0.1761 |
| LCC Acc           | **0.5000** |     0.2778 | 0.4444 |     0.4444 |       0.3333 |     0.4444 |

Macro average：

$$
FullKV=0.6139
$$

$$
SnapKV=0.6023
$$

$$
Quest=0.6033
$$

$$
\boxed{HMO=0.6018}
$$

所以目前正确的结论是：

> **HMO 在极低预算下表现有竞争力，但还没有总体超过强 KV compression baseline。**

这点千万不要包装过头。

---

## 4.1 现在最漂亮的结果是什么

主要是 Needle / LongEval：

$$
10\%\text{ middle-KV budget}
$$

仍然：

$$
Acc=1.0
$$

同时 naive budgeted Recent / Uniform KV：

Needle：

$$
0.08/0.04
$$

LongEval：

$$
0/0
$$

说明简单地：

> “随便保留 10% KV”

完全不够。

有意义的 KV selection 确实很重要。

但是 SnapKV / Quest 等强方法同样达到 1.0。

因此它证明的是：

$$
\boxed{\text{HMO policy 有效}}
$$

还没有证明：

$$
\boxed{\text{hybrid-memory-aware policy 优于 attention-only policy}}
$$

这两句话差别非常大。

而后者才是论文真正需要证明的。

---

# 4.2 Memory 数字很好看，但实际 GPU memory 目前没有同步下降

例如 Needle：

$$
2.124\,GB
\rightarrow
0.252\,GB
$$

tracked KV 大约减少：

$$
8.4\times
$$

NarrativeQA：

$$
1.497
\rightarrow
0.206\,GB
$$

约：

$$
7.3\times
$$

从 KV accounting 看非常不错。

但是 peak VRAM：

Needle：

$$
59.39\,GB
\rightarrow59.74\,GB
$$

Narrative：

$$
55.94\,GB
\rightarrow56.17\,GB
$$

基本没有下降，甚至略高。

因为模型权重、Prefill intermediates、额外 probe、Refresh replay 等都会占显存。

所以当前不能在论文里写：

> “HMO 将 GPU memory 减少 8×。”

最多能写：

> “HMO 将 tracked resident attention KV 减少约 7–8×。”

真正系统级 benefit 还没有证明。

---

# 4.3 当前实验还有两个必须修的问题

一个是 baseline fidelity。

目前：

* PyramidKV-lite；
* Quest-lite；
* SAGE-KV-lite；

都是项目自己的适配版本，不完全等同官方实现。

ICLR reviewer 很容易质疑：

> 为什么你的方法和官方 SOTA 不是 apples-to-apples？

第二个甚至更加严重：**metric 实现。**

当前代码中的所谓 Exact Match：

$$
\text{GT}\subseteq\text{Prediction}
$$

实际上是 normalized ground truth 是否出现在 prediction 中，不是严格 exact string match。

当前 F1 又使用：

```python
set(pred_tokens) & set(gt_tokens)
```

而不是标准 QA F1 的 multiset token overlap。

这些正式投稿前必须对齐官方 LongBench / benchmark metrics。

否则 E1 主表都有被重新解释的风险。

---

# 5. 参考近几年 ICLR 录用工作，HMO 能不能迭代成 ICLR

先给结论：

> **HMO 现在已经具备一个有意义的 ICLR research question，但当前版本还不具备 ICLR-ready 的证据强度。**

我会把当前状态评成：

$$
\boxed{\text{有潜力，但当前偏 Weak Reject}}
$$

问题主要不在 idea 完全不行，而在于：

> **目前证据还没有证明 HMO 最关键的那条科学主张。**

---

## 5.1 ICLR 最近接受的这类工作有什么共同特点

StreamingLLM 并不复杂，但它有一个非常清晰的新观察：

$$
\text{Attention Sink}
$$

然后由这个 observation 自然推出方法，并做到了百万级 streaming 和最高 22.2× speedup。([OpenReview][1])

DuoAttention 同样有一个清晰 observation：

$$
\text{Retrieval Heads}
\neq
\text{Streaming Heads}
$$

然后围绕这个观察构造训练、cache layout、prefill、decode 完整系统。([OpenReview][2])

CAKE 的 observation 是：

$$
\text{不同 layer 的 KV demand 不同}
$$

进一步做 global budget allocation，并同时给：

* accuracy；
* memory；
* 128K；
* real decoding latency；

而不是只报告 proxy memory。([OpenReview][3])

Gated DeltaNet 的 ICLR 2025 工作更明显：不仅提出 gated delta rule，还给出对应的 parallel training algorithm，并覆盖 language modeling、retrieval、length extrapolation、long-context 等多类任务。([OpenReview][7])

到 ICLR 2026，PM-KVQ 这样的 KV cache 论文已经做到 7B–70B、多种 reasoning benchmarks，同时给出约 2.73–5.18× throughput improvement。([OpenReview][8])

所以现在 ICLR 对这类 inference efficiency 工作的要求已经很高：

$$
\boxed{
\text{新 observation}
+
\text{方法由 observation 推导}
+
\text{强 baseline}
+
\text{多模型 / 多任务}
+
\text{真实 efficiency improvement}
}
$$

---

# 5.2 HMO 目前最强和最弱的地方

我给当前 HMO 一个严格评分：

| 维度           |       当前评价 | 原因                                                |
| ------------ | ---------: | ------------------------------------------------- |
| 问题重要性        |   **8/10** | Hybrid Attention 正迅速成为实际模型架构                      |
| 核心 intuition |   **8/10** | explicit / recurrent memory mismatch 很自然          |
| 2026 后的新颖性   |   **6/10** | HAM 已高度接近两种记忆互补这一思想                               |
| 方法完整度        |   **6/10** | \(\sigma,\alpha,\phi\)+action 已完整，但 heuristic 较多  |
| 机制证据         |   **4/10** | 还没正式证明 \(\phi\) 比 attention-only 更能预测 memory need |
| 主实验          |   **5/10** | E1 有结果，但 HMO 没有明显超过强 baseline                     |
| 系统效率         |   **3/10** | tracked KV 降了，peak VRAM / latency benefit 尚未建立    |
| 跨模型泛化        |   **3/10** | 正式主结果目前基本仍是 Qwen3.5                               |
| Baseline 严谨性 |   **4/10** | 多个 `-lite` 实现                                     |
| 当前 ICLR 完成度  | **约 5/10** | 核心故事有，但证据链还断着                                     |

---

# 5.3 真正决定 HMO 能否变成 ICLR 的，是 E3

HMO 最应该出现的一张论文核心图不是：

```text
HMO Accuracy > Recent KV
```

这个太弱。

真正应该出现：

$$
x=\sigma,\alpha,\phi
$$

$$
y=\text{oracle value of retaining / refreshing this segment}
$$

然后证明：

$$
\operatorname{Corr}
(\phi,\text{oracle gain})
\gg
\operatorname{Corr}
(\alpha,\text{oracle gain})
$$

并且：

$$
\operatorname{Corr}
(\phi,\text{oracle gain})
\gg
\operatorname{Corr}
(\sigma,\text{oracle gain})
$$

或者至少：

$$
AUC_\phi
>
AUC_\alpha
>
\text{random}
$$

这才真正证明：

> **知道 recurrent state 的情况，确实比只看 Attention 能更准确地知道哪些 KV 值得保存。**

你们早期内部实验里其实已经出现过一个很有希望的迹象：

$$
\sigma\text{ alone 很弱}
$$

而：

$$
\phi=\sigma\alpha
$$

明显更强。

但这目前还不能当正式论文结果。

如果正式 E3 在 Qwen3.5-27B、Kimi Linear、多种任务上都能稳定复现：

$$
\boxed{
\text{hybrid signal}
>
\text{attention-only signal}
}
$$

HMO 的论文价值会一下提升很多。

---

# 5.4 第二个必须解决的是 Refresh

当前 Refresh：

```text
发现 Segment 23 高风险
        ↓
重新 replay 整个 32K prompt
        ↓
取 Segment 23 KV
```

这从 research prototype 来说没问题。

从 efficiency paper 来说问题很严重。

因为你为了：

$$
\text{省几百 MB KV}
$$

可能增加了一次：

$$
32K\text{ Prefill}
$$

甚至多个 Refresh 就多个 Prefill。

ICLR reviewer 很可能会问：

> Memory saving 是用多少 computation 换来的？

如果：

$$
TTFT_{\text{HMO}}
\gg
TTFT_{\text{Full}}
$$

即使 KV 少了很多，系统价值也会明显下降。

所以最终最好把 Refresh 变成：

$$
\boxed{\text{bounded/local replay}}
$$

例如只 replay 一个有限 suffix / segment context，或者能够利用 recurrent state checkpoint 恢复。

有趣的是 DASC 最新工作也出现了 “suffix refresh” 思路，这说明这个方向本身是合理的。([arXiv][5])

---

# 5.5 如果让我决定接下来怎么把它推向 ICLR

优先级我会非常明确：

1. **先做 E3，不要先堆更多复杂 policy。** 首先证明 recurrent signal 本身有科学价值，而且 \(\phi\) 真比 attention-only 更好。
2. **重新做正式 metric + faithful baselines。** 至少 SnapKV、CAKE、Quest、DuoAttention 或同等级强 baseline 要公平。
3. **把真实 efficiency 做出来。** tracked KV、peak VRAM、TTFT、decode latency、throughput 都要有。
4. **解决 Refresh 的 full replay 问题。** 否则 HMO 很可能是“memory-efficient but compute-expensive”。
5. **跨 Hybrid architecture。** Qwen3.5 + Kimi Linear 至少两个 family，最好覆盖不同 recurrent rule。
6. **做完整 budget curve，而不是只做 10%。** 例如 5% / 10% / 20% / 30%，显示 HMO 在极低 memory 区域的 Pareto advantage。
7. **重新定位 novelty。** 避开 HAM 已经占据的“KV + RNN complementary memory”宽泛 claim，强调 **pretrained Hybrid LLM 的 inference-time recurrent-state-aware KV allocation**。

---

## 最终判断

如果现在这个版本直接按照 E1 写成论文投稿，我认为 **ICLR 接收可能性偏低**。用一个仅供项目规划的粗略区间表达，我会放在 **10%～20% 左右**，主要风险会集中在：

> “incremental over KV compression”、“heuristic signals”、“limited evidence of recurrent channel benefit”、“no actual system speed/memory gain”。

但如果接下来能够完成三件事：

$$
\boxed{
\text{E3 得到非常漂亮的机制证据}
}
$$

$$
+
$$

$$
\boxed{
\text{跨 Qwen/Kimi 的稳定 Pareto improvement}
}
$$

$$
+
$$

$$
\boxed{
\text{真实 TTFT/throughput/memory benefit}
}
$$

那么它就会从：

> “一个复杂的 KV compression controller”

转变成：

> **“发现 Hybrid LLM 中 recurrent-memory reliability 可以预测 explicit-KV demand，并据此构建训练无关的推理时记忆编排机制。”**

后者是明显更接近 ICLR 的论文故事。

如果这些关键证据都成立，我会认为它可以进入**有现实竞争力的 ICLR 投稿区间**。最大的风险已经不再是“idea 有没有意义”，而是 **2026 年 HAM、DASC、HYPIC 等工作正在快速占据 Hybrid Memory 这个问题空间，HMO 必须尽快把自己的独特 scientific claim 锁定，并用机制实验把它证明出来**。([arXiv][4])

下一步最值得做的不是继续读代码，而是把 **\(\sigma\)、\(\alpha\)、\(\phi\) 为什么能够代表“需要显式 KV”的逻辑彻底讲透**。这正好对应 HMO 最核心、也最可能决定论文成败的设计。

[1]: https://openreview.net/pdf?id=NG7sS51zVF&utm_source=chatgpt.com "Published as a conference paper at ICLR 2024"
[2]: https://openreview.net/pdf/5723b4f3ab2bb241158f3f35ad3ac5b22b62192e.pdf?utm_source=chatgpt.com "Published as a conference paper at ICLR 2025"
[3]: https://openreview.net/pdf?id=EQgEMAD4kv&utm_source=chatgpt.com "Published as a conference paper at ICLR 2025"
[4]: https://arxiv.org/abs/2603.22325?utm_source=chatgpt.com "Hybrid Associative Memories"
[5]: https://arxiv.org/abs/2608.30386?utm_source=chatgpt.com "DASC: Decay-Aware State Compression for Hybrid Linear-Attention Serving"
[6]: https://arxiv.org/abs/2607.01299?utm_source=chatgpt.com "HYPIC: Accelerating Hybrid-Attention LLM Serving with Position-Independent Caching"
[7]: https://openreview.net/pdf/3b2c0abfb1a7e49a0aec0aacedcce4ef8fbd620d.pdf?utm_source=chatgpt.com "Published as a conference paper at ICLR 2025"
[8]: https://openreview.net/pdf/aa856ae96a62b942c63e9f66b198e0a68f9601e8.pdf?utm_source=chatgpt.com "Under review as a conference paper at ICLR 2026"

---
PZ:接下来应该补设计还是补实验？
---

现在应该**先补实验，不要继续堆设计**。

更准确地说，当前 HMO 已经到了一个很典型的阶段：

> **设计已经足够复杂，真正缺的是证据判断这些设计到底有没有必要。**

如果现在继续加新的 score、新的 action、新的 anchor、新的 policy，很容易把项目做成“不断调 heuristic”，最后论文反而更难讲。

我建议把 V6.1 暂时冻结，把接下来工作按这个顺序推进。

1. **第一优先级：E3 机制实验。** 这是现在最重要的一步。要回答最核心的问题：\(\sigma\) 到底有没有意义？\(\alpha\) 到底有没有意义？\(\phi=\sigma\alpha\) 是否真的比 attention-only 更能预测“这个 segment 值不值得保留/refresh”？最好直接做 oracle gain 相关性、AUC、top-k hit rate，并画出 \(\sigma/\alpha/\phi\) 与 oracle gain 的对照。如果 E3 不成立，继续优化 HMO policy 意义很小；如果 E3 很强，论文最关键的 scientific insight 就立住了。

2. **第二优先级：修正评测严谨性。** 把当前 EM/F1 对齐官方 benchmark metric，并确认 SnapKV、Quest、SAGE-KV 等 baseline 的实现口径。这个最好在大规模补实验之前做，因为 metric 一改，现有 E1 数字可能需要重跑。否则后面跑得越多，返工越大。

3. **第三优先级：E2 消融。** 至少拆成 `sigma-only`、`alpha-only`、`phi`、w/o Refresh、w/o RTS，再和 Full HMO 对比。这里不是为了堆表格，而是验证：“双通道信号”和“四动作空间”到底各自贡献了多少。

4. **第四优先级：budget curve + 第二个 hybrid model。** 现在只有 10% budget 不够。至少补 5% / 10% / 20% / 30%，看 HMO 是否真的形成更好的 memory-quality Pareto curve。同时把 Kimi Linear 正式跑起来，证明它不是 Qwen3.5-specific heuristic。

5. **第五优先级才是补设计。** 这个阶段根据实验暴露的问题再改。最可能需要改的不是 \(\sigma\) 公式，而是 **Refresh**：当前 full replay 的计算代价太大。如果 E3 已经证明“哪些 segment 需要被救”是可预测的，那么再去设计 cheaper Refresh 才有明确目标。

可以把整个决策逻辑理解成：

$$
\text{先验证观察}
\rightarrow
\text{再验证组件}
\rightarrow
\text{再验证泛化和效率}
\rightarrow
\text{最后针对瓶颈改设计}
$$

而不是：

$$
\text{当前结果一般}
\rightarrow
\text{继续加设计}
\rightarrow
\text{再调参数}
\rightarrow
\text{越来越难解释}
$$

现在最需要避免的就是后一条路。

### 我对当前阶段的判断

目前 **HMO 的设计量已经超过实验量**。

你现在已经有：

$$
\sigma=\rho+c+p
$$

$$
\alpha
$$

$$
\phi=\sigma\alpha
$$

以及：

$$
KV / Refresh / RTS / Drop
$$

再加 budget allocation、首尾保护、segment policy，这已经足够一篇论文的方法部分了。

但现在真正还没被回答的是：

> **为什么 recurrent state saturation 应该影响 KV allocation？**

这比“RTS 要不要再改一种打分”“要不要再加 anchor”重要一个数量级。

所以如果让我安排明天开始的工作，我会直接：

> **冻结 V6.1 → 修 metric → 正式跑 E3。**

而且 E3 结果出来以后，我们应该设一个明确的 **go / no-go 判断**：

* 如果 \(\phi\) 显著优于 \(\alpha\)，说明核心假设成立，继续做 HMO；
* 如果 \(\sigma\) 和 \(\phi\) 都几乎没有额外信息量，说明当前 HMO 的 scientific premise 有问题，此时才应该回头改设计；
* 如果 \(\sigma\) 有用但 \(\phi=\sigma\alpha\) 不好，说明问题可能出在信号融合方式，而不是整个方向；
* 如果机制很强但 E1 不强，说明主要问题在 action/policy，需要重新设计调度；
* 如果机制和 E1 都强但 TTFT 很差，就集中优化 Refresh。

这会让后续每一次“改设计”都有实验依据，而不是继续拍脑袋迭代。

**一句话结论：现在先补实验，而且第一件事就是把 E3 做扎实。设计先冻结。**
