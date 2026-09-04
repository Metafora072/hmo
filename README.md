# HMO

Hybrid Memory Orchestration(HMO) 是一个面向 hybrid-attention LLM 的无训练推理时记忆控制器。它联合使用 DeltaNet recurrent-state saturation 与 full-attention dependence 信号，在固定 Attention KV 字节预算下，为长上下文 segment 分配 `KV`、`Refresh`、`RTS` 或 `Drop` 动作。

当前论文主线聚焦证据型长上下文问答与检索。已有正式证据覆盖
Qwen3.5-0.8B/9B 的 8K--16K 机制任务，以及 0.8B 原生 LongBench QA；
Qwen3.5-27B/32K 是已经冻结、尚未付费执行的 C3 最终验证包。

## 目录

```text
.
├── docs/                 # 论文规划、设计、版本历史与学习材料
├── experiments/          # 原型、Phase 2 正式实验、工具和轻量结果
├── models/               # 本地模型入口；权重和缓存不进入 Git
├── openchat/             # PZ 与多模型的证据优先协作记录
├── references/           # 参考实现快照，不是运行时依赖
├── AGENTS.md             # 研究与协作约定
└── env.example.sh        # 可移植环境变量模板
```

更细的文档索引见 [docs/README.md](docs/README.md)，实验结果保存规则见 [experiments/results/README.md](experiments/results/README.md)，研究评审与跨模型决策见 [openchat/README.md](openchat/README.md)。

## 环境

本机路径、Conda 环境和缓存位置通过环境变量配置。可从 `env.example.sh` 建立本地的 `env.local.sh`，后者已被 `.gitignore` 排除。

C3 正式单卡环境和模型准备流程见
[experiments/C3_27B_ONE_SHOT_RUNBOOK.md](experiments/C3_27B_ONE_SHOT_RUNBOOK.md)。
旧 [experiments/PHASE2_A100_RUNBOOK.md](experiments/PHASE2_A100_RUNBOOK.md)
仅用于 V6.1 历史复现。当前代码支持通过以下变量覆盖部署路径：

- `HMO_PROJECT_ROOT`
- `HMO_DATA_ROOT`
- `HMO_MODEL_ROOT`
- `HMO_RESULTS_ROOT`
- `HMO_CONDA_SH`
- `HMO_CONDA_ENV`

## 常用入口

控制器契约 smoke：

```bash
python experiments/test_controller.py
```

C3 零 GPU 协议检查：

```bash
HMO_PYTHON=/path/to/hmo/python bash experiments/phase2/run_c3_27b.sh validate
```

历史 Phase 2 入口需要显式设置 `HMO_ALLOW_LEGACY_PHASE2=1`，不应混入 C3。
最小 C3 GPU preflight 在获得租卡/下载确认后执行：

```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/phase2/run_c3_27b.sh preflight
```

## Git 边界

Git 提交核心代码、配置模板、文档以及总结性实验材料。以下内容只保留在本机或外部存储：模型权重、Hugging Face/ModelScope cache、Conda 环境、逐样本 JSONL、运行日志和下载的论文 PDF。

`AI_Production/` 是独立工具仓库，具有自己的 Git 历史和远程地址，不属于 HMO 源码。
