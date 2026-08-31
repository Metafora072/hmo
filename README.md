# HMO

Hybrid Memory Orchestration(HMO) 是一个面向 hybrid-attention LLM 的无训练推理时记忆控制器。它联合使用 DeltaNet recurrent-state saturation 与 full-attention dependence 信号，在固定 Attention KV 字节预算下，为长上下文 segment 分配 `KV`、`Refresh`、`RTS` 或 `Drop` 动作。

当前论文主线聚焦证据型长上下文问答与检索。已有正式结果以 Qwen3.5-27B、32K context、10% middle-KV budget 为核心设置。

## 目录

```text
.
├── docs/                 # 论文规划、设计、版本历史与学习材料
├── experiments/          # 原型、Phase 2 正式实验、工具和轻量结果
├── models/               # 本地模型入口；权重和缓存不进入 Git
├── references/           # 参考实现快照，不是运行时依赖
├── AGENTS.md             # 研究与协作约定
└── env.example.sh        # 可移植环境变量模板
```

更细的文档索引见 [docs/README.md](docs/README.md)，实验结果保存规则见 [experiments/results/README.md](experiments/results/README.md)。

## 环境

本机路径、Conda 环境和缓存位置通过环境变量配置。可从 `env.example.sh` 建立本地的 `env.local.sh`，后者已被 `.gitignore` 排除。

正式单卡环境和模型准备流程见 [experiments/PHASE2_A100_RUNBOOK.md](experiments/PHASE2_A100_RUNBOOK.md)。当前代码支持通过以下变量覆盖部署路径：

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

Phase 2 预检：

```bash
bash experiments/phase2/run_single_a100.sh preflight
```

最小 E1：

```bash
python experiments/phase2/e1_main/run.py \
  --model qwen3.5-0.8b \
  --gpu_id 0 \
  --n_samples 2 \
  --benchmarks needle \
  --context-lengths 8192 \
  --methods full_kv,hmo_full \
  --run-name smoke_check
```

## Git 边界

Git 提交核心代码、配置模板、文档以及总结性实验材料。以下内容只保留在本机或外部存储：模型权重、Hugging Face/ModelScope cache、Conda 环境、逐样本 JSONL、运行日志和下载的论文 PDF。

`AI_Production/` 是独立工具仓库，具有自己的 Git 历史和远程地址，不属于 HMO 源码。
