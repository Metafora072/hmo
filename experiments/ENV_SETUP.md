# HMO Research — Environment & Code Documentation

**Project**: Hybrid Memory Orchestration (HMO) for NeurIPS 2026
**Created**: 2026-04-14
**Conda Env**: `hmo_research`

---

## Hardware

| Item | Spec |
|------|------|
| GPU | 2× NVIDIA RTX 5090 (32GB each) |
| 实验用GPU | **GPU 1 only** (`CUDA_VISIBLE_DEVICES=1`) |
| GPU 0 | 占用中，不可使用 |

## Conda Environment

```bash
conda activate hmo_research
```

- Python: 3.11
- PyTorch: 2.7.0+cu128
- CUDA (driver): 13.0
- CUDA (nvcc): 12.0

## Core Dependencies

| Package | Version | 用途 |
|---------|---------|------|
| torch | 2.7.0+cu128 | 基础框架 |
| transformers | 5.5.4 | 模型加载/推理 |
| accelerate | 1.13.0 | 模型分片/设备映射 |
| datasets | 4.8.4 | HuggingFace数据集 |
| tokenizers | 0.22.2 | 分词器 |
| sentencepiece | 0.2.1 | 分词器后端 |
| modelscope | 1.35.4 | 国内模型下载 |
| auto-gptq | (installed) | GPTQ量化模型加载 |
| optimum | 2.1.0 | 量化推理优化 |
| bitsandbytes | 0.49.2 | 量化支持 |
| numpy | 2.4.3 | 数值计算 |
| pandas | 3.0.2 | 数据处理 |
| matplotlib | 3.10.8 | 绘图 |
| seaborn | 0.13.2 | 统计绘图 |
| scikit-learn | 1.8.0 | 指标计算 |
| scipy | 1.17.1 | 统计检验 |
| einops | 0.8.2 | 张量操作 |
| tqdm | 4.67.3 | 进度条 |
| loguru | 0.7.3 | 日志 |

## Directory Structure

```
experiments/
├── ENV_SETUP.md          # 本文件
├── CODE_LOG.md           # 代码变更日志
├── utils/                # 共享工具
│   ├── model_loader.py   # 模型加载（Qwen3.5 hybrid架构适配）
│   ├── eval_harness.py   # 统一评测框架
│   ├── dataset_utils.py  # 数据集加载与预处理
│   ├── metrics.py        # 指标计算（accuracy, F1, latency等）
│   └── hooks.py          # DeltaNet层hook工具
├── v1_saturation/        # V1: saturation detection验证
├── v2_refresh/           # V2: state-to-token refresh验证
├── v3_rts/               # V3: RTS skeleton验证
├── v4_joint/             # V4: 联合HMO验证
├── results/              # 实验结果输出
└── logs/                 # 运行日志
```

## Model Matrix

| 模型 | 量化 | 权重大小 | KV cache@128K | 总显存 | Phase |
|------|------|---------|--------------|--------|-------|
| Qwen3.5-0.8B | BF16 | ~1.6GB | ~0.5GB | ~3GB | Phase 1 验真 |
| Qwen3.5-4B | BF16 | ~8GB | ~1GB | ~11GB | Phase 1 双重验证 + Phase 2 消融 |
| Qwen3.5-9B | GPTQ-Int4 | ~4.5GB | ~4GB | ~12GB | Phase 2 主实验 |
| Kimi-Linear-48B-A3B | GGUF IQ4_XS | ~26.5GB | 极小(MLA) | ~28GB | Phase 2 跨家族(4K-8K) |

## Usage

所有实验脚本统一使用 GPU 1：

```bash
source /home/dsf/anaconda3/etc/profile.d/conda.sh
conda activate hmo_research
CUDA_VISIBLE_DEVICES=1 python experiments/v1_saturation/run.py
```
