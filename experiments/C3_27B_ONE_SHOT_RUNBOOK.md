# C3 Qwen3.5-27B 单卡一次性运行指南

本文件是当前大模型实验的唯一运行入口。旧
`PHASE2_A100_RUNBOOK.md` / `run_single_a100.sh` 仅用于复现 V6.1 历史实验，
不能用于 C3。

## 1. 冻结范围

- 模型：`Qwen/Qwen3.5-27B` BF16
- revision：`fc05daec18b0a78c049392ed2e771dde82bdf654`
- 官方权重字节数：`55,563,022,432`
- GPU：单张 A100 80GB 或 H100 80GB；进程只暴露一张卡
- 持久盘：至少 120 GiB 空闲，建议 150 GiB
- 协议：`refine-logs/c3_27b_protocol.json`
- 协议 SHA256：`a5121b8d820ae49f8e584659894fee374244ce43d555a38b4f86fa13fa2097d4`
- 必跑核心：preflight 后 432 generation cells

本机已经验证过的参考软件栈是 Python 3.11.15、PyTorch 2.7.0+cu128、
Transformers 5.5.4、NumPy 2.2.6。租用机可以使用等价 CUDA 构建，但必须先
通过 CPU tests 和两-cell preflight；不要在付费核心阶段升级依赖。

## 2. 数据盘布局

```text
/data/hmo/
├── repo/                       # clean HMO checkout
├── models/Qwen3.5-27B/         # pinned BF16 weights
├── datasets/LongBench/data.zip # pinned archive
├── results/c3_27b_<commit>/    # JSONL, summary, manifest, logs
└── cache/                      # HF cache
```

模型与运行结果放持久盘。进程退出后 GPU 显存和主存会释放；权重、cache、
JSONL 和日志仍占磁盘，直到人工删除。若租用平台把数据盘与实例生命周期绑定，
先确认释放实例后数据是否保留。

## 3. 租卡前零 GPU 检查

在当前仓库执行：

```bash
HMO_PYTHON=/home/pz/miniconda3/envs/hmo_research_v6/bin/python \
  bash experiments/phase2/run_c3_27b.sh validate

/home/pz/miniconda3/envs/hmo_research_v6/bin/python -m unittest \
  experiments.test_c3_protocol \
  experiments.test_c3_cost_estimator \
  experiments.test_pareto_runner \
  experiments.test_native_tasks_runner -v
```

租用时 checkout 本次最终 clean commit，并记录 `git rev-parse HEAD`。不要从
有未提交改动的目录启动；launcher 与 run manifest 都会拒绝 dirty tree。

## 4. 准备权重与数据

权重应尽量在计费前下载到持久卷。需要下载时：

```bash
export HF_HOME=/data/hmo/cache/huggingface
hf download Qwen/Qwen3.5-27B \
  --revision fc05daec18b0a78c049392ed2e771dde82bdf654 \
  --local-dir /data/hmo/models/Qwen3.5-27B
```

LongBench archive 必须匹配 SHA256：

```text
cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64
```

本机当前副本位于：

```text
/mnt/nvme0/hmo/datasets/LongBench/5e628be450b7e67fb7ae6e201bd6d8f7056f7672/data.zip
```

## 5. 公共环境变量

```bash
export HMO_PROJECT_ROOT=/data/hmo/repo
export HMO_C3_MODEL_PATH=/data/hmo/models/Qwen3.5-27B
export HMO_C3_RESULTS_ROOT=/data/hmo/results/c3_27b_$(git -C /data/hmo/repo rev-parse --short HEAD)
export HMO_C3_PROBE_ROOT="$HMO_C3_RESULTS_ROOT/probe_cache"
export HMO_LONGBENCH_ARCHIVE=/data/hmo/datasets/LongBench/data.zip
export HMO_PYTHON=/path/to/hmo/env/bin/python
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

核对 `df -h /data/hmo`、`nvidia-smi`、模型 `config.json` 和 archive SHA 后再跑。

## 6. 两-cell Preflight

按仓库约定使用 detached `screen`，日志放外部结果目录：

```bash
mkdir -p "$HMO_C3_RESULTS_ROOT/logs"
screen -dmS hmo_c3_preflight bash -lc \
  'set -o pipefail; cd "$HMO_PROJECT_ROOT"; bash experiments/phase2/run_c3_27b.sh preflight 2>&1 | tee "$HMO_C3_RESULTS_ROOT/logs/preflight.log"'
screen -ls
```

它只运行一条 exact-32K Needle 的 HMO 10% 与 Full，共 2 次 generation。
检查：

```bash
bash experiments/phase2/run_c3_27b.sh status
tail -100 "$HMO_C3_RESULTS_ROOT/logs/preflight.log"
```

`synthetic_preflight/pareto_summary.json` 应为 complete，只有 1 个 budget row；
HMO/Full 都有生成文本和 resident bytes，runtime 中含 peak allocated/reserved、
model load，row 中含 sample preparation 与逐系统耗时。

preflight 不以答案对错决定是否继续。它只回答：是否真实跑通、80GB 是否有余量、
恢复路径是否正确、核心预计需要多少时间和费用。OOM 时保留日志，换 H100 80GB
或修执行基础设施，不能现场改方法与预算。

## 7. 计算费用并确认

输入供应商当前单卡时价：

```bash
"$HMO_PYTHON" experiments/phase2/estimate_c3_cost.py \
  "$HMO_C3_RESULTS_ROOT/synthetic_preflight/pareto_summary.json" \
  --hourly-rate <PRICE_PER_GPU_HOUR>
```

估算覆盖 312 个合成 generation cells 和 120 个原生 QA generation cells，
并加 25% 余量。把输出的 `projected_gpu_hours`、`projected_cost`、GPU 型号和
preflight 峰值发给 PZ；收到明确费用确认后才启动核心。这个确认是成本控制，
不是科学 Gate。

## 8. 核心运行

确认后在一个 detached session 中顺序执行，避免并发争抢显存：

```bash
screen -dmS hmo_c3_core bash -lc \
  'set -o pipefail; cd "$HMO_PROJECT_ROOT"; \
   bash experiments/phase2/run_c3_27b.sh core-synthetic 2>&1 | tee "$HMO_C3_RESULTS_ROOT/logs/core_synthetic.log" && \
   bash experiments/phase2/run_c3_27b.sh core-native 2>&1 | tee "$HMO_C3_RESULTS_ROOT/logs/core_native.log"'
screen -ls
```

- C3-S：24 条 exact-32K 合成样本，5/10/20%，输出 72 rows；Full 每样本只生成一次。
- C3-N：复用 C2 冻结的 24 条原生 QA，10%，输出 24 rows。
- launcher 默认 `--resume`；同目录中已完成的 key 不会再次生成。
- C3-S 与 C3-N 不设置结果 continuation gate，合成完成后直接进入原生任务。

## 9. 完成核对与释放

```bash
wc -l "$HMO_C3_RESULTS_ROOT/synthetic_core/pareto_results.jsonl"
wc -l "$HMO_C3_RESULTS_ROOT/native_core/native_longbench_results.jsonl"
sha256sum "$HMO_C3_RESULTS_ROOT"/*/*summary.json
nvidia-smi
```

预期分别为 72 与 24 rows，两个 summary 均为 `complete`。重点检查所有 compressed
arms 的 `post_query_resident_kv_bytes` 逐 case 相等、manifest 绑定同一 clean commit、
协议 SHA 一致、无重复 key。进程结束即释放 GPU 显存；若实例仍在运行，平台仍可能
继续计费，因此结果同步到持久盘后应停止实例。保留权重可避免下次重新下载，但会持续
占用约 55.6 GB 加 cache 的磁盘空间。

可选扩展不在 launcher 中自动提供。先归档核心结果并完成 result-to-claim，再决定是否
追加样本、20% 原生预算、HotpotQA-32K-Aug 或 64K stress。
