# Phase 2 单卡 A100 80GB 保姆式运行指南

这份文档对应当前仓库里的 `Phase 2` 正式实验代码：`E1 ~ E6`。  
目标不是讲原理，而是让你把代码和模型一起放到**租用服务器的数据盘**上，然后**尽量一键跑通**。

这份 runbook 只做部署与运行层面的整理：
- 不改 HMO 理论
- 不改 detector / action space / budget 定义
- 只把现有代码改成更适合“数据盘部署 + 单卡 A100”运行

---

## 1. 最终环境定稿

这份 runbook 只支持**一套最终环境**，不要再在服务器上来回切版本：

- **Python 3.10**
- **PyTorch 2.10.0**
- **torchvision 0.25.0**
- **torchaudio 2.10.0**
- **CUDA wheel: cu128**

一句话：

> 最终就停在 `Python 3.10 + torch 2.10.0 + cu128`，后面不要再改。

### 为什么这次只定这一套

因为这套是当前 Phase 2 最稳的交集：

- 服务器驱动支持 `CUDA 12.8`
- `torch 2.10.0 + cu128` 可以直接匹配当前机器
- `Python 3.10` 对 `gptqmodel==5.7.0` 明显比 `Python 3.12` 稳
- `Qwen3.5` 已经能在这条线上成功加载和跑 smoke
- `Qwen3.5` 的 gated delta rule 已经绑定到 `fla.ops...`，不是纯 `torch_*` fallback

### 明确不要再做的事

- 不要再回到 `torch 2.7.1`
- 不要再追 `torch 2.11`
- 不要再用 `Python 3.12`
- 不要再为了消一条 FLA warning 去推翻整个环境

如果你已经有一套满足下面条件的环境：

- `python -V` -> `Python 3.10.x`
- `torch.__version__` -> `2.10.0+cu128`
- `torch.cuda.is_available()` -> `True`

那就**不要重建环境**，只补缺包。

---

## 2. 推荐的服务器目录布局

因为租用服务器上**代码和模型都要放在数据盘**，所以推荐你直接按下面这个结构来：

```text
/data/hmo/
├── dsf_llm/                # 代码仓库
├── model/                  # 所有模型权重
├── cache/                  # HF / MS 下载缓存（可选）
└── tmp/                    # 临时文件（可选）
```

如果你更喜欢把结果也放到数据盘根目录，也可以这样：

```text
/data/hmo/
├── dsf_llm/
├── model/
├── results/
└── cache/
```

但我更推荐先保持默认：

- 代码：`/data/hmo/dsf_llm`
- 模型：`/data/hmo/model`
- 结果：`/data/hmo/dsf_llm/experiments/results`

这样最省事，因为现在代码已经支持通过环境变量读取这些路径。

---

## 3. 需要多大磁盘

按当前 Phase 2 配置，本地实测/确认过的大致体积是：

- `Qwen3.5-27B BF16`：约 `34G`
- `Qwen3.5-27B-GPTQ-Int4`：约 `25G`
- `Kimi-Linear-48B-A3B-Instruct-GPTQ-Int4`：约 `29G`
- `conda` 环境：约 `15G`
- 仓库代码：不到 `1G`
- 日志、结果、缓存余量：建议至少预留 `20G`

### 建议

- **只跑 Qwen 的 E1/E2/E3/E4/E6**：至少 `100G`
- **Qwen + Kimi 的 E5 也要跑**：至少 `150G`
- **最舒服**：`200G`

一句话版本：

> 直接租 `150G` 以上数据盘，不要卡太死。

---

## 4. 准备数据盘上的代码目录

以下都假设你的数据盘根目录是：

```bash
/data/hmo
```

先建立目录：

```bash
mkdir -p /data/hmo
mkdir -p /data/hmo/model
mkdir -p /data/hmo/cache
mkdir -p /data/hmo/tmp
```

### 4.1 方式 A：通过 GitHub 同步代码

如果你的代码已经在 GitHub 上，这是最推荐的方式。

#### 本地先推到 GitHub

```bash
cd /home/dsf/dsf_llm

# 看看当前改动
git status

# 提交本地改动
git add .
git commit -m "prepare phase2 a100 deployment"

# 推到你的远程分支
git push origin <your-branch>
```

#### 服务器上拉代码

首次拉取：

```bash
cd /data/hmo
git clone <你的仓库地址> dsf_llm
cd /data/hmo/dsf_llm
git checkout <your-branch>
```

如果服务器上已经有一份旧代码，就更新：

```bash
cd /data/hmo/dsf_llm
git fetch origin
git checkout <your-branch>
git pull --ff-only origin <your-branch>
```

如果你用的是私有仓库，记得先配置：

- SSH key
- 或者 GitHub PAT

### 4.2 方式 B：从本地打包再上传

如果你不想走 GitHub，也可以直接在本地压缩后上传到服务器。

#### 本地打包

```bash
cd /home/dsf

tar czf dsf_llm_phase2.tar.gz \
  --exclude='dsf_llm/.git' \
  --exclude='dsf_llm/experiments/results' \
  --exclude='dsf_llm/__pycache__' \
  --exclude='dsf_llm/.pytest_cache' \
  --exclude='dsf_llm/.mypy_cache' \
  --exclude='dsf_llm/*.pyc' \
  dsf_llm
```

把压缩包传上去：

```bash
scp dsf_llm_phase2.tar.gz <user>@<server>:/data/hmo/
```

#### 服务器上解压

```bash
cd /data/hmo
tar xzf dsf_llm_phase2.tar.gz
cd /data/hmo/dsf_llm
```

如果服务器上已经有旧目录，建议先备份或删掉旧代码目录，再解压：

```bash
mv /data/hmo/dsf_llm /data/hmo/dsf_llm_backup_$(date +%Y%m%d_%H%M%S)
```

### 4.3 代码放好之后再继续

不管你是走 GitHub 还是本地压缩上传，最终都应该满足：

```bash
cd /data/hmo/dsf_llm
ls
```

你应该能看到至少这些目录/文件：

- `experiments/`
- `paper/`（如果你把论文也一起传了）
- `CLAUDE.md`
- 仓库里的其他顶层文件

### 4.4 模型不要从本地上传

模型权重体积太大，不建议走本地压缩上传。  
**代码可以传，模型不要传。**

模型应该直接在服务器上下载到：

```bash
/data/hmo/model
```

这样最稳，也最不容易传坏。

### 4.5 最后再进入代码目录

无论你是走 GitHub 还是本地压缩上传，最后都要站到仓库根目录：

```bash
cd /data/hmo/dsf_llm
```

---

## 5. 设置统一环境变量

这一部分很重要。  
现在代码和一键脚本都已经支持下面这些环境变量，所以**不要再手改 Python 文件里的路径**。

把下面这段加入你的 shell：

```bash
export HMO_PROJECT_ROOT=/data/hmo/dsf_llm
export HMO_DATA_ROOT=/data/hmo
export HMO_MODEL_ROOT=/data/hmo/model
export HMO_RESULTS_ROOT=/data/hmo/dsf_llm/experiments/results

# 如果你有自定义 conda 路径，填这里
export HMO_CONDA_SH=/data/miniconda/etc/profile.d/conda.sh
export HMO_CONDA_ENV=hmo_research

# 单卡 A100
export CUDA_VISIBLE_DEVICES=0
```

建议顺手把缓存也放到数据盘：

```bash
export HF_HOME=/data/hmo/cache/huggingface
export HUGGINGFACE_HUB_CACHE=/data/hmo/cache/huggingface/hub
export TRANSFORMERS_CACHE=/data/hmo/cache/huggingface/transformers
export MODELSCOPE_CACHE=/data/hmo/cache/modelscope
export TMPDIR=/data/hmo/tmp
```

创建这些目录：

```bash
mkdir -p "$HMO_RESULTS_ROOT"
mkdir -p "$HF_HOME"
mkdir -p "$HUGGINGFACE_HUB_CACHE"
mkdir -p "$TRANSFORMERS_CACHE"
mkdir -p "$MODELSCOPE_CACHE"
mkdir -p "$TMPDIR"
```

### 建议写进 `~/.bashrc`

这样以后登录服务器就不需要重复导出：

```bash
cat >> ~/.bashrc <<'EOF'
export HMO_PROJECT_ROOT=/data/hmo/dsf_llm
export HMO_DATA_ROOT=/data/hmo
export HMO_MODEL_ROOT=/data/hmo/model
export HMO_RESULTS_ROOT=/data/hmo/dsf_llm/experiments/results
export HMO_CONDA_SH=/data/miniconda/etc/profile.d/conda.sh
export HMO_CONDA_ENV=hmo_research
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/data/hmo/cache/huggingface
export HUGGINGFACE_HUB_CACHE=/data/hmo/cache/huggingface/hub
export TRANSFORMERS_CACHE=/data/hmo/cache/huggingface/transformers
export MODELSCOPE_CACHE=/data/hmo/cache/modelscope
export TMPDIR=/data/hmo/tmp
EOF
source ~/.bashrc
```

### 每个新的 SSH 会话都先做这一步

如果你开了一个新的 shell / tmux pane / 新 SSH 会话，先执行：

```bash
source ~/.bashrc
```

然后再跑 `run_single_a100.sh`。  
不要假设上一会话里导出的 `HMO_*` 变量会自动继承到新会话。

---

## 6. 创建运行环境

### 6.1 如果服务器本身已经有合适环境

只有满足下面三条，才可以直接复用：

- `Python 3.10.x`
- `torch 2.10.0+cu128`
- `torch.cuda.is_available() == True`

如果不是这三条，就不要将就，直接按下面步骤重建 `hmo_research`。

### 6.2 如果需要自己建 conda 环境

```bash
source "$HMO_CONDA_SH"
conda create -n hmo_research python=3.10 -y
conda activate hmo_research
```

先装最关键的基础工具：

```bash
pip install --upgrade pip
pip install setuptools==77.0.3 wheel ninja packaging psutil
```

安装 PyTorch 三件套：

```bash
pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
```

安装实验依赖：

```bash
export PIP_DEFAULT_TIMEOUT=120
export PIP_RETRIES=10

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir \
  transformers accelerate datasets huggingface_hub tokenizers sentencepiece \
  numpy scipy pandas matplotlib seaborn scikit-learn pyarrow \
  einops tqdm loguru tiktoken rouge fuzzywuzzy==0.18.0 \
  optimum bitsandbytes compressed-tensors==0.15.0.1 modelscope
```

安装 GPTQ 相关：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir --no-build-isolation \
  gptqmodel==5.7.0
```

安装 FLA / Kimi 相关依赖：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir \
  fla-core==0.4.2 flash-linear-attention==0.4.2

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir --no-build-isolation \
  causal-conv1d==1.6.1
```

### 6.3 立刻做依赖验证

```bash
python - <<'PY'
import torch
import transformers
import datasets
import importlib

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)

for name in [
    "gptqmodel",
    "compressed_tensors",
    "modelscope",
    "fla",
    "causal_conv1d",
]:
    print(name, "->", bool(importlib.util.find_spec(name)))
PY
```

如果这里不过，不要开始跑实验。

### 6.4 再确认 Qwen3.5 绑定到的 gated delta rule

```bash
python - <<'PY'
from transformers.models.qwen3_5 import modeling_qwen3_5 as m
print("chunk:", m.chunk_gated_delta_rule.__module__, m.chunk_gated_delta_rule.__name__)
print("recurrent:", m.fused_recurrent_gated_delta_rule.__module__, m.fused_recurrent_gated_delta_rule.__name__)
PY
```

理想输出应当包含：

- `fla.ops.gated_delta_rule.chunk`
- `fla.ops.gated_delta_rule.fused_recurrent`

这说明 Qwen3.5 的核心 gated delta rule 已经绑定到 FLA。

如果这里已经是 `fla.ops...`，就不要再为了某一条“cpp extensions” warning 去继续折腾版本。

---

## 7. 把模型直接下载到数据盘

模型不要从本地传，直接在服务器上下载到：

```bash
$HMO_MODEL_ROOT
```

### 7.1 Qwen3.5-27B BF16

推荐优先用 `ModelScope`：

```bash
python - <<'PY'
from modelscope import snapshot_download
print(snapshot_download("Qwen/Qwen3.5-27B", local_dir="/data/hmo/model/Qwen3.5-27B"))
PY
```

### 7.2 Qwen3.5-27B GPTQ-Int4

```bash
python - <<'PY'
from modelscope import snapshot_download
print(snapshot_download("Qwen/Qwen3.5-27B-GPTQ-Int4", local_dir="/data/hmo/model/Qwen3.5-27B-GPTQ-Int4"))
PY
```

### 7.3 Kimi-Linear-48B-A3B-Instruct-GPTQ-Int4

如果你沿用当前项目使用的 Kimi GPTQ 版本，下载到：

```bash
/data/hmo/model/Kimi-Linear-48B-A3B-Instruct-GPTQ-Int4
```

如果你已经有既定来源，就按那个来源来；只要最终目录名保持上面这个即可。

### 7.4 下载后检查目录

```bash
du -sh "$HMO_MODEL_ROOT/Qwen3.5-27B"
du -sh "$HMO_MODEL_ROOT/Qwen3.5-27B-GPTQ-Int4"
du -sh "$HMO_MODEL_ROOT/Kimi-Linear-48B-A3B-Instruct-GPTQ-Int4"
```

如果目录明显只有几百 MB，那就是没下完。

### 7.5 Kimi 兼容性自检

`E5` 依赖 Kimi 的 remote code。你在服务器上拿到 Kimi 模型目录后，建议先做一次快速自检：

```bash
grep -R -nE "def bytes_to_unicode|OutputRecorder|fused_kda_gate" \
  "$HMO_MODEL_ROOT/Kimi-Linear-48B-A3B-Instruct-GPTQ-Int4"
```

理想情况下，你应该能看到：

- `tokenization_kimi.py` 里有 `def bytes_to_unicode():`
- `modeling_kimi.py` 里有 `try: ... OutputRecorder`
- `modeling_kimi.py` 里有：
  - `g = fused_kda_gate(g, self.A_log, dt_bias=self.dt_bias)`

如果没有这三项，说明你拿到的是未兼容当前环境的旧版 remote code，这时再补 Kimi patch。

---

## 8. 进入仓库并做 preflight

```bash
source ~/.bashrc
cd "$HMO_PROJECT_ROOT"
bash experiments/phase2/run_single_a100.sh preflight
```

你应该看到：

- 正确的 `project root / model root / results root`
- Python 版本正确
- `nvidia-smi` 正常
- 三个模型目录能列出来
- `py_compile` 全通过

### 如果 preflight 失败

优先检查：

1. `HMO_MODEL_ROOT` 是否正确
2. `HMO_CONDA_SH` 是否正确
3. `HMO_CONDA_ENV` 是否正确
4. 模型目录是否真的下完

---

## 9. 上机前先做 E1-E6 代码正确性检查

只有 `preflight` 还不够。  
`preflight` 只能说明：

- 路径没写错
- 语法能过
- 模型目录存在

但它**不能**证明 `E1-E6` 在这台租用服务器上都能真正走通各自的主代码路径。  
所以，正式开跑前，建议你先做一轮**最小 smoke**。

### 9.1 先切到单独的 smoke 结果目录

这样不会污染正式结果：

```bash
export HMO_RESULTS_ROOT=/data/hmo/dsf_llm/experiments/results_smoke
mkdir -p "$HMO_RESULTS_ROOT"
```

每次运行必须使用新的 `--run-name`；只有参数、代码、模型和环境完全一致时才能在同一目录 `--resume`。

当前 E1/E3 及全部 GPU smoke 仍受 E3-v2 预注册的 P0-A-D gate 约束；本节命令只在 P0-B、P0-C、P0-D 完成后执行。

### 9.2 E1 smoke

用最小 benchmark / 最小 context / 最小方法集合先确认主循环能走通：

```bash
cd "$HMO_PROJECT_ROOT"
python experiments/phase2/e1_main/run.py \
  --gpu_id 0 \
  --n_samples 2 \
  --benchmarks needle \
  --context-lengths 8192 \
  --methods full_kv,h2o,hmo_full \
  --run-name smoke_check
```

通过标准：

- 结果目录存在：
  - `$HMO_RESULTS_ROOT/e1_main/smoke_check/`
- 至少生成：
  - `e1_main.jsonl`
- 没有一上来就出现模型加载、cache、controller 主路径错误

### 9.3 E2 smoke

```bash
cd "$HMO_PROJECT_ROOT"
python experiments/phase2/e2_ablation/run.py \
  --gpu_id 0 \
  --n_samples 2 \
  --context_length 8192 \
  --run-name smoke_check
```

通过标准：

- 生成：
  - `$HMO_RESULTS_ROOT/e2_ablation/smoke_check/e2_ablation.jsonl`
- 6 个方法都至少能开始写结果

### 9.4 E3 smoke

旧 `e3_mechanism` 协议已降级为 legacy，只保留代码审计用途，不再执行 GPU smoke。待 P0-B-D 完成并实现预注册的 E3-v2 后，在独立 `--run-name` 下补充新的 smoke 命令和通过标准。

### 9.5 E4 smoke

只抽两个 setting，避免一上来跑 13 组：

```bash
cd "$HMO_PROJECT_ROOT"
python experiments/phase2/e4_sensitivity/run.py \
  --gpu_id 0 \
  --n_samples 2 \
  --context_length 8192 \
  --settings budget_1,det_full \
  --run-name smoke_check
```

通过标准：

- 生成：
  - `$HMO_RESULTS_ROOT/e4_sensitivity/smoke_check/e4_sensitivity.jsonl`
- 至少这两个 setting 能正常落盘

### 9.6 E5 smoke

`E5` 已经有现成 smoke 命令，继续沿用：

```bash
cd "$HMO_PROJECT_ROOT"
bash experiments/phase2/run_single_a100.sh e5_smoke
```

通过标准：

- 生成：
  - `$HMO_RESULTS_ROOT/e5_kimi/smoke_check/e5_kimi.jsonl`
- 不出现：
  - CPU/disk offload
  - Kimi remote code 兼容性错误

### 9.7 E6 smoke

```bash
cd "$HMO_PROJECT_ROOT"
python experiments/phase2/e6_overhead/run.py \
  --gpu_id 0 \
  --n_samples 2 \
  --n_repeats 1 \
  --run-name smoke_check
```

通过标准：

- 生成：
  - `$HMO_RESULTS_ROOT/e6_overhead/smoke_check/e6_overhead.jsonl`
- baseline / H2O / HMO 三条 profiling 路径都能至少跑到落盘

### 9.8 smoke 全部通过后，再切回正式结果目录

```bash
export HMO_RESULTS_ROOT=/data/hmo/dsf_llm/experiments/results
mkdir -p "$HMO_RESULTS_ROOT"
```

这一步别忘了，不然你正式实验会继续写进 `results_smoke/`。

### 9.9 对“确保代码完全正确”的真实边界

最诚实的说法是：

- **文档不可能单独保证“完全正确”**
- 能做的是在目标服务器上把：
  - 语法
  - 模型加载
  - 每个实验的主路径
  - 结果落盘
  这四层都 smoke 一遍

如果上面 `E1-E6` smoke 都通过，那么对租用服务器来说，这已经是最接近“可以放心正式开跑”的状态。

---

## 10. 按顺序启动实验

### 最推荐的运行顺序

1. `E1 timing`
2. `E1 formal`
3. `E2`
4. `E3`
5. `E3 analyze`
6. `E4`
7. `E6`
8. `E5 smoke`
9. `E5`

原因：

- `E1 ~ E4, E6` 都是 Qwen 主线
- `E5` 是 Kimi 跨家族验证，风险最高，放后面更稳

---

## 11. 一键命令总表

### 10.1 E1 timing

```bash
cd "$HMO_PROJECT_ROOT"
bash experiments/phase2/run_single_a100.sh e1_timing
```

### 10.2 E1 formal

```bash
cd "$HMO_PROJECT_ROOT"
bash experiments/phase2/run_single_a100.sh e1_formal
```

### 10.3 E2

```bash
cd "$HMO_PROJECT_ROOT"
bash experiments/phase2/run_single_a100.sh e2
```

### 10.4 E3

```bash
cd "$HMO_PROJECT_ROOT"
bash experiments/phase2/run_single_a100.sh e3
```

### 10.5 E3 analyze

```bash
cd "$HMO_PROJECT_ROOT"
bash experiments/phase2/run_single_a100.sh e3_analyze
```

### 10.6 E4

```bash
cd "$HMO_PROJECT_ROOT"
bash experiments/phase2/run_single_a100.sh e4
```

### 10.7 E6

```bash
cd "$HMO_PROJECT_ROOT"
bash experiments/phase2/run_single_a100.sh e6
```

### 10.8 E5 smoke

先 smoke：

```bash
cd "$HMO_PROJECT_ROOT"
bash experiments/phase2/run_single_a100.sh e5_smoke
```

### 10.9 E5 formal

只有 smoke 过了再开：

```bash
cd "$HMO_PROJECT_ROOT"
bash experiments/phase2/run_single_a100.sh e5
```

### 10.10 一键跑 Qwen 主线

```bash
cd "$HMO_PROJECT_ROOT"
bash experiments/phase2/run_single_a100.sh all_qwen
```

---

## 12. 建议用 `tmux` 跑，不要直接挂前台

### 启动 tmux

```bash
tmux new -s hmo_phase2
```

### 在 tmux 里跑

```bash
cd "$HMO_PROJECT_ROOT"
bash experiments/phase2/run_single_a100.sh e1_formal
```

### 脱离 tmux

按：

```text
Ctrl-b d
```

### 重新连回 tmux

```bash
tmux attach -t hmo_phase2
```

---

## 13. 结果目录长什么样

正式 Phase 2 结果会写到：

```text
$HMO_RESULTS_ROOT/
├── e1_main/
├── e2_ablation/
├── e3_mechanism/
├── e4_sensitivity/
├── e5_kimi/
└── e6_overhead/
```

每个实验目录下通常会有：

- `*.jsonl`：逐 cell 增量结果
- `*.json`：summary
- `*.log`：日志

这套 JSONL 是 crash-safe 的：

- 实验中断后可以继续
- `--resume` 会跳过已经完成且没有 `error` 的 cell

所以服务器重启、SSH 断开，不一定要从头跑。

---

## 14. 如何判断实验是不是在正常跑

### 看 GPU

```bash
nvidia-smi
```

### 看日志

例如看 `E1`：

```bash
tail -f "$HMO_RESULTS_ROOT/e1_main/e1_run.log"
```

或者看 jsonl 是否在增长：

```bash
wc -l "$HMO_RESULTS_ROOT/e1_main/e1_main.jsonl"
```

### 看是不是已经有 summary

```bash
ls -lah "$HMO_RESULTS_ROOT/e1_main"
```

如果 `summary.json` 还没有，通常说明还没全跑完。

---

## 15. 最常见的坑

### 14.1 模型路径不对

症状：

- 一启动就找不到模型
- 或者 preflight 里模型目录是 `missing`

解决：

```bash
echo "$HMO_MODEL_ROOT"
find "$HMO_MODEL_ROOT" -maxdepth 1 -mindepth 1 -type d | sort
```

### 14.2 conda 路径不对

症状：

- `run_single_a100.sh` 一开始就说找不到 `conda.sh`

解决：

```bash
export HMO_CONDA_SH=/你的真实conda路径/etc/profile.d/conda.sh
```

### 14.3 Kimi 先别一上来正式跑

`E5` 当前已经改成单卡 A100 路径，但因为它是：

- 跨家族
- 模型大
- GPTQ
- remote code 多

所以**一定先跑 `e5_smoke`**，不要直接正式开整套。

### 14.4 不要再手改硬编码路径

现在这套代码已经支持：

- `HMO_MODEL_ROOT`
- `HMO_RESULTS_ROOT`
- `HMO_PROJECT_ROOT`

所以只改环境变量，不要再去手改 Python 文件里的绝对路径。

---

## 16. 最小可执行清单

如果你只想要一版最短指令，按下面做：

```bash
# 1. 进入数据盘
cd /data/hmo/dsf_llm

# 2. 导环境变量
export HMO_PROJECT_ROOT=/data/hmo/dsf_llm
export HMO_DATA_ROOT=/data/hmo
export HMO_MODEL_ROOT=/data/hmo/model
export HMO_RESULTS_ROOT=/data/hmo/dsf_llm/experiments/results
export HMO_CONDA_SH=/data/miniconda/etc/profile.d/conda.sh
export HMO_CONDA_ENV=hmo_research
export CUDA_VISIBLE_DEVICES=0

# 3. 激活环境
source "$HMO_CONDA_SH"
conda activate "$HMO_CONDA_ENV"

# 4. 建议顺手写进 ~/.bashrc，之后每个新会话只需 source ~/.bashrc
bash -lc 'cat >> ~/.bashrc <<'"'"'EOF'"'"'
export HMO_PROJECT_ROOT=/data/hmo/dsf_llm
export HMO_DATA_ROOT=/data/hmo
export HMO_MODEL_ROOT=/data/hmo/model
export HMO_RESULTS_ROOT=/data/hmo/dsf_llm/experiments/results
export HMO_CONDA_SH=/data/miniconda/etc/profile.d/conda.sh
export HMO_CONDA_ENV=hmo_research
export CUDA_VISIBLE_DEVICES=0
EOF'
source ~/.bashrc

# 5. 预检查
bash experiments/phase2/run_single_a100.sh preflight

# 6. 跑 E1
bash experiments/phase2/run_single_a100.sh e1_timing
bash experiments/phase2/run_single_a100.sh e1_formal

# 7. 跑剩余 Qwen 正式实验
bash experiments/phase2/run_single_a100.sh e2
bash experiments/phase2/run_single_a100.sh e3
bash experiments/phase2/run_single_a100.sh e3_analyze
bash experiments/phase2/run_single_a100.sh e4
bash experiments/phase2/run_single_a100.sh e6

# 8. 最后再试 Kimi
bash experiments/phase2/run_single_a100.sh e5_smoke
bash experiments/phase2/run_single_a100.sh e5
```

---

## 17. 最后一句建议

如果你只能记住一件事，就记这个：

> 先把代码和模型都放到数据盘，先配好 `HMO_*` 环境变量，再跑 `preflight`；`preflight` 不过，就不要开正式实验。

这样会省掉后面绝大多数折腾。
