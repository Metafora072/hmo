# P0-A 实现与验证报告

日期：2026-09-01
分支：`dev/e3-v2-p0a`

## 结论

P0-A 的代码实现和无 GPU 完整性测试已通过。官方指标、运行身份和结果行现在形成同一条可追溯链路；但这只解除 P0-A 自身的实现阻塞，不解除 GPU gate。按照 E3-v2 预注册，E1、E3 和任何 GPU pilot 仍需等待 P0-B、P0-C、P0-D 完成。

## 1. 官方指标

引入 `experiments/vendor/longbench_metrics.py`，只保留 HMO 当前使用的英文指标，固定上游 LongBench revision：

```text
2e00731f8d0bff23dc4325161044d0ed8af94c1e
```

映射如下：

| 数据集 | 主指标 |
|---|---|
| HotpotQA / NarrativeQA / Qasper | 官方 token F1，使用 `Counter` 保留重复 token 计数 |
| GovReport | 官方 ROUGE-L F1 |
| LCC | 官方首个非注释代码行抽取 + `fuzzywuzzy.fuzz.ratio` |
| Needle / LongEval-Lines | 原有归一化答案包含判定 |

LongBench 子集不再静默回退到自定义 exact match；未注册子集会明确失败。上游 MIT 文本保存在 `experiments/vendor/LONGBENCH_LICENSE`。运行环境和 A100 runbook 固定 `fuzzywuzzy==0.18.0`，manifest 同时记录可选 `python-Levenshtein` 是否存在，避免相似度后端变化不可见。

## 2. 不可变 Run Manifest

新增 `experiments/utils/run_manifest.py`。正式入口在加载模型前创建一次 `run_manifest.json`，其 SHA-256 `manifest_id` 覆盖：

- 原始科学参数，排除纯操作参数 `resume`；
- 解析后的 benchmark、context、method、setting 和 repeat 网格；
- Git commit、branch 和代码工作树状态；
- 模型 registry alias、HF snapshot revision、`config.json` SHA-256、architecture；
- Python、平台、CUDA 可见设备及核心依赖版本；
- 固定 LongBench revision、数据集指标映射和多答案 `max` 约定。

完整性行为：

1. 源码工作树不干净时，正式运行拒绝启动；`experiments/results/**` 运行产物不计入源码脏状态。
2. manifest 使用 exclusive create，不覆盖已有文件。
3. 同目录只有 manifest 完全一致时才能 `--resume`。
4. 科学参数、代码、模型、环境或指标协议变化时拒绝续跑。
5. 非空旧结果目录如果没有 manifest，拒绝事后补挂。
6. 每条正式 JSONL 和 summary 都携带同一个 `manifest_id`。

E1、E2、E4、E5、E6 支持经过路径校验的独立 `--run-name`。legacy E3 默认在模型加载前 fail-closed；只有显式 `--allow-legacy-protocol` 才能为代码审计运行，且不能作为 E3-v2 证据。

## 3. 验证证据

未加载模型，未执行 GPU 实验。

```text
11/11 P0-A unittest passed
all experiments/*.py py_compile passed
run_single_a100.sh bash -n passed
git diff --check passed
all E1-E6 entrypoints --help passed
legacy E3 pre-model gate passed
Qwen3.5-0.8B provenance resolved:
  snapshot 2fc06364715b967f1860aea9cf38778875588b17
  config SHA-256 prefix b90b86f35c8e
```

测试覆盖重复 token F1、multi-ground-truth max、GovReport ROUGE-L、LCC 代码行抽取、未知 LongBench fail-closed、同参续跑、改参拒绝、跨环境拒绝、旧目录拒绝、固定指标 revision 和结果行 `manifest_id` 绑定。

## 4. 后续动作

下一实现包是 P0-B：拆分 context/query，确保 KV 干预发生在 query suffix 处理前，并对干预后的生成独立评分。在 P0-B-D 完整性测试全部通过前，保持 GPU execution BLOCK。
