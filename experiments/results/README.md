# 实验结果保存规则

仓库只提交可审阅、体积稳定的总结性结果：

- `run_manifest.json`（正式运行的不可变参数、代码、模型、环境与指标协议）
- `*_summary.json`
- 汇总 `*.csv`
- 结果表和结论 `*.md`
- 论文图表所需的小型派生数据

以下运行产物由 `.gitignore` 排除：

- 逐样本 `*.jsonl`
- stdout/stderr 日志
- checkpoint、cache 和临时文件
- 大规模中间张量或完整生成轨迹

大型输出应保存在外部高速盘中。仓库内可以保留轻量说明或路由入口，但不要提交模型权重或完整训练/推理输出。

## 正式运行契约

每次正式运行使用独立的 `--run-name` 子目录。runner 会在加载模型前创建一次 `run_manifest.json`，并把它的 `manifest_id` 写入每条 JSONL 和总结文件。只有科学参数、解析后的实验网格、Git commit/branch、模型 snapshot/config、Python/CUDA/依赖环境和固定指标协议完全一致时，`--resume` 才会继续；不得覆盖 manifest，也不得给已有结果的旧目录补挂 manifest。

## 当前结构

```text
results/
├── e1_main/
│   └── <run-name>/      # manifest、逐样本输出及总结
├── archive/v4_copy/     # 从旧 V4 工作区迁移的总结
└── legacy_exports/      # 旧根目录 JSONL，本地保留且不提交
```
