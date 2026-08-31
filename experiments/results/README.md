# 实验结果保存规则

仓库只提交可审阅、体积稳定的总结性结果：

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

## 当前结构

```text
results/
├── e1_main/             # 当前 E1 运行及总结
├── archive/v4_copy/     # 从旧 V4 工作区迁移的总结
└── legacy_exports/      # 旧根目录 JSONL，本地保留且不提交
```
