# 本地模型目录

模型权重、Hugging Face cache 和指向本机 cache 的软链均不进入 Git。

运行代码按 `HMO_MODEL_ROOT` 查找模型。默认目录名由 `experiments/utils/model_loader.py` 中的 `MODEL_REGISTRY` 定义，例如：

```text
models/
├── Qwen3.5-0.8B/
├── Qwen3.5-27B/
├── Qwen3.5-27B-GPTQ-Int4/
└── Kimi-Linear-48B-A3B-Instruct-GPTQ-Int4/
```

目录可以是实际权重目录，也可以是指向外部模型存储的软链。
