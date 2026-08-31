# 参考材料

`qwen3_5_source/` 保存项目开发期间使用的 Transformers Qwen3.5 源码快照，便于核对层结构、hook 接口和 cache 行为。

该目录不是 HMO 的运行时依赖。实际执行使用当前 Python 环境中安装的 `transformers` 以及 `HMO_MODEL_ROOT` 下模型自带的 remote code。修改参考快照不会自动改变实验行为。
