# 实验入口

正式 V10.14 的实现保留原始模块名 `traceaad_v10_14_3`：

```bash
uv run python -m experiments.traceaad_v10_14_3.run --task tsp_construct --run-name trial_1 --budget 1000
```

参数与复现说明见[该版本运行记录](traceaad_v10_14_3/README.md)，结果见[正式实验分析](../docs/03-机制探索与验证/2026-09-29-V10.14-正式实验结果与分析.md)。旧版本和对比方法的运行脚本保留在各自目录；批次记录见 [experiments_result](../experiments_result/README.md)。

V10.15 的独立实现和运行、选择、held-out 入口见 [V10.15 运行说明](traceaad_v10_15/README.md)。

V10.16 修订了 V10.15 的父代分配和 finalist 选择，见 [V10.16 运行说明](traceaad_v10_16/README.md) 和[机制设计](../docs/01-主线机制设计/TraceAAD-V10.16-机制设计.md)。所有方法的训练、选择和 held-out 评价已改为按 CPU 时间计时（`timeout × worker 数` 个 CPU 秒，另设 4 倍墙钟安全上限），实验入口把 BLAS/OpenMP 限为单线程；OBP 每个实例重新执行候选程序，VRPTW 模板写明了 depot 规则。模型采样默认改为 Qwen3.8 官方的非 thinking 配置（temperature 0.7、top_p 0.8、top_k 20、presence_penalty 1.5），并对所有后端显式发送全部采样参数；此前的运行在关闭 thinking 时使用的是 thinking 模式参数。这些是任务协议的变化，与此前结果比较时需要注明。
