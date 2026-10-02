# 实验入口

正式 V10.14 的实现保留原始模块名 `traceaad_v10_14_3`：

```bash
uv run python -m experiments.traceaad_v10_14_3.run --task tsp_construct --run-name trial_1 --budget 1000
```

参数与复现说明见[该版本运行记录](traceaad_v10_14_3/README.md)，结果见[正式实验分析](../docs/03-机制探索与验证/2026-09-29-V10.14-正式实验结果与分析.md)。旧版本和对比方法的运行脚本保留在各自目录；批次记录见 [experiments_result](../experiments_result/README.md)。

V10.15 的独立实现和运行、选择、held-out 入口见 [V10.15 运行说明](traceaad_v10_15/README.md)。

V10.16 把种群扩展为程序与生成事件（失败、重复与修复都进入数据结构），按“质量 × 经验”选择出发点，并在上下文中给出从当前程序出发的尝试和实测的调用次数与耗时；入口见 [V10.16 运行说明](traceaad_v10_16/README.md)，设计见[机制设计](../docs/01-主线机制设计/TraceAAD-V10.16-机制设计.md)。

首批 V10.15 实验后的修正（协议与采样）见[机制设计 §10.3](../docs/01-主线机制设计/TraceAAD-V10.15-机制设计.md)；2026-10-02 起父代与 finalist 直接以程序为单位，不再按训练分合并（§10.5）。实验入口把 BLAS/OpenMP 限为单线程，评价超时仍按墙钟计；OBP 每个实例重新执行候选程序，VRPTW 模板写明了 depot 规则。模型采样默认改为 Qwen3.8 官方的非 thinking 配置（temperature 0.7、top_p 0.8、top_k 20、presence_penalty 1.5），并对所有后端显式发送全部采样参数；此前的运行在关闭 thinking 时使用的是 thinking 模式参数。这些是任务协议的变化，与此前结果比较时需要注明。
