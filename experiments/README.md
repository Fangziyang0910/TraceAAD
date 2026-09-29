# 实验入口

正式 V10.14 的实现保留原始模块名 `traceaad_v10_14_3`：

```bash
uv run python -m experiments.traceaad_v10_14_3.run --task tsp_construct --run-name trial_1 --budget 1000
```

参数与复现说明见[该版本运行记录](traceaad_v10_14_3/README.md)，结果见[正式实验分析](../docs/03-机制探索与验证/2026-09-29-V10.14-正式实验结果与分析.md)。旧版本和对比方法的运行脚本保留在各自目录；批次记录见 [experiments_result](../experiments_result/README.md)。
