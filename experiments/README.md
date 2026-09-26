# 实验脚本

这里保存可继续使用的运行入口。原始结果在 [experiments_result](../experiments_result/README.md)，可阅读的成绩表在 [docs/02-实验结果](../docs/02-实验结果/)；旧版本的运行代码已移除，历史数据仍保留。

单路 V10.13：`uv run python -m experiments.traceaad_v10_13.run --task tsp_construct --run-name trial_1 --budget 1000`。正式批次使用 `uv run python -m experiments.traceaad_v10_13.launch --batch <批次名> --session-prefix <前缀> --watch`。初始化对照的固定协议入口是 `experiments.traceaad_initialization.launch`。

对比方法使用同一个批量入口，例如 `uv run python -m experiments.launch --method eoh --batch <批次名> --repeats 3 --dry-run`；去掉 `--dry-run` 后启动。可用 `--tasks tsp_construct cvrp_aco` 限定任务，用 `--run-arg=--budget=100` 向各路运行脚本传递参数。各方法的 `run.py` 保存自身的算法参数；通用调度、后端和 held-out 评价位于 `infra/`。
