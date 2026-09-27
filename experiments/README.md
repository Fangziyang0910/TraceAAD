# 实验脚本

这里保存可继续使用的运行入口。原始结果在 [experiments_result](../experiments_result/README.md)，可阅读的成绩表在 [docs/02-实验结果](../docs/02-实验结果/)；旧版本的运行代码已移除，历史数据仍保留。

最新实现为 [V10.14](traceaad_v10_14/README.md)：`uv run python -m experiments.traceaad_v10_14.run --task tsp_construct --run-name trial_1 --budget 1000`。先加 `--dry-run` 可检查配置。它按候选尝试计预算，包含局部重验与独立选择集，不能把其 `--budget` 直接与旧版评价次数等同。

已有正式实验 V10.13：`uv run python -m experiments.traceaad_v10_13.run --task tsp_construct --run-name trial_1 --budget 1000`。其批次使用 `uv run python -m experiments.traceaad_v10_13.launch --batch <批次名> --session-prefix <前缀> --watch`。初始化对照的固定协议入口是 `experiments.traceaad_initialization.launch`。

对比方法使用同一个批量入口，例如 `uv run python -m experiments.launch --method eoh --batch <批次名> --repeats 3 --dry-run`；去掉 `--dry-run` 后启动。可用 `--tasks tsp_construct cvrp_aco` 限定任务，用 `--run-arg=--budget=100` 向各路运行脚本传递参数。各方法的 `run.py` 保存自身的算法参数；通用调度、后端和 held-out 评价位于 `infra/`。

训练监控统一入口：`uv run python -m experiments.monitor --host 0.0.0.0 --port 8765`。在可达网络内通过 `http://<本机IP>:8765` 远程访问；页面可切换实验，查看运行状态、最佳程序和有明确记录序号的最佳值曲线。

运行列表直接展示每路的历史最佳 fitness 阶梯曲线，默认越高越好，也可切换原始任务目标值。空心点为首个有效候选，实心点保留每次严格刷新该路历史最优的突破，不对突破点抽样；悬停、键盘聚焦或点按可查看 fitness、目标值、改进幅度、算子／参考方式和候选来源。V10.14 横轴为候选尝试，V10.13 为评价次数，其他历史记录按其原始序号展示。列表与打开的详情每 15 秒刷新，日志增量读取，不修改训练状态。

顶部同时显示运行均速之和与整批预计搜索剩余时间，每路进度条下显示平均速度和 ETA，详情提供搜索耗时与预计完成时刻。V10.14 按候选预算计速（包含失败、重复和精确重建），优先使用已完成候选记录中的累计运行耗时，排除已记录的暂停间隔；旧版按其预算口径与时间记录计算墙钟均速，不能跨口径比较。至少完成 3 个预算单位且累计 1 分钟后才估算。整批 ETA 取所有待完成运行中最大的剩余时间；有排队、受阻、缺少可靠时间或长期无进展的运行时，不报告完整批次 ETA，并显示可估算路数。超过 15 分钟且超过平均单次耗时 5 倍未见候选完成时暂缓估算，这不等同于判定进程退出。搜索耗尽预算或进入冻结／独立选择阶段后标为“搜索已结束”，其后的选择和测试不在 ETA 中。
