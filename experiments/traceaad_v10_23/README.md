# V10.23：按改动种类定义的步骤

规则见[按改动种类定义的步骤](../../docs/01-搜索方法/按改动种类定义的步骤.md)，实验条件见[实验准则](../PROTOCOL.md)。执行、调度与结果格式与 [V10.22](../traceaad_v10_22/README.md) 相同。

```bash
# 计划（加 --launch 启动）；server3 上运行前先 push
uv run python -m experiments.infra.remote push
ssh B3-server3 'cd /home/fzy/code/LLM4AD/TraceAAD && .venv/bin/python -m experiments.traceaad_v10_23.launch_host --suite co6 --batch <批次> --eval-workers 8 --scheduler-socket /tmp/traceaad-1005/scheduler.sock'
# 单路
uv run python -m experiments.traceaad_v10_23.run --task tsp_construct --backend server3 --seed 0 --repeat 1 --run-name <运行名> --eval-workers 8
# 冻结后的同规模测试
uv run python -m experiments.infra.evaluate experiments_result/traceaad_v10_23/<任务>/<运行> --primary --condition traceaad
# 诊断
uv run python -m experiments.traceaad_v10_23.diagnose --run-dir experiments_result/traceaad_v10_23/<任务>/<运行>
```

当前批次 `20261009_server3_v1023c`（10 月 9 日晚启动）：六任务各 3 路，种子 0–2，在 server3 运行，调度器管 0–51 号物理核和两个端点各 9 个模型槽位，每次评价至多 8 个实例并行。本机 tmux 会话 `watch_v1023` 每分钟拉取结果（`uv run python -m experiments.infra.remote watch --experiment traceaad_v10_23 --batch 20261009_server3_v1023c`），日志在 `experiments_result/traceaad_v10_23/launch_logs/watch_20261009_server3_v1023c.log`。

当前批次的超时报告写出运行到了哪里，见[行为检查](../../docs/03-现象与检验/2026-10-09-V10.23行为检查.md#超时时说明运行到了哪里)。
