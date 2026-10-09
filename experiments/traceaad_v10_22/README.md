# V10.22：从精确输出出发的探索

规则见[从精确输出出发的探索](../../docs/01-搜索方法/从精确输出出发的探索.md)，实验条件见[实验准则](../PROTOCOL.md)。执行、调度与结果格式与 [V10.21](../traceaad_v10_21/README.md) 相同。

```bash
# 计划（加 --launch 启动）
uv run python -m experiments.traceaad_v10_22.launch_local --suite co6 --batch <批次> --eval-workers 8 --scheduler-socket /tmp/traceaad-1000/scheduler.sock
# 单路
uv run python -m experiments.traceaad_v10_22.run --task tsp_construct --backend server3 --seed 0 --repeat 1 --run-name <运行名> --eval-workers 8
# 冻结后的同规模测试
uv run python -m experiments.infra.evaluate experiments_result/traceaad_v10_22/<任务>/<运行> --primary --condition traceaad
# 诊断
uv run python -m experiments.traceaad_v10_22.diagnose --run-dir experiments_result/traceaad_v10_22/<任务>/<运行>
```
