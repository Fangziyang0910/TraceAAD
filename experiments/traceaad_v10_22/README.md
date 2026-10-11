# V10.22：从精确输出出发的探索

> 旧本地模型批次已归档。下文保存历史启动与修订记录，“当前”按对应日期读取；现有入口仍可用于新批次，新实验遵守[当前规范](../../docs/08-实验规范与共识/README.md)，旧批次复现使用[历史身份与源码快照](../../docs/08-实验规范与共识/历史实验身份.md)。

> 旧本地模型批次已全部归档（2026-10-11），本页保留方法入口与历史分析；当前运行状态以新批次清单为准。归档见[数据入口](../../archive/local_models_20261011/README.md)。


规则见[从精确输出出发的探索](../../docs/01-搜索方法/从精确输出出发的探索.md)，实验条件见[实验准则](../../docs/08-实验规范与共识/README.md)。执行、调度与结果格式与 [V10.21](../traceaad_v10_21/README.md) 相同。

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
