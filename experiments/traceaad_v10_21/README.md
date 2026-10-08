# V10.21：按实例陈述计算预算

规则见[按实例陈述计算预算](../../docs/01-搜索方法/按实例陈述计算预算.md)，实验条件见[实验准则](../PROTOCOL.md)。

```bash
# 计划（加 --launch 启动）
uv run python -m experiments.traceaad_v10_21.launch_local --suite co6 --batch <批次>
# 单路
uv run python -m experiments.traceaad_v10_21.run --task jssp_construct --backend server3 --seed 0 --repeat 1 --run-name <运行名>
# 冻结后的同规模测试
uv run python -m experiments.infra.evaluate experiments_result/traceaad_v10_21/<任务>/<运行> --primary --condition traceaad
```

当前批次：`experiments_result/traceaad_v10_21/batch_20261008_local_v1021.json`，六个任务各 3 路，server3 两个端点各 9 路，本机评价。
