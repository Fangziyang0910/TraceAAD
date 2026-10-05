# 任务预检

在更换任务集之前，用短搜索检查候选任务是否合适：现有方法是否已经饱和，我们的模型能否写出有效程序，短搜索后还有多少提升空间，两路之间差多少。

候选来自 CO-Bench（36 个运筹问题，数据见 `https://huggingface.co/datasets/CO-Bench/CO-Bench`，本机位于 `/home/fang/code/LLM4AD/data/CO-Bench`，可用环境变量 `COBENCH_DATA` 改）。评价器在 [benchmarks/co_bench](../../benchmarks/co_bench/evaluation.py)：程序是完整的 `solve`，每个实例在单独进程中运行，限时 10 秒（与 CO-Bench 相同），超时的实例得 0；报错或不可行使整次评价失败；分数为 CO-Bench 的相对已知最好解的归一化分，越大越好。

每个候选 2 路 V10.18 搜索，预算 100 次生成，开发集取 CO-Bench 开发集中按规模轮转的 8 个实例，独立选择用测试集中的 8 个实例，最后在完整测试集上评价选中的程序。

```bash
uv run python -m experiments.task_pilot.run --task cob_jssp --backend server3 --seed 0 --run-name r1 --budget 100
uv run python -m experiments.task_pilot.analyze   # 写 experiments_result/task_pilot/pilot_summary.json
```
