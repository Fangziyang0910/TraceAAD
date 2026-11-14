# EvoTrace v3

V3把算法个体间的质量竞争与树上的发展投入结合起来。每层比较“直接开发当前个体”和“进入子树”，受评后更新种群，将移除经历凝练为节点和边上的经验。完整规则见[V3设计](../docs/EvoTrace-v3设计.md)。

本包只依赖Python标准库，不依赖V1、V2或其他方法包。外部任务可直接运行；仓库基准与公共调度器由实验适配器提供。

```python
from evotrace_v3 import Config, EvoTrace, OpenAIClient, Task
from evotrace_v3.allocation import AllocationConfig

task = Task(
    "Return a small nonnegative integer.",
    "def solve(x):\n    pass\n",
    lambda source, solve: abs(solve(1)),
    time_limit=2,
)
model = OpenAIClient("http://localhost:8000/v1", "my-model")
search = EvoTrace(
    task, model, config=Config(evaluation_budget=1000),
    allocation_config=AllocationConfig(exploration_weight=1.0),
    run_dir="runs/example_v3",
)
result = search.run()
```

外部任务入口：

```bash
python3 -m evotrace_v3 --task-factory my_task:make_task \
  --run-dir runs/example_v3 --base-url http://localhost:8000/v1 --model my-model
```

仓库实验使用`--method evotrace.v3`或`evotrace_v3`，默认结果目录为`evotrace_v3`。评价、调度与最终测试流程见[统一入口](../experiments/traceaad/README.md)。

模块职责：

- `models.py`：个体、派生边及直接/分支评价计数；所有生成算子按实际评价记账，交叉参考只读。
- `allocation.py`：排名质量加探索项，沿真实树选择一个开发个体；记录逐层选择依据。
- `topology.py`：质量与差异代表、个体自身近期活动及必要分叉；移除区域划分与预算交接。
- `reflection.py`：凝练路径和旁支经历，保留实际比较对象、结果标签与旧经验。
- `context.py`：Refine读形成路径，Crossover读真实共同祖先与双支，Explore读种群个体。
- `search.py`：生成、修复、评价、维护与已完成尝试恢复。
- `task.py`、`llm.py`、`evaluation.py`、`storage.py`：本包自有运行组件。

`attempts.jsonl`中的已提交尝试是恢复依据。简化记录包含更新后的树，恢复不重复反思。使用原任务、配置与`--resume`恢复；V1/V2记录须由各自冻结源码恢复，不能接着当作V3运行。

本地核验：

```bash
.venv/bin/python -m pytest -q tests/evotrace_v3 tests/experiments/test_traceaad_entry.py
```
