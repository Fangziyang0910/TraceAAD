# EvoTrace v3

V3把算法个体间的质量竞争与树上的发展投入结合起来。所有保留可行个体按质量权重除以历史子树投入直接抽样，受评后更新种群，将移除经历凝练为节点和边上的经验。完整规则见[V3设计](../docs/EvoTrace-v3设计.md)。

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
    allocation_config=AllocationConfig(quality_ratio=10.0),
    run_dir="runs/example_v3",
)
result = search.run()
```

外部任务入口：

```bash
python3 -m evotrace_v3 --task-factory my_task:make_task \
  --run-dir runs/example_v3 --base-url http://localhost:8000/v1 --model my-model
```

`quality_ratio`默认10，控制相同投入下最好与最差个体的权重比；独立命令行可用`--quality-ratio`设置。它是统一的选择强度，不随任务或停滞状态自动切换。收缩只交接经验，保留个体的投入计数不变。

仓库实验使用`--method evotrace.v3`或`evotrace_v3`，默认结果目录为`evotrace_v3`。评价、调度与最终测试流程见[统一入口](../experiments/traceaad/README.md)。

模块职责：

- `models.py`：个体、派生边及节点自身的直接/子树评价计数；所有生成算子按实际评价记账，交叉参考只读。
- `allocation.py`：按 `quality_ratio ** 排名质量 / (1 + 子树评价数)` 全局抽样；记录实际父代概率。
- `topology.py`：质量与差异代表、个体自身近期活动及必要分叉；移除区域划分，保留存活个体的历史投入。
- `reflection.py`：凝练路径和旁支经历，保留实际比较对象、结果标签与旧经验。
- `context.py`：Refine读形成路径，Crossover读真实共同祖先与双支，Explore读种群个体。
- `search.py`：生成、修复、评价、维护与已完成尝试恢复。
- `task.py`、`llm.py`、`evaluation.py`、`storage.py`：本包自有运行组件。

`attempts.jsonl`中的已提交尝试是恢复依据。简化记录包含更新后的树，恢复不重复反思。使用原任务、配置与`--resume`恢复；V1/V2记录须由各自冻结源码恢复，不能接着当作V3运行。先前逐层选择的V3记录也需使用当时冻结源码；当前规则为`rank_subtree_sampling`。

本地核验：

```bash
.venv/bin/python -m pytest -q tests/evotrace_v3 tests/experiments/test_traceaad_entry.py
```
