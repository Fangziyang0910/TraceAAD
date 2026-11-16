# EvoTrace v3

V3 以演化树统一组织算法种群、经验压缩与生成上下文：通过路径收缩与子树剪枝成批简化不活跃区域，凝练为边经验（`edge.experience`）与节点经验（`node.experience`），并在同一棵树上按全局名次与谱系共享投入分配父代、按算子职责读取上下文。完整设计见 [EvoTrace-v3设计](../docs/EvoTrace-v3设计.md)。

本包只依赖 Python 标准库，不依赖 V1、V2 或其他方法包。外部任务可直接运行；仓库基准与公共调度器由实验适配器提供。

```python
from evotrace_v3 import Config, EvoTrace, OpenAIClient, Task
from evotrace_v3.allocation import AllocationConfig
from evotrace_v3.topology import TopologyConfig

task = Task(
    "Return a small nonnegative integer.",
    "def solve(x):\n    pass\n",
    lambda source, solve: abs(solve(1)),
    time_limit=2,
)
model = OpenAIClient("http://localhost:8000/v1", "my-model")
search = EvoTrace(
    task, model, config=Config(evaluation_budget=1000),
    allocation_config=AllocationConfig(rank_half_life=10.0),
    topology_config=TopologyConfig(active_evaluations=40, region_size=4),
    run_dir="runs/example_v3",
)
result = search.run()
```

外部任务命令行入口：

```bash
python3 -m evotrace_v3 --task-factory my_task:make_task \
  --run-dir runs/example_v3 --base-url http://localhost:8000/v1 --model my-model
```

核心超参数（独立入口与仓库统一入口均开放同名命令行参数）：
- `--rank-half-life`（默认 `10`）：平均名次每落后多少位，个体基础质量权重减半；分母取当前保留路径最优可行祖先的 `subtree_evaluations`。
- `--active-evaluations`（默认 `40`）：个体出生、作为主父代受评或作为交叉参考被引用后的活跃留存评价窗口；活跃个体、全局最优个体及两两最近公共祖先组成保留骨架。
- `--region-size`（默认 `4`）：路径收缩区域或子树剪枝区域触发简化与反思所需的最小节点数。

模块职责：
- `models.py`：算法节点 `Node`、派生边 `Edge` 与树种群容器 `TreePopulation`；维护评价时钟与祖先链 `subtree_evaluations`。
- `allocation.py`：在去重可行个体上按名次质量与路径最优祖先的累计子树投入全局抽样父代（`rank_decay_lineage_sampling`），并按质量门槛与最大 AST 距离挑选交叉参考。
- `topology.py`：反思树简化（`plan_simplification` / `apply_simplification`），划分路径收缩（`PathContraction`）与子树剪枝（`SubtreePruning`）。
- `reflection.py`：将路径收缩区域合成为边经验，将子树剪枝区域合成为节点经验；当次比较进入提示词，历史比较元数据落盘保存。
- `context.py`：`Refine` 读取演化路径，`Crossover` 读取共同祖先及分叉双支，`Explore` 读取种群全局概览。
- `search.py`：生成、修复、评价、树简化更新与确定性断点恢复。
- `task.py`、`llm.py`、`evaluation.py`、`storage.py`：独立任务封装、模型客户端、隔离评价器与原子文件存储。

仓库实验入口使用 `--method evotrace.v3` 或 `evotrace_v3`，详见[统一实验入口](../experiments/traceaad/README.md)。历史冻结批次需使用各自快照源码恢复。

本地测试：

```bash
.venv/bin/python -m pytest -q tests/evotrace_v3 tests/experiments/test_traceaad_entry.py
```
