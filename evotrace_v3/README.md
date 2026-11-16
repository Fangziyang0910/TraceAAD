# EvoTrace v3

V3 以演化树统一派生历史与经验归属，以覆盖代表集统一空间竞争与预算分配：通过路径收缩与子树剪枝成批简化非骨架区域，凝练为边经验（`edge.experience`）与节点经验（`node.experience`），并在代表集上按名次与直接开发投入抽样父代、按算子职责读取树上下文。完整设计见 [EvoTrace-v3设计](../docs/EvoTrace-v3设计.md)。

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
    topology_config=TopologyConfig(max_representatives=24, cover_distance=0.15, region_size=4),
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
- `--max-representatives`（默认 `24`）：代表集 $R$ 的最大规模；可行个体按适应度从优到劣贪心扫描，不被已选更优代表覆盖者进入 $R$。
- `--cover-distance`（默认 `0.15`）：AST 二元组 Jaccard 距离小于该阈值时视为被更优代表覆盖（设为 `0.0` 即关闭结构覆盖）。
- `--region-size`（默认 `4`）：路径收缩区域或子树剪枝区域触发简化与反思所需的最小节点数。
- `--rank-half-life`（默认 `10`）：在代表集 $R$ 内平均名次每落后多少位，基础质量权重减半；分母为 `1 + direct_evaluations`。

模块职责：
- `models.py`：算法节点 `Node`、派生边 `Edge` 与树种群容器 `TreePopulation`；维护评价时钟与主父代 `direct_evaluations`。
- `topology.py`：贪心覆盖代表集（`representatives`）与反思树简化（`plan_simplification` / `apply_simplification`），划分路径收缩（`PathContraction`）与子树剪枝（`SubtreePruning`）。
- `allocation.py`：在代表集上按名次质量与直接开发投入抽样父代（`representative_rank_direct_sampling`），并在代表集中按质量门槛与最大 AST 距离挑选交叉参考。
- `reflection.py`：将路径收缩区域合成为边经验，将子树剪枝区域合成为节点经验；当次比较进入提示词，历史比较元数据落盘保存。
- `context.py`：`Refine` 读取演化路径，`Crossover` 读取共同祖先及分叉双支，`Explore` 读取种群全局概览。
- `search.py`：生成、修复、评价、树简化更新与确定性断点恢复。
- `task.py`、`llm.py`、`evaluation.py`、`storage.py`：独立任务封装、模型客户端、隔离评价器与原子文件存储。

仓库实验入口使用 `--method evotrace.v3` 或 `evotrace_v3`，详见[统一实验入口](../experiments/traceaad/README.md)。历史冻结批次需使用各自快照源码恢复。

本地测试：

```bash
.venv/bin/python -m pytest -q tests/evotrace_v3 tests/experiments/test_traceaad_entry.py
```
