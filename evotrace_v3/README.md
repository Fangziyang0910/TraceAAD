# EvoTrace v3

V3通过树状种群，将算法优胜劣汰与设计经验积累结合。路径收缩时，反思形成的边经验保存在重连边上（`edge.experience`）；子树剪枝时，节点经验保存在相应的保留节点上（`node.experience`）。后续反思继续整合已有经验与新的算法变化及评价结果。预算依据代表算法的适应度排名和以其为主父代消耗的评价次数分配，精化、交叉与探索读取相关演化轨迹和设计经验。研究论证见[研究概述](../docs/树状种群与反思简化.md)，精确规则见[V3设计](../docs/EvoTrace-v3设计.md)。

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
- `--max-representatives`（默认 `24`）：代表集 $R$ 的最大规模；可行个体按适应度从优到劣贪心扫描，不被排序更靠前的已选代表覆盖者进入 $R$。
- `--cover-distance`（默认 `0.15`）：AST 二元组 Jaccard 距离小于该阈值时视为被排序更靠前的代表覆盖（设为 `0.0` 即关闭结构覆盖）。
- `--region-size`（默认 `4`）：路径收缩区域或子树剪枝区域触发简化与反思所需的最小节点数。
- `--rank-half-life`（默认 `10`）：在代表集 $R$ 内平均名次每落后多少位，适应度排名对应的权重减半；分母为 `1 + direct_evaluations`。

模块职责：
- `models.py`：算法节点 `Node`、派生边 `Edge` 与树种群容器 `TreePopulation`；维护主父代 `direct_evaluations`。
- `topology.py`：贪心覆盖代表集（`representatives`）与反思树简化（`plan_simplification` / `apply_simplification`），划分路径收缩（`PathContraction`）与子树剪枝（`SubtreePruning`）。
- `allocation.py`：先按算子权重抽取 `Refine / Explore / Crossover`；`Explore` 直接挂于虚拟根并展示代表集上的最优与结构分散代表；`Refine / Crossover` 按 $2^{-\text{rank}/H}/(1+n_r)$ 抽样主父代，`Crossover` 再按 $2^{-\text{rank}/H}\cdot\operatorname{code\_distance}(\text{parent}, r)$ 软抽样参考代表。
- `reflection.py`：路径收缩与子树剪枝共用一个事实反思模板；区域内每个算法给出来源与结果（根区域附相对当前最优保留分数的基准比较）、`design`、失败原因、已有经验与源码，生成的经验写入收缩边或保留节点。反思失败时沿用区域内已有经验。
- `context.py`：`Refine` 读取主父代的演化路径与从它出发的尝试；`Crossover` 在此基础上加一张带沿途经验与源码的参考代表卡片；`Explore` 只读取根经验及所选已有做法的分数与 `design`（不含局部演化经验与源码）。`include_experiences=False` 是隐藏经验的消融。
- `search.py`：生成、修复、评价、树简化更新与确定性断点恢复。
- `task.py`、`llm.py`、`evaluation.py`、`storage.py`：独立任务封装、模型客户端、隔离评价器与原子文件存储。

仓库实验入口使用 `--method evotrace.v3` 或 `evotrace_v3`，详见[统一实验入口](../experiments/traceaad/README.md)。历史冻结批次需使用各自快照源码恢复。

本地测试：

```bash
.venv/bin/python -m pytest -q tests/evotrace_v3 tests/experiments/test_traceaad_entry.py
```
