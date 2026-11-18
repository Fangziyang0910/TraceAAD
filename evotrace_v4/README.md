# EvoTrace v4

V4 以**树状种群（Tree-Structured Population）**统一组织候选算法、派生脉络与设计经验：虚拟根仅负责初始个体的出生，后续精化（`Refine`）、交叉（`Crossover`）与探索（`Explore`）均从在树可行个体 $p$ 派生子代 $v$ 建立新生边 $p \to v$，当前树上全部存活可行个体共同构成现役种群。每步在全树可行节点上按适应度排名与同一谱系位置已投入评价次数抽样主父代；当相邻骨架节点间的路径区或各骨架节点下的合并子树区累积达到局部阈值 $m$ 时，执行事实反思与同胚树归约，分别凝练为收缩边上的边经验（`edge.experience`）与保留节点/根上的节点经验（`node.experience` / `root_experience`）。完整规范见 [V4 设计文档](../docs/EvoTrace-v4设计.md)。

本包仅依赖 Python 标准库，不依赖 V1–V3 或仓库内其他模块。

```python
from evotrace_v4 import Config, EvoTrace, OpenAIClient, Task
from evotrace_v4.allocation import AllocationConfig
from evotrace_v4.topology import TopologyConfig

task = Task(
    "Return a small nonnegative integer.",
    "def solve(x):\n    pass\n",
    lambda source, solve: abs(solve(1)),
    time_limit=2,
)
model = OpenAIClient("http://localhost:8000/v1", "my-model")
search = EvoTrace(
    task,
    model,
    config=Config(evaluation_budget=1000, initial_size=5),
    allocation_config=AllocationConfig(rank_half_life=6.0, explore_references=6),
    topology_config=TopologyConfig(max_representatives=24, cover_distance=0.15, region_size=4),
    run_dir="runs/example_v4",
)
result = search.run()
```

外部任务独立命令行入口：

```bash
python3 -m evotrace_v4 --task-factory my_task:make_task \
  --run-dir runs/example_v4 --base-url http://localhost:8000/v1 --model my-model
```

仓库统一实验入口使用 `--method evotrace.v4` 或 `evotrace_v4`。

本地测试：

```bash
.venv/bin/python -m pytest -q tests/evotrace_v4 tests/experiments/test_traceaad_entry.py
```
