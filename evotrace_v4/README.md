# EvoTrace v4

V4 以**树状种群（Tree-Structured Population）**统一组织候选算法、派生脉络与设计经验：虚拟根仅负责初始个体的出生，后续精化（`Refine`）、交叉（`Crossover`）与探索（`Explore`）均从在树可行个体 $p$ 派生子代 $v$ 建立新生边 $p \to v$，当前树上全部存活可行个体共同构成现役种群。每步在全树可行节点上按适应度排名与同一谱系位置已投入评价次数抽样主父代；当相邻骨架节点间的路径区或各骨架节点下的合并子树区累积达到局部阈值 $m$ 时，执行设计反思与同胚树归约，分别凝练为收缩边上的边经验（`edge.experience`）与保留算法节点上的节点经验（`node.experience`）；虚拟根下的子树区直接剪去。完整规范见 [V4 设计文档](../docs/EvoTrace-v4设计.md)。

精化与交叉从初始版本到主父代逐边读取形成路径。每条原始边展示对应子代生成时自述的设计意图，直接取自已有的 `analysis`。每条收缩边展示该段路径的反思经验，后续反思也沿用这份经验。

每个压缩区域凝练一段经验。后续反思可以修订已有经验；模型返回空经验或调用失败时，沿用该区域的已有经验。

三个生成算子分别负责改进当前实现、组合两个实现中的想法，以及尝试另一种计算方式。上下文分别给出候选的设计自述、实测成绩和该位置的局部反思，成绩按实际精度显示。精化读取形成路径、近期尝试与主父代源码；有保留前驱且空间允许时，一并展示其源码。两个位置展示相同代码时共用一份代码，分别保留分数。交叉读取主父代和参考个体的源码。探索读取主父代与至多 5 个异质代表的卡片。默认算子先验为精化 0.50、交叉 0.30、探索 0.20，排名半衰期 $h=6$。

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
    allocation_config=AllocationConfig(rank_half_life=6.0, explore_references=5),
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
