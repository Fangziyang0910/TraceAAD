# EvoTrace v3

V3用树状种群组织候选竞争与派生关系，通过反思树简化成批压缩相关尝试，形成节点和边上的设计经验。父代采样、种群维护与三类生成算子共同使用这棵树。完整规则见[V3设计](../docs/EvoTrace-v3设计.md)。

本包只依赖Python标准库，不依赖V1、V2或其他方法包。外部任务可直接运行；仓库基准与公共调度器由实验适配器提供。

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

外部任务入口：

```bash
python3 -m evotrace_v3 --task-factory my_task:make_task \
  --run-dir runs/example_v3 --base-url http://localhost:8000/v1 --model my-model
```

`rank_half_life`默认10，表示排名每落后10位，质量权重减半；独立入口和仓库实验入口均可用`--rank-half-life`设置。它保持相对名次的选择强度，不随停滞状态切换。收缩只交接经验，保留个体的投入计数不变。

种群没有名额上限。距出生、作为主父代开发或被选作`Crossover`参考不足`active_evaluations`次评价（默认40）的独立可行个体，连同全局最优个体，留在活跃工作集；求两两LCA闭包后，虚树之外的冷退节点唯一划分为中间演化链（$\to V$）与旁支尝试（$\to S$），仅当区域节点数达到`region_size`（默认4）时成批执行简化与反思，不足4个节点的冷退区域原样暂留，不产生单节点碎片化反思。同标准化源码且同实测分数的精确复制品在抽样与工作集中折叠为最早编号，不会获得独立抽样名额。独立入口与实验入口均支持`--active-evaluations`和`--region-size`。AST结构距离只辅助交叉参考选择，不决定生存；嵌入尚未启用。

仓库实验使用`--method evotrace.v3`或`evotrace_v3`，默认结果目录为`evotrace_v3`。评价、调度与最终测试流程见[统一入口](../experiments/traceaad/README.md)。

模块职责：

- `models.py`：个体、派生边及节点自身的直接/子树评价计数；主父代按实际评价记账，交叉参考刷新`last_used`而不计直接投入。
- `allocation.py`：在独立可行个体上按 `2 ** (-平均名次差 / rank_half_life) / (1 + 路径最优可行祖先的 subtree_evaluations)` 全局抽样；记录实际父代概率与谱系峰值计数。
- `topology.py`：反思树简化（工作集留存、LCA虚树闭包与阈值摊还压缩）；剪枝、路径收缩及经验区域划分，保留存活个体的历史投入。
- `reflection.py`：联系问题结构、计算思想与设计取舍，凝练算法改进经验；保留实测比较与同起点的简短参照。
- `context.py`：Refine读形成路径，Crossover读真实共同祖先与双支，Explore读种群个体。
- `search.py`：生成、修复、评价、维护与已完成尝试恢复。
- `task.py`、`llm.py`、`evaluation.py`、`storage.py`：本包自有运行组件。

`attempts.jsonl`中的已提交尝试是恢复依据。简化记录包含更新后的树，恢复不重复反思。使用原任务、配置与`--resume`恢复；入口会读取原生配置中省略的参数，显式改变参数会拒绝恢复。V1/V2以及采用旧生存或预算规则的V3记录须由各自冻结源码恢复。当前方法机制包含`reflective_tree_simplification`，父代选择规则为`rank_decay_lineage_sampling`，简化配置为`active_evaluations`与`region_size`，经验表示为`algorithm_design_experience`。模型只返回一段算法改进经验；程序自动保存区域的比较来源，不要求模型输出编号或逐条解释，旧规则和旧经验格式不能直接续入当前实现。

本地核验：

```bash
.venv/bin/python -m pytest -q tests/evotrace_v3 tests/experiments/test_traceaad_entry.py
```
