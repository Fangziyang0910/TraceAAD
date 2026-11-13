# EvoTrace v2

独立的树状种群搜索包。动态反思简化按最近使用与质量差异一次选定保留算法，补齐必要分叉，同时剪枝和收缩路径，将经验与历史投入写回保留节点和边。完整机制见[EvoTrace V2正式设计](../docs/EvoTrace-v2设计.md)，本页说明使用与实现接口。

V2不依赖V1、TraceAAD或其他方法包。标准库即可运行外部任务；仓库基准和公共调度器由实验适配器提供。

```python
from evotrace_v2 import Config, EvoTrace, OpenAIClient, Task
from evotrace_v2.topology import TopologyConfig

search = EvoTrace(
    task, model,
    config=Config(evaluation_budget=1000),
    topology_config=TopologyConfig(cold_evaluations=40),
    run_dir="runs/example",
)
result = search.run()
```

外部任务入口：

```bash
python -m evotrace_v2 --task-factory my_task:make_task \
  --run-dir runs/example --base-url http://localhost:8000/v1 --model my-model
```

仓库实验入口支持 `--method evotrace.v2` 或 `evotrace_v2`，评价、调度与最终测试流程沿用[统一入口](../experiments/traceaad/README.md)。正式运行使用独立冻结快照，当前批次见[操作说明](../experiments/infra/AGENT_OPERATIONS.md)。

## 核心模块

- `topology.py`：一次质量与差异选择、必要分叉闭包、区域划分与历史统计交接。默认冷却40次评价，无固定容量。
- `reflection.py`：两类区域反思，携带实际代码、差分、评价、开发统计与旧经验；每区一次调用。
- `search.py`：生成、评价、入树与动态更新。初始化完成后检查使用状态并统一维护整棵树。
- `allocation.py`：父代沿用质量与累计尝试采样，交叉参考在所有种群规模使用同一质量与差异规则。
- `context.py`：按路径、分叉和方向读取完整相关材料；仅在真实输入窗口不足时减少展示，记录计数和省略项。
- `task.py`、`llm.py`、`evaluation.py`、`storage.py`：本包独立持有的任务、请求、评价与记录。

`attempts.jsonl`中的已完成尝试是恢复状态的唯一来源。发生简化时，记录触发区域、移除节点、反思请求和更新后的树；恢复不重做反思。`tree_latest.json`是便捷视图。恢复必须使用同一方法、配置与任务。

最近40次评价内出生或用作父代/参考的可行节点，以及前半候选中的质量与差异代表，共同确定保留集合。一次维护只选一次，随后补回必要分叉。没有独立宽度/长度阈值、反思积累队列或固定容量。结构锚点不排斥质量代表。访问和历史统计分别守恒，方向累计评价保留原编号。

旧宽度/长度策略的运行目录需要其原冻结源码恢复；当前代码明确拒绝旧配置。测试与证据见[实施记录](../docs/记录/2026-11-13-动态反思简化实现与验证.md)。
