# EvoTrace v2

EvoTrace以反思压缩搜索经历，使树状种群同时承载可开发实现与设计经验。动态反思简化按成绩、代码差异与实际使用确定保留算法，再由保留结构确定路径与旁支区域，将其经历凝练到承接关系的节点和边。完整机制见[EvoTrace V2正式设计](../docs/EvoTrace-v2设计.md)，本页说明使用与实现接口。

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

- `topology.py`：一次质量与差异选择、必要分叉闭包与区域划分。近期按版本自身或同路线父代的局部开发判断，默认冷却40次评价，无固定容量。
- `reflection.py`：两类区域反思，读取实际代码、差分、评价与旧经验，输出算法设计经验；每区一次调用。路径提供中间受评实现及明确的净结果；比较始终指向尝试的实际父代，虚拟根不提供独立冠军对照；程序将当前比较转换为结果标签，日志和路径经验前缀保留分数，模型区分观察和可能解释。
- `search.py`：生成、评价、入树与动态更新。初始化完成后检查使用状态并统一维护整棵树。
- `allocation.py`：采样前在当前树重算质量代表，用代表成绩的中位数与MAD归一化；全部保留可行节点仍参与竞争。父代按路线最大节点权重竞争，路线内部按同一质量与开发折扣权重采样，交叉参考在所有种群规模使用同一质量与差异规则。
- `context.py`：Refine读取本路线发展，跨路线Crossover分别读取两条路线及来源摘要，Explore读取去除重复设计描述后的方向经验；当前父代的经验只展示一次。仅在真实输入窗口不足时减少展示，记录计数和省略项。
- `task.py`、`llm.py`、`evaluation.py`、`storage.py`：本包独立持有的任务、请求、评价与记录。

`attempts.jsonl`中的已完成尝试是恢复状态的唯一来源。发生简化时，记录触发区域、移除节点、反思请求和更新后的树；恢复不重做反思。`tree_latest.json`是便捷视图。恢复必须使用同一方法、配置与任务。

保留规则、经验归属、采样公式与默认参数统一见[正式设计](../docs/EvoTrace-v2设计.md)。修改运行条件后应建立新的冻结批次；恢复使用原批次配置与源码。
