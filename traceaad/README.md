# TraceAAD 实现

当前重新设计的 Base 独立实现位于 [`traceaad_base/`](../traceaad_base/README.md)，入口为 `python -m traceaad_base`。下表中的 `traceaad.base` 保留旧版机制及其兼容入口。

V14、V15、V16、V17、V18 与 V18.1 的目录各自包含完整实现：配置、搜索、提示、选择、个体、评价、历史、状态、存储，以及本地源码解析、模型客户端和调度客户端。包内模块使用相对导入。整个目录可以复制、改名后独立导入。

| Python 包 | 机制与使用说明 | 实验入口 |
| --- | --- | --- |
| `traceaad.base` | [Base](base/README.md) | [基础入口](../experiments/traceaad/run.py) |
| `traceaad.v14` | [V14](v14/README.md) | [V14 入口](../experiments/traceaad_v14/run.py) |
| `traceaad.v15` | [V15](v15/README.md) | [V15 入口](../experiments/traceaad_v15/run.py) |
| `traceaad.v16` | [V16](v16/README.md) | [V16 入口](../experiments/traceaad_v16/run.py) |
| `traceaad.v17` | [V17](v17/README.md) | [V17 入口](../experiments/traceaad_v17/run.py) |
| `traceaad.v18` | [V18](v18/README.md) | [V18 入口](../experiments/traceaad_v18/run.py) |
| `traceaad.v18_1` | [V18.1](v18_1/README.md) | [V18.1 入口](../experiments/traceaad_v18_1/run.py) |

六个机制包统一导出 `Config`、`Program`、`TraceAAD`、`Evaluation`、`LLM` 和 `OpenAIAPI`，并保留各版原有类名。运行需要 Python 3.14 及 NumPy、OpenAI、Requests 等第三方依赖。调用方传入任务评价对象和模型客户端；也可使用包内导出的接口实现它们。

```python
from traceaad.v17 import Config, TraceAAD

search = TraceAAD(evaluation=task, llm=client, run_dir="runs/example", config=Config())
summary = search.run()
```

各版的 `experiments/traceaad_v*/` 自带单路运行、批次启动、环境配置、模型路由和存储工具。标准实验从 `benchmarks` 读取任务及数据；机制包的任务对象也可由其他调用方提供。

Base 的实现位于 `base/`。`from traceaad import Config, Program, TraceAAD` 保留为指向 Base 的兼容入口。
