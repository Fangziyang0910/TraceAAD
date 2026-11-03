# TraceAAD V19.2 独立实现

本包基于 [TraceAAD Base](../docs/TraceAAD-Base.md)，采用 [V19.2 机制](../docs/机制设计/V19.2.md)：有界连续开发、每段一个竞争代表和实际源码反馈。当前修订标识为 `bounded_development_20261101`，检查点 schema 5；尚未完成在线性能对照。

搜索、提示、选择、模型客户端、隔离评价和存储均在本目录内实现。包内使用相对导入；复制整个目录即可独立使用。运行库只依赖 Python 标准库，任务及生成算法所需的 NumPy 等依赖由任务环境提供。当前验证环境为 Linux、Python 3.14。

## 运行

在仓库根目录执行离线示例，实际经过生成回复解析、评价、修复、初始化和演化，不连接模型：

```bash
uv run python -m traceaad_v19_2 demo --run-dir /tmp/traceaad-v19-demo
```

运行现有基准任务时，模型地址和名称由环境或参数提供，密钥从 `OPENAI_API_KEY` 读取：

```bash
uv run python -m traceaad_v19_2 run \
  --task tsp_construct \
  --base-url "$OPENAI_BASE_URL" --model "$OPENAI_MODEL" \
  --run-dir runs/traceaad_v19_2/tsp
```

`--base-url` 使用 OpenAI 兼容接口的 `/v1` 地址，也可直接给出 `/chat/completions` 地址。`--task` 支持仓库的八个任务，读取任务已有的数据、求解器和评价配置；每次评价的总时限取任务的 `timeout_seconds`。

server4共享资源池的18路实验入口见 [实验调度说明](../experiments/traceaad_v19_2/README.md)。

默认配置为 1000 次评价、5 个有效初始个体、温度 1.0、最大输出 16384 tokens、单次模型请求总时限 600 秒。可用 `--budget`、`--initial-size`、`--seed`、`--temperature`、`--max-tokens` 和 `--model-timeout` 调整。开发额度默认 `--development-calls 4`（按实际模型请求计，含修复），段外排名混合 `--parent-floor 0.05`。`--no-protect-new-segments` 关闭初始化段优先开发；`--no-local-history` 使用旧上下文，均供消融。`--reselect-parent` 保留请求块和修复规则，仅在普通开发过程的后续Refine重新选段代表。三个首步权重可用 `--refine-weight`、`--explore-weight`、`--crossover-weight` 调整。`--no-example` 省略初始样例；`--quiet` 只输出最终 JSON。

`OpenAIClient` 默认以 JSON Schema 约束 `analysis`、`design`、`code` 三个必填字符串，禁止额外字段。端点须支持 `response_format.type=json_schema`；请求不自动降级为普通 JSON 模式。解析仍检查字段非空，格式错误沿用一次修复。

继续中断运行时，在相同命令后添加 `--resume`。省略的搜索参数从检查点恢复，显式参数须与原运行一致。已发出的调用和已启动的评价保留计数，中断中的尝试结束后继续下一次生成。模型配置仍由调用方提供。

## 接入自己的任务与模型

```python
from traceaad_v19_2 import Config, OpenAIClient, Task, TraceAAD

def evaluate(source, function):
    values = [function(x) for x in (1, 2, 3)]
    if any(value < 0 for value in values):
        raise ValueError("Scores must be nonnegative")
    return sum(values) / len(values)

task = Task(
    task_description="Design a nonnegative scoring function; minimize its mean value.",
    template_program='def score(x: int) -> float:\n    """Return a nonnegative score."""\n    pass\n',
    evaluate_program=evaluate,
    time_limit=30,
    score_definition="Mean score; lower is better.",
)
client = OpenAIClient("http://localhost:8000/v1", "your-model")
search = TraceAAD(task, client, config=Config(), run_dir="runs/custom")
result = search.run()
```

`Task` 提供任务说明、函数模板、评价函数及总时限。`example_code` 和 `example_design` 可选。模板含多个函数时指定 `entrypoint`。评价前检查入口名称和参数，随后在独立进程内装载代码并调用评价函数；全部实例共享一个总时限。随机种子由 `evaluation_seed` 指定。

已有任务对象可用 `Task.from_evaluator(evaluator)` 接入；其 `template_program` 默认同时作为初始样例。需要自行装载源码的任务设置 `executes_source=True`，此时评价函数的第二个参数为 `None`。

独立项目也可提供返回 `Task` 的工厂：

```bash
python -m traceaad_v19_2 run --task-factory my_tasks:make_task \
  --base-url "$OPENAI_BASE_URL" --model "$OPENAI_MODEL" --run-dir runs/custom
```

`--task-options '{"name": "example"}'` 可向工厂传递关键字参数。模型也可替换为实现 `complete(prompt, *, temperature, max_tokens, timeout)` 的对象，返回文本或 `ModelReply`；该接口每次只发起一次请求，并遵守传入的总时限。

本目录可单独安装：`uv pip install ./traceaad_v19_2`，安装后也可使用 `traceaad-v19-2` 命令。

## 核心实现

| 文件 | 内容 |
| --- | --- |
| [models.py](models.py)、[config.py](config.py) | 个体、尝试记录与默认配置 |
| [search.py](search.py) | 初始化、三算子生成、共享一次修复、评价预算与停止 |
| [prompts.py](prompts.py) | 当前/anchor 完整源码；同段最近3个有效、2个失败尝试及实际差分 |
| [selection.py](selection.py) | AST 身份、每段代表、排名分配与有界接续 |
| [task.py](task.py)、[evaluation.py](evaluation.py)、[runtime.py](runtime.py) | 任务适配、独立评价与总时限 |
| [llm.py](llm.py)、[storage.py](storage.py) | OpenAI 兼容请求、完整记录与检查点 |

模型输出 `analysis`、`design`、`code`，始终返回完整代码。analysis/design 是待核对的计算说明，实际父子源码差分与评价结果组成局部反馈。Explore / Crossover 的有效新实现获得出生 episode 的剩余开发额度；连续 Refine 允许暂时退步，同时保留段内与全局最好程序。额度耗尽后回到每段最好代表的排名选择。

不再估计质量尺度、祖先信用或父代—算子后验。失败保存类型和成本，`outcome_fitness=None`，不制造数值奖励。精确 AST 重复仍评价、存档和计费，但不增加段或保护份额。修复前后源码分别保存在评价日志。

默认最多4次模型请求构成一个 episode，repair 也消耗该额度。最后一个请求失败时不能追加修复。0.45/0.30/0.25 是 episode 首步比例，并非总操作比例。具体边界、依据和实验对照见[机制文档](../docs/机制设计/V19.2.md)。

恢复只接受 schema 5。旧 schema 4 记录保留原语义，不在新版中续跑。

## 运行记录与验证

每个运行目录保存：

- `run_config.json`、`source/`：任务与搜索配置、实现源码快照。
- `calls/`：每次实际请求的提示、回复、用量、耗时与错误。
- `evaluations/`：每次完整评价的分析、设计、代码、结果与个体归属。
- `programs/`、`attempts/`：个体最终版本、生成及修复归属。
- `state.json`：程序库、计数与随机状态，用于恢复。
- `result.json`、`best.py`：停止原因、总成本与最好有效程序；无有效程序时不产生 `best.py`。

离线测试覆盖身份去重、段代表、短期退步、修复额度、失败计费、上下文边界、检查点恢复和配置传递；离线通过不代表在线性能改善：

```bash
uv run pytest -q tests/independent_v19_2
```
