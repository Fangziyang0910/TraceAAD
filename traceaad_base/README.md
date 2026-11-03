# TraceAAD Base 独立实现

本包实现[当前 Base 机制](../docs/TraceAAD-Base.md)：用祖先与子代记录指导生成，用直接子代质量和模型调用成本选择父代。

搜索、提示、选择、模型客户端、隔离评价和存储均在本目录内实现。包内使用相对导入；复制整个目录即可独立使用。运行库只依赖 Python 标准库，任务及生成算法所需的 NumPy 等依赖由任务环境提供。当前验证环境为 Linux、Python 3.14。

## 运行

在仓库根目录执行离线示例，实际经过生成回复解析、评价、修复、初始化和演化，不连接模型：

```bash
uv run python -m traceaad_base demo --run-dir /tmp/traceaad-base-demo
```

运行现有基准任务时，模型地址和名称由环境或参数提供，密钥从 `OPENAI_API_KEY` 读取：

```bash
uv run python -m traceaad_base run \
  --task tsp_construct \
  --base-url "$OPENAI_BASE_URL" --model "$OPENAI_MODEL" \
  --run-dir runs/traceaad_base/tsp
```

`--base-url` 使用 OpenAI 兼容接口的 `/v1` 地址，也可直接给出 `/chat/completions` 地址。`--task` 支持仓库的八个任务，读取任务已有的数据、求解器和评价配置；每次评价的总时限取任务的 `timeout_seconds`。

server4共享资源池的18路实验入口见 [实验调度说明](../experiments/traceaad_base/README.md)。

默认配置为 1000 次评价、5 个有效初始个体、温度 1.0、最大输出 16384 tokens、单次模型请求总时限 600 秒。可用 `--budget`、`--initial-size`、`--seed`、`--temperature`、`--max-tokens` 和 `--model-timeout` 调整。`--no-example` 省略初始样例；`--quiet` 只输出最终 JSON。

`OpenAIClient` 默认以 JSON Schema 约束 `analysis`、`design`、`code` 三个必填字符串，禁止额外字段。端点须支持 `response_format.type=json_schema`；请求不自动降级为普通 JSON 模式。解析仍检查字段非空，格式错误沿用一次修复。

继续中断运行时，在相同命令后添加 `--resume`。省略的搜索参数从检查点恢复，显式参数须与原运行一致。已发出的调用和已启动的评价保留计数，中断中的尝试结束后继续下一次生成。模型配置仍由调用方提供。

## 接入自己的任务与模型

```python
from traceaad_base import Config, OpenAIClient, Task, TraceAAD

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
python -m traceaad_base run --task-factory my_tasks:make_task \
  --base-url "$OPENAI_BASE_URL" --model "$OPENAI_MODEL" --run-dir runs/custom
```

`--task-options '{"name": "example"}'` 可向工厂传递关键字参数。模型也可替换为实现 `complete(prompt, *, temperature, max_tokens, timeout)` 的对象，返回文本或 `ModelReply`；该接口每次只发起一次请求，并遵守传入的总时限。

本目录可单独安装：`uv pip install ./traceaad_base`，安装后也可使用 `traceaad-base` 命令。

## 核心实现

| 文件 | 内容 |
| --- | --- |
| [models.py](models.py)、[config.py](config.py) | 个体、尝试记录与默认配置 |
| [search.py](search.py) | 初始化、三算子生成、共享一次修复、评价预算与停止 |
| [prompts.py](prompts.py) | 机制文档中的提示词；8 个祖先、3 个有效子代、2 个失败子代 |
| [selection.py](selection.py) | 当前倒数排名、全部有效直接子代质量与实际调用成本 |
| [task.py](task.py)、[evaluation.py](evaluation.py)、[runtime.py](runtime.py) | 任务适配、独立评价与总时限 |
| [llm.py](llm.py)、[storage.py](storage.py) | OpenAI 兼容请求、完整记录与检查点 |

模型依次输出 `analysis`、`design`、`code`；前两项各用两三句，最多100个英文词或200个中文字。`analysis` 随个体保存，说明本次计算变化及其依据。演化只展示当前父代完整代码；祖先与已有子代通过父代编号、分析、设计和评价呈现局部改进脉络，交叉参考展示设计和评价。修复更新同一个体的三个字段，原始回复及修复前版本仍保存在记录中。有效子代全部参与质量统计，是否超过父代不影响计分。程序库不去重，也不截断。

## 运行记录与验证

每个运行目录保存：

- `run_config.json`、`source/`：任务与搜索配置、实现源码快照。
- `calls/`：每次实际请求的提示、回复、用量、耗时与错误。
- `evaluations/`：每次完整评价的分析、设计、代码、结果与个体归属。
- `programs/`、`attempts/`：个体最终版本、生成及修复归属。
- `state.json`：程序库、计数与随机状态，用于恢复。
- `result.json`、`best.py`：停止原因、总成本与最好有效程序；无有效程序时不产生 `best.py`。

离线测试覆盖选择公式、历史边界、共享修复、预算、恢复、请求超时、评价进程清理、八任务接入，以及复制改名后仅用标准库运行：

```bash
uv run pytest -q tests/independent_base
```
