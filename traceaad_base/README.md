# TraceAAD Base 独立实现

本包实现[当前 Base 机制](../docs/TraceAAD-Base.md)：不利用轨迹经验的基础演化搜索。演化生成读取当前算法与交叉参考，父代按当前成绩做纯Boltzmann采样，算子使用固定概率。

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

server4共享资源池的18路实验入口见 [实验入口](../experiments/traceaad/README.md)，使用 `--method base`。

默认配置为 1000 次评价、5 个有效初始个体、模型生成温度 1.0、最大输出 16384 tokens、单次模型请求总时限 600 秒。可用 `--budget`、`--initial-size`、`--seed`、`--temperature`、`--max-tokens` 和 `--model-timeout` 调整。父代采样温度单独由 `--selection-temperature` 设置，对应 `Config.selection_temperature`，默认 1.0，单位与成绩相同；实验单路入口和 `launch_host` 也支持该参数。温度不按候选范围自动缩放，六任务的默认值尚未校准。`--no-example` 省略初始样例；`--quiet` 只输出最终 JSON。

`OpenAIClient` 默认以 JSON Schema 约束 `analysis`、`design`、`code` 三个必填字符串，禁止额外字段。端点须支持 `response_format.type=json_schema`；请求不自动降级为普通 JSON 模式。解析仍检查字段非空，格式错误沿用一次修复。

继续本版中断运行时，在相同命令后添加 `--resume`。新运行保存 `generation_protocol=static_algorithm_context_v1`、`selection_rule=boltzmann_fitness_v1` 与父代采样温度；恢复要求选择规则和温度相同。旧 Base 日志使用原始源码快照恢复，不能混入本版。省略的搜索参数从 `run_config.json` 读取。已完成的尝试读回程序库后继续；中断时未完成的尝试直接丢弃。每个尝试的随机数由种子与尝试序号确定，恢复后的抽样与不中断时一致。

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
| [prompts.py](prompts.py) | 当前算法与交叉参考；初始化及失败反馈保持原样；修复读取本次候选和自身错误 |
| [selection.py](selection.py) | 当前排名与实际调用成本；固定算子概率 |
| [task.py](task.py)、[evaluation.py](evaluation.py)、[runtime.py](runtime.py) | 任务适配、独立评价与总时限 |
| [llm.py](llm.py)、[storage.py](storage.py) | OpenAI 兼容请求与追加记录 |

模型依次输出 `analysis`、`design`、`code`；前两项各用两三句，最多100个英文词或200个中文字。`analysis` 随个体保存，说明本次计算变化及其依据。演化展示当前父代的完整代码、自包含设计和成绩，交叉参考展示设计和成绩。初始化保持原样，展示样例、已有初始候选及其分析与成绩，以及最近失败初始尝试的反馈。演化生成不读取祖先、子代、失败子代历史、来源和此前分析。修复保持原样，读取本次候选的代码、设计、分析、来源编号和自身错误，不查询其他节点。`ancestor_count`、`offspring_count` 仅供机制版本使用，`failure_count` 仍控制初始化失败反馈。修复更新同一个体的设计和代码，修复说明接在原分析之后，原始回复及修复前版本仍保存在记录中。父代权重为 `名次^-1.5 × 3/(3+调用数)`，子代不向父代计功；相同源码只算一个候选。程序库不去重，也不截断。

## 运行记录与验证

每个运行目录保存：

- `run_config.json`：任务与搜索配置。
- `attempts.jsonl`：每个完成的尝试一行，含调用摘要、各次评价结果和最终个体；恢复只读这一文件。
- `calls.jsonl`：每次实际请求的提示、回复、用量、耗时与错误。恢复不读这一文件；实验入口在搜索结束后把它压缩为 `calls.jsonl.gz`。

两个日志只追加。进程被杀后，恢复先截掉文件尾未写完的一行；中间行损坏时直接报错。
- `result.json`、`best.py`：停止原因、总成本与最好有效程序；无有效程序时不产生 `best.py`。

离线测试覆盖选择公式、生成对轨迹记录的不变性、共享修复、预算、恢复、请求超时、评价进程清理、八任务接入，以及复制改名后仅用标准库运行：

```bash
uv run pytest -q tests/independent_base
```

V23.2 为 Base 加机制1：生成侧沿用本版提示，选择侧复用 V22.2 的机制1。
