# EvoTrace v1

完整机制见[EvoTrace V1正式设计](../docs/EvoTrace-v1设计.md)，包括树状种群、容量选择、反思简化、预算分配和三类生成上下文。本页说明使用与实现接口。

EvoTrace 在有界树状种群上反复生成、评价、淘汰和凝练经验。这个包独立持有搜索循环、任务接口、模型客户端、评价进程和存储，只依赖Python标准库。方法见[树状种群与反思简化](../docs/树状种群与反思简化.md)。

## 使用

```python
from evotrace_v1 import Config, EvoTrace, OpenAIClient, Task

# evaluate_program(source, function) 返回越小越好的有限实数；time_limit 为整次评价秒数。
task = Task(
    "Minimize the score at x=3.",
    "def solve(x):\n    pass\n",
    lambda source, function: function(3),
    time_limit=2,
)
model = OpenAIClient(base_url, model_name, api_key=api_key)
search = EvoTrace(task, model, config=Config(evaluation_budget=1000), run_dir=run_dir)
result = search.run()
```

也可传入自己的模型对象，实现 `complete(prompt, temperature=..., max_tokens=..., timeout=...)`，返回JSON文本或 `evotrace_v1.models.ModelReply`。`Task.from_evaluator` 可接入已有任务评价器。

初始化默认得到5个可行算法。已评价的失败尝试也入树，`fitness=None`，直到重构时整理；示例代码不入树。每次尝试最多使用一次格式修复或程序修复，修复重评计入评价预算。生成与修复统一输出 `analysis → code → design`：分别说明本次计算选择、完整实现和客观的实现概述。两类反思统一返回单字段 `experience`，作为后续生成的经历材料。

只检查初始化时，从 `evotrace_v1.initialization` 导入 `initialize`，调用 `initialize(task, model, config=..., run_dir=...)`。它运行同一搜索循环并在获得指定数量的可行初始个体时停止，返回树、停止原因、实际评价次数和模型调用次数。

## 独立命令入口

将本目录复制为独立工作目录下的 `evotrace_v1/` 即可使用，也可通过 `pip install ./evotrace_v1` 安装。外部任务文件实现 `make_task()` 并返回 `Task`：

```bash
python -m evotrace_v1 \
  --task-factory my_task:make_task \
  --base-url http://localhost:8000/v1 --model my-model \
  --budget 1000 --run-dir runs/example
```

模型凭据从 `OPENAI_API_KEY` 读取，可用 `--api-key-env` 指定其他环境变量。`--resume` 从原目录恢复。`--task TASK_NAME` 可选择本仓库的任务；只有使用该选项时才导入仓库任务注册表，任务自身的依赖由调用方提供。

公共资源调度和独立测试仍可通过[统一实验入口](../experiments/traceaad/README.md)的 `--method evotrace.v1` 使用。该入口同时服务其他比较方法，EvoTrace使用自身的配置、模型客户端和评价器。

## 代码入口

| 文件 | 职责 |
| --- | --- |
| [search.py](search.py) | 生成、修复、评价、生长、反思、预算与恢复的完整循环 |
| [models.py](models.py) | 程序、尝试、评价、树节点与边，以及入树计数 |
| [topology.py](topology.py) | 质量与差异选择、结构闭包、移除区域划分与重连 |
| [allocation.py](allocation.py) | 加权父代采样、算子比例与交叉参考 |
| [context.py](context.py) | 初始化、修复与三算子的生成提示 |
| [reflection.py](reflection.py) | 路径与旁支的证据、反思调用和经验合并 |
| [task.py](task.py)、[evaluation.py](evaluation.py)、[runtime.py](runtime.py) | 任务接口、合法性检查、进程隔离与超时清理 |
| [llm.py](llm.py)、[storage.py](storage.py) | 模型请求与用量、运行记录与完整行恢复 |
| [cli.py](cli.py)、[initialization.py](initialization.py) | 独立命令与仅初始化入口 |

树节点保存当前可用的算法和选择经验，边保存派生关系与变异经验。方向编号、累计评价数、累计子代数和存活配对跨重构保留。原始程序与尝试日志支持历史复查和恢复。

## 配置与恢复

- `Config`：评价预算、初始化数量、模型采样、随机种子、初始化失败展示与连续未评价上限。
- `TopologyConfig`：容量、质量与差异阈值、新方向保护，见[树状种群管理](../docs/树状种群管理.md)。
- `AllocationConfig`：父代质量系数、算子比例与参考池。
- `ContextConfig`：展示窗口、经验展示和参考源码开关。
- `reflection_model`：默认复用生成模型；`None` 使用确定性摘要。经验展示与反思调用分别控制。

辅助接口直接从所属模块导入，例如 `evotrace_v1.reflection.reflective_simplify`。反思输入输出见[反思简化设计](../docs/反思简化设计.md)。

`run_config.json` 统一记录搜索、种群、分配、上下文与反思配置。`attempts.jsonl` 保存已完成尝试、反思材料和对应的重构快照，是恢复的唯一状态记录；`calls.jsonl` 保存生成请求与响应。`tree_latest.json` 是最近一次已提交重构的便捷视图，不参与恢复。

Python接口用原配置创建 `EvoTrace(..., resume=True)`；命令入口自动读取原配置。运行条件不一致会拒绝恢复。使用旧记录格式的批次继续由其冻结源码恢复。

验证：`uv run pytest -q tests/evotrace_v1`。其中独立运行测试只复制本包，并屏蔽其他研究包的导入。
