# EvoTrace V5

V5 以**树状种群**组织候选算法，使**淘汰即反思（反思式淘汰）**：在纯黑盒目标评价下，每次分支剪枝或路径收缩直接更新树结构，并在对应节点或收缩边上原位凝练一段算法设计经验；更新后的树直接用于**演化脉络上下文与三算子生成**及**分叉局部后验预算分配**。设计细节见 [V5 设计](../docs/EvoTrace-v5设计.md)。

V5 的设置只有 `Config`（评价预算、初始个体数、种子、采样与失败上限），以及 `reflection` 与 `include_experiences` 两个开关。V3/V4 的 AST 覆盖、名次半衰期、探索参考数、局部区域阈值等参数不属于 V5，传入会被拒绝。

## 在 Python 中使用

```python
from evotrace_v5 import Config, EvoTrace, OpenAIClient, Task

task = Task(
    "Return a small nonnegative integer.",
    "def solve(x):\n    pass\n",
    lambda source, solve: abs(solve(1)),
    time_limit=2,
)
model = OpenAIClient("http://localhost:8000/v1", "my-model")
search = EvoTrace(
    task, model, config=Config(evaluation_budget=1000, initial_size=5), run_dir="runs/example_v5"
)
result = search.run()
```

`reflection=False` 关闭反思，`include_experiences=False` 不把经验写入生成上下文。恢复时传 `resume=True`。

## 命令行

自备任务时用 `--task-factory module:function`，返回 `Task` 或带有 `task_description`、`template_program`、`evaluate_program`、`timeout_seconds` 的评价器；`--task-options` 以 JSON 传给它。模型端点只要兼容 OpenAI chat completions，本机测试服务也可以。

```bash
python3 -m evotrace_v5 --task-factory my_task:make_task \
  --run-dir runs/example_v5 --base-url http://localhost:8000/v1 --model my-model
```

`--task` 读取本仓库的基准注册表，只在仓库内可用。凭据从 `--api-key-env` 指定的环境变量读取，默认 `OPENAI_API_KEY`。

设置写入 `RUN_DIR/run_config.json`。加 `--resume` 可恢复当前树、经验、各方向的花费以及原有运行设置。已派发的评价保留其编号。若进程中断且结果未知，这一评按已支出处理，从下一步继续。显式给出不同的值会被拒绝。只有 `--budget` 可以改，用来有意延长预算；新值写回 `run_config.json`，后续恢复沿用。已有运行目录不会被覆盖，没有保存记录的目录不能恢复。

## 统一实验入口

```bash
.venv/bin/python -m experiments.traceaad.run \
  --method evotrace.v5 --task tsp_construct --run-name v5_dry --batch v5_dry \
  --scheduler-socket /tmp/traceaad-1003/scheduler.sock --env-file .env --dry-run
```

`--method evotrace.v5` 与 `evotrace_v5` 等价，结果写入 `experiments_result/evotrace_v5/`。支持 `scheduled` 与 `opendesign-eco` 两种后端、`--reflection/--no-reflection`、`--resume`。V3/V4 的树参数、`--selection-temperature`、`--reference-code` 对 V5 报错。批量启动与队列见[统一入口](../experiments/traceaad/README.md)。

## 本地测试

```bash
.venv/bin/python -m pytest -q tests/evotrace_v5 tests/experiments/test_traceaad_entry.py
```
