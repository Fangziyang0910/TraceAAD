# TraceAAD V16

研究问题与机制设计见[独立方法说明](../../docs/机制设计/V16.md)。

V16 结合同龄表现与近期进步分配开发机会，用修改说明和跨谱系参考组织生成上下文。前者保留仍在有效追赶的方向，后者让后续生成利用具体修改及其评价结果。

## 乐观估计

一个谱系由无父代程序及其沿主父代关系产生的后代组成，年龄为已建立的程序记录数，包含失败、修复与回访。挑战谱系 D 在检查点 r（5、10、20……）的当前成绩可能落后，但近期修改仍在持续缩小差距。只比较阶段成绩会忽略这种进步。

以下两条满足任一条，D 就继续参与开发：

1. **同龄比较**：`b_D(r) ≤ b_C(min(r, e_C))`；
2. **乐观估计**：`b_D(r) − s_D · r ≤ b_C(min(2r, e_C))`。

其中 `b(e)` 是前 `e` 次尝试中的最好有效成绩，`e_C` 是冠军谱系当前的尝试数。`s_D` 是 D 在窗口 `[h, r]` 上每次尝试的平均改进，`h = max(首个有效尝试的年龄, ⌈r/2⌉)`；`h = r` 时 `s_D = 0`，第二条退化为与冠军在 `2r` 处的成绩比较。

窗口从 `r/2` 开始，减少粗糙初版到早期修正的大幅跳变对外推的支配。该估计是确定性的近期趋势外推。其轨迹校准见 [slope_projection.json](../../experiments_result/analysis/v15_20261027/slope_projection.json)。下一次尝试优先给投入最少的活跃挑战谱系；没有合格挑战者时开发冠军。

## 上下文

相近程序的完整描述容易掩盖实际改动，同一谱系的参考又可能缺少新的计算。对应的上下文重复统计见 [context_reuse.json](../../experiments_result/analysis/v16_context_20261027/context_reuse.json)。生成上下文采用以下规则：

1. **修改说明**（`changes=True`）。修改操作的回复在 `design` 之后多写一个 `change` 字段：相对修改对象改了哪些计算或参数值，改前与改后，至多 60 词。形成路径各步、修改对象此前的尝试和全局最好成绩的改进记录都以 `Change` 加结果出现；修改对象、路径起点、参考程序和无父代的程序仍以 `Design` 出现，修改对象的 `Design` 写在它的代码之前。此前尝试的列表注明这些程序已经评价过、不要重复。修复程序沿用原程序的 `change`。
2. **Crossover 指令**（随 `changes`）。要求保留当前算法的主决策规则，只把参考设计中的计算实现到当前代码中。
3. **跨谱系参考程序**（`foreign_reference=True`）。参考程序在其他谱系成绩不劣于其中位数的有效程序中均匀抽取；没有其他谱系的有效程序时用全档案。

## 开关与记录

`projection`、`changes`、`foreign_reference` 默认都为 `True`，可分别关闭近期进步外推、修改轨迹表达和跨谱系参考限制。首个检查点默认 5，停滞发散窗口默认 40。

程序记录的 `change` 保存修改说明。`request.allocation` 记录 `lineage`、`champion`、`effort`、`checkpoint` 与 `projected`，最后一项表示谱系是否只因乐观估计而保持活跃。`summary.json` 的 `v15` 字段保存谱系分配统计。

```bash
.venv/bin/python -m experiments.traceaad_v16.run --task tsp_construct --dry-run
.venv/bin/python -m experiments.traceaad_v16.launch_host \
  --batch <batch> --suite co6 --repeats 3 --budget 1000 \
  --backends server3 --eval-workers 8 \
  --scheduler-socket /tmp/traceaad-1003/scheduler.sock
```

## Python 包入口

本目录自带本版的配置、搜索、提示、选择、个体、评价、历史、状态、存储、执行限制与 `runtime/` 运行实现。公开入口为 `Config`、`Program`、`TraceAAD`、`Evaluation`、`LLM`、`OpenAIAPI`，原有类名 `TraceAADV16` 也可使用。任务评价对象与模型客户端由调用方传入；本包提供源码解析、模型请求、隔离执行和调度客户端。运行需要 Python 3.14 与 NumPy、OpenAI、Requests 等第三方依赖。

```python
from traceaad.v16 import Config, TraceAAD

search = TraceAAD(evaluation=task, llm=client, run_dir="runs/example", config=Config())
summary = search.run()
```

`loop.py` 负责生成、评价、修复与检查点，`search.py` 和包内辅助模块组织本机制。各模块使用包内相对导入；复制整个目录并改名后可作为独立 Python 包导入。实验入口位于 `experiments/traceaad_v16/`，自带运行准备、模型路由、批次启动与存储工具；标准实验从 `benchmarks` 读取任务和数据。
