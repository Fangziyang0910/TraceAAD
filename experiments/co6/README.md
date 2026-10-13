# 六个组合优化任务：实验入口

数据、限时、模型、选择与报告规则见[实验规范与共识](../../docs/08-实验规范与共识/README.md)；任务选择依据见[任务设计](../../docs/04-研究认识与构想/2026-10-07-六个组合优化任务的设计.md)。本页提供源码与运行入口。

## 任务、接口与框架

每个任务对应自己的目录。模板保存问题定义、函数签名与可修改起始代码；评价器保存固定框架或完整解核验逻辑；生成器按种子重建数据，无需实例文件。当前采用四项函数进化、两项完整求解器进化。

任务目录：[TSP](../../benchmarks/tsp_construct/)、[CVRP](../../benchmarks/cvrp_aco/)、[FSSP](../../benchmarks/fssp_gls/)、[图着色](../../benchmarks/graph_colouring/)、[JSSP](../../benchmarks/jssp_construct/)、[OP](../../benchmarks/op_aco/)。规模、函数签名、固定框架与计分表集中在规范正文第1节。

JSSP的每个作业有自己的机器顺序，与FSSP中全部作业沿同一机器顺序加工不同。`processing_times[j,k]`是作业j第k道工序的时长，`machine_order[j,k]`才是机器ID；k是**工序位置**。`solve`返回同形状的有限非负开始时间矩阵。模型可整体替换构造与搜索算法；评价器仅检查作业先后与机器互斥并计算makespan。[模板](../../benchmarks/jssp_construct/template.py)、[评价器](../../benchmarks/jssp_construct/evaluation.py)、[数据](../../benchmarks/jssp_construct/dataset.py)

图着色的 `solve` 输入完整邻接矩阵，返回每个顶点的非负整数颜色。模型可整体替换DSATUR起始代码，自行决定构造、修复、局部搜索和重启。评价器仅检查相邻顶点异色，并统计实际不同颜色数。计算预算见规范正文第3节。

两项接口标识为 `full-solver-v1`。旧评分函数实验使用其启动源码与配置。

OP函数在每实例开始时调用一次，返回与距离矩阵同形的边先验，由所有蚂蚁与迭代复用。外层限制访问不重复、访问后仍可返回仓库，并保证总路程不超过预算。目标为最大化收集奖励，评价返回负平均奖励。[模板](../../benchmarks/op_aco/template.py)、[评价器](../../benchmarks/op_aco/evaluation.py)、[数据](../../benchmarks/op_aco/dataset.py)

## 命令

```bash
uv run python -m experiments.traceaad_v12_0.run --task jssp_construct --dry-run
uv run python -m experiments.traceaad_v12_0.launch_host --suite co6 --batch <新批次> --repeats 3 --budget 1000
```

启动入口默认仅打印计划；加 `--launch` 才启动。各基线使用同一任务注册：

```bash
uv run python -m experiments.infra.local_queue --method eoh --suite co6 --batch <批次> --slots 3 --dry-run
```

冻结后运行同规模测试：

```bash
uv run python -m experiments.infra.evaluate <运行目录> --primary
```
