# 六个组合优化任务：实验入口

主实验采用 **TSP50、CVRP50、FSSP50×20、图着色300、JSSP20×20、OP50**。数据、限时、选择与报告规则见[实验准则](../PROTOCOL.md)；任务选择的依据见[任务设计](../../docs/04-研究认识与构想/2026-10-07-六个组合优化任务的设计.md)。

## 任务、接口与框架

每个任务对应自己的目录。模板保存问题定义、函数签名与可修改起始代码；评价器保存固定框架或完整解核验逻辑；生成器按种子重建数据，无需实例文件。当前采用四项函数进化、两项完整求解器进化。

| task key | 规模 | 待进化函数 | 固定求解框架 |
| --- | --- | --- | --- |
| `tsp_construct` | 50节点 | `select_next_node` | 逐步构造旅行商回路 |
| `cvrp_aco` | 50客户 | `heuristics` | ACO的静态边先验 |
| `fssp_gls` | 50作业×20机器 | `get_matrix_and_jobs` | NEH、真实局部改进、扰动引导移动，200轮 |
| `graph_colouring` | 300顶点 | `solve(adjacency)` | 无固定框架，返回完整颜色数组；起始代码为DSATUR |
| `jssp_construct` | 20作业×20机器，400工序 | `solve(processing_times, machine_order)` | 无固定框架，返回完整开始时间矩阵；起始代码为GT＋MWKR |
| `op_aco` | 50节点，路程预算3 | `heuristics` | ACO的静态边先验 |

JSSP的每个作业有自己的机器顺序，与FSSP中全部作业沿同一机器顺序加工不同。`processing_times[j,k]`是作业j第k道工序的时长，`machine_order[j,k]`才是机器ID；k是**工序位置**。`solve`返回同形状的有限非负开始时间矩阵。模型可整体替换构造与搜索算法；评价器仅检查作业先后与机器互斥并计算makespan。[模板](../../benchmarks/jssp_construct/template.py)、[评价器](../../benchmarks/jssp_construct/evaluation.py)、[数据](../../benchmarks/jssp_construct/dataset.py)

图着色的 `solve` 输入完整邻接矩阵，返回每个顶点的非负整数颜色。模型可整体替换DSATUR起始代码，自行决定构造、修复、局部搜索和重启。评价器仅检查相邻顶点异色，并统计实际不同颜色数。两项完整求解器的每实例10秒CPU包含全部 `solve` 及其辅助函数计算，整实例墙钟上限30秒；其余四项保持候选函数2秒CPU、整实例20秒墙钟。

**2026-10-11接口切换：** 两项均写入 `task_eval.interface=full-solver-v1`，任务key保留；新批次不得续用旧评分函数的运行目录。已有结果保持原值，旧实验的补测与复现使用其源码快照。当前入口会拒绝用新接口重测旧配置。

OP函数在每实例开始时调用一次，返回与距离矩阵同形的边先验，由所有蚂蚁与迭代复用。外层限制访问不重复、访问后仍可返回仓库，并保证总路程不超过预算。目标为最大化收集奖励，评价返回负平均奖励。[模板](../../benchmarks/op_aco/template.py)、[评价器](../../benchmarks/op_aco/evaluation.py)、[数据](../../benchmarks/op_aco/dataset.py)

FSSP、图着色、JSSP的成绩为平均参考偏差 `100×(objective-reference)/abs(reference)`；参考分别为NEH、DSATUR构造和MWKR的Giffler–Thompson调度。负值表示超过参考。

## 命令

```bash
uv run python -m experiments.traceaad_v10_17.run_current --task jssp_construct --dry-run
uv run python -m experiments.traceaad_v10_17.launch_current --suite co6 --batch <新批次> --repeats 3 --budget 1000
```

启动入口默认仅打印计划；加 `--launch` 才启动。各基线使用同一任务注册：

```bash
uv run python -m experiments.infra.local_queue --method eoh --suite co6 --batch <批次> --slots 3 --dry-run
```

冻结后运行同规模测试：

```bash
uv run python -m experiments.infra.evaluate <运行目录> --primary
```
