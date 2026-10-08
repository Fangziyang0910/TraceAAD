# 六个组合优化任务：实验入口

主实验采用 **TSP50、CVRP50、FSSP50×20、图着色300、JSSP20×20、OP50**。数据、限时、选择与报告规则见[实验准则](../PROTOCOL.md)；任务选择的依据见[任务设计](../../docs/04-研究认识与构想/2026-10-07-六个组合优化任务的设计.md)。

## 任务、接口与框架

每个任务对应自己的目录。模板保存问题定义、函数签名与返回值使用方式；评价器保存固定外层求解器；生成器按种子重建数据，无需实例文件。

| task key | 规模 | 待进化函数 | 固定求解框架 |
| --- | --- | --- | --- |
| `tsp_construct` | 50节点 | `select_next_node` | 逐步构造旅行商回路 |
| `cvrp_aco` | 50客户 | `heuristics` | ACO的静态边先验 |
| `fssp_gls` | 50作业×20机器 | `get_matrix_and_jobs` | NEH、真实局部改进、扰动引导移动，200轮 |
| `graph_colouring` | 300顶点 | `score_coloring_moves` | 构造、减色与冲突修复 |
| `jssp_construct` | 20作业×20机器，400工序 | `score_operations` | Giffler–Thompson构造中的冲突集合排序 |
| `op_aco` | 50节点，路程预算3 | `heuristics` | ACO的静态边先验 |

JSSP的每个作业有自己的机器顺序，与FSSP中全部作业沿同一机器顺序加工不同。`processing_times[j,k]`是作业j第k道工序的时长，`machine_order[j,k]`才是机器ID。候选行 `(j,k)` 中的k是**工序位置**。函数返回与实际候选集合等长的有限评分向量，分数越大越优先。它不返回开始时间或机器指派。外层负责作业先后约束、机器互斥与最终可行性核验。[模板](../../benchmarks/jssp_construct/template.py)、[评价器](../../benchmarks/jssp_construct/evaluation.py)、[数据](../../benchmarks/jssp_construct/dataset.py)

OP函数在每实例开始时调用一次，返回与距离矩阵同形的边先验，由所有蚂蚁与迭代复用。外层限制访问不重复、访问后仍可返回仓库，并保证总路程不超过预算。目标为最大化收集奖励，评价返回负平均奖励。[模板](../../benchmarks/op_aco/template.py)、[评价器](../../benchmarks/op_aco/evaluation.py)、[数据](../../benchmarks/op_aco/dataset.py)

FSSP、图着色、JSSP的成绩为平均参考偏差 `100×(objective-reference)/abs(reference)`；参考分别为NEH、DSATUR构造和MWKR的Giffler–Thompson调度。负值表示超过参考。

## 命令

```bash
uv run python -m experiments.traceaad_v10_20.run --task jssp_construct --dry-run
uv run python -m experiments.traceaad_v10_20.launch_local --suite co6 --experiment traceaad_v10_20 --batch <批次> --repeats 3 --budget 1000
```

启动入口默认仅打印计划；加 `--launch` 才启动。各基线使用同一任务注册：

```bash
uv run python -m experiments.launch --method eoh --suite co6 --batch <批次> --dry-run
```

冻结后运行同规模测试：

```bash
uv run python -m experiments.infra.evaluate <运行目录> --primary
```
