# 六个组合优化任务：实验入口

主实验采用 **TSP50、CVRP50、FSSP50×20、图着色300、JSSP20×20、OP50**。选取依据是相关AAD工作、实际求解难度与算法开发空间；不以问题类型尽量不同作为目标。[任务设计与预检证据](../../docs/04-研究认识与构想/2026-10-07-六个组合优化任务的设计.md)

## 任务、接口与框架

每个任务对应自己的目录。模板保存问题定义、函数签名与返回值消费方式；评价器保存固定外层求解器；生成器按种子重建数据，无需实例文件。

| task key | 规模 | 待进化函数 | 固定求解框架 |
| --- | --- | --- | --- |
| `tsp_construct` | 50节点 | `select_next_node` | 逐步构造旅行商回路 |
| `cvrp_aco` | 50客户 | `heuristics` | ACO的静态边先验 |
| `fssp_gls` | 50作业×20机器 | `get_matrix_and_jobs` | NEH、真实局部改进、扰动引导移动 |
| `graph_colouring` | 300顶点 | `score_coloring_moves` | 构造、减色与冲突修复 |
| `jssp_construct` | 20作业×20机器，400工序 | `score_operations` | Giffler–Thompson构造中的冲突集合排序 |
| `op_aco` | 50节点，路程预算3 | `heuristics` | ACO的静态边先验 |

JSSP的每个作业有自己的机器顺序，与FSSP中全部作业沿同一机器顺序加工不同。`processing_times[j,k]`是作业j第k道工序的时长，`machine_order[j,k]`才是机器ID。候选行 `(j,k)` 中的k是**工序位置**。函数返回与实际候选集合等长的有限评分向量，分数越大越优先。它不返回开始时间或机器指派。外层负责作业先后约束、机器互斥与最终可行性核验。[模板](../../benchmarks/jssp_construct/template.py)、[评价器](../../benchmarks/jssp_construct/evaluation.py)、[数据](../../benchmarks/jssp_construct/dataset.py)

OP函数在每实例开始时调用一次，返回与距离矩阵同形的边先验，由所有蚂蚁与迭代复用。外层限制访问不重复、访问后仍可返回仓库，并保证总路程不超过预算。目标为最大化收集奖励，评价返回负平均奖励。[模板](../../benchmarks/op_aco/template.py)、[评价器](../../benchmarks/op_aco/evaluation.py)、[数据](../../benchmarks/op_aco/dataset.py)

## 数据与最终选择

流程为 **训练集进化 → 冻结训练成绩最好的一个程序 → 同规模独立测试**，不设验证集。默认 `--final-selection training`；`validation` 只用于复现历史条件。测试不参与选择、继续修改或替换失败程序。

| task key | 训练数量 | 同规模主测试数量 | 训练／主测试限时（整套，秒） |
| --- | ---: | ---: | ---: |
| `tsp_construct` | 16 | 16 | 30／3000 |
| `cvrp_aco` | 10 | 64 | 120／3600 |
| `fssp_gls` | 16 | 100 | 60／375 |
| `graph_colouring` | 16 | 100 | 60／375 |
| `jssp_construct` | 16 | 100 | 60／375 |
| `op_aco` | 5 | 64 | 60／3600 |

JSSP与FSSP、图着色相同，在评价器初始化时生成指定划分，内存复用，不计入候选时限。JSSP总种子20261007、任务流4、训练流0、测试流2，每个作业独立随机排列20台机器，时长独立均匀整数1–99。这是本项目明确规定的Taillard-style分布，没有声称重建原Taillard实例或原随机生成器。

OP沿用ReEvo／DeepACO的生成规则与既有数量：训练种子1234，测试种子4567，节点均匀分布于单位正方形，奖励按到仓库距离定义，OP50预算3。ACO为20只蚂蚁×50次迭代。数据种子独立于算法搜索种子。原TSP、CVRP配置与跨规模条件继续保留。

FSSP、图着色、JSSP的成绩为平均参考偏差 `100×(objective-reference)/abs(reference)`，越小越好；参考分别为NEH、DSATUR构造和MWKR的Giffler–Thompson调度，均为可行解，不是已证明最优值。负值表示超过参考。TSP/CVRP返回平均长度；OP返回负平均奖励。不同任务的绝对数值不能直接横向比较。

## 替换的依据

此前退出主套件的两项任务及其源码和原始试验档案已删除。任务取舍的认识集中在设计文档，当前入口只保留六项主任务。

替代预检只用训练数据。JSSP20×20完整16个训练实例，单线程CP-SAT每例3.75秒均得到有效解，均未证明最优，上下界仍有差距。OP50抽查2例，单线程HiGHS每例12秒未证明最优；这只是该整数规划模型的限时结果，不代表所有专用求解器。旧VRPTW50两例约0.03、0.06秒证明最优，因此本轮不直接用回这一配置。[预检结果](../../experiments_result/task_pilot/20261007_task_replacement/)

OP是[ReEvo的ACO主实验任务](https://arxiv.org/html/2402.01145#S5.SS2)。静态JSSP是[CO-Bench的任务与方法比较对象](https://arxiv.org/html/2504.04310v3#A1.SS19)。本项目采用固定关键函数接口，各AAD方法在同一框架重新运行；不把CO-Bench原生完整solve的成绩当作直接可比结果。

## 命令

```bash
uv run python -m experiments.traceaad_v10_20.run --task jssp_construct --dry-run
uv run python -m experiments.traceaad_v10_20.run --task op_aco --dry-run
uv run python -m experiments.traceaad_v10_20.launch_local --suite co6 --experiment traceaad_v10_20 --batch co6_example --repeats 3 --budget 1000
```

上面的启动入口默认仅打印计划；加 `--launch` 才启动。各基线使用同一任务注册：

```bash
uv run python -m experiments.launch --method eoh --suite co6 --batch co6_example --dry-run
```

冻结后用统一入口运行同规模主测试：

```bash
uv run python -m experiments.infra.evaluate experiments_result/traceaad_v10_20/jssp_construct/jssp_rep1 --condition traceaad --primary
```

## 当前实验

训练监控统一使用 [V10.20页面](http://127.0.0.1:8765/#b=traceaad_v10_20)。FSSP、图着色各三路继续运行；JSSP三路使用1000次候选预算、server3双端点和搜索种子0、1、2。OP复用`20261006_local_v1020`已完成的原三路搜索，不重新进化。

FSSP／图着色清单：`experiments_result/traceaad_v10_20/batch_20261007_local_v1020_new4_seeded.json`，现仅包含这六路。

JSSP清单：`experiments_result/traceaad_v10_20/batch_20261007_local_v1020_replacement.json`，现仅包含JSSP三路，`reused_runs`另外记录原OP的训练最好节点、哈希与成绩。只有测试程序与训练最好程序相同时才能复用测试记录，否则评价对应程序。已停止的九路和相关日志、清单条目已删除，不再出现在训练页。
