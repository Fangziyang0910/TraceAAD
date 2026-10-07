# 六任务 AHD 实验：四项新增任务

本套主实验关注同规模独立测试。保留现有 TSP50 构造式和 CVRP50–ACO；这里提供 FSSP、MDMKP、图着色和集合覆盖。四项均固定外层求解器，只进化一个关键函数。[选择依据与研究目的](../../docs/04-研究认识与构想/2026-10-07-六个组合优化任务的设计.md)

## 主数据与补充数据

| task key | 固定规模 | 训练 | 验证 | 同规模主测试 | 标准补充测试 |
| --- | --- | ---: | ---: | ---: | ---: |
| `fssp_gls` | 50 作业、20 机器 | 16 | 32 | 100 | 10 |
| `mdmkp_search` | 100 物品、10 上界、5 下界 | 18 | 36 | 108 | 30 |
| `graph_colouring` | 300 顶点、边密度约 0.5 | 16 | 32 | 100 | 10 |
| `set_cover_construct` | 200 元素、2000 集合、密度 0.02 | 16 | 32 | 100 | 10 |

主数据采用本项目明确定义的生成分布，训练、验证、测试分别从独立种子流生成。它们不是 OR-Library 实例的复刻。标准实例只用于补充测试，不参与搜索或最终程序选择，不并入主测试成绩。TSP 与 CVRP 的既有数据数量、随机种子和求解预算沿用原配置。

主数据生成规则见 [generated.py](generated.py) 和 [data/manifest.json](data/manifest.json)：

- FSSP：每个加工时间独立采样为 1–99 的均匀整数。作业访问所有机器，机器顺序一致，所有机器使用同一作业排列。
- 图着色：`G(300,0.5)`，每条无向边独立采样，无自环。
- 集合覆盖：每个元素独立、均匀选择 40 个不同集合覆盖它；集合成本独立采样为 1–100 的均匀整数。
- MDMKP：三档紧度 `α=0.25/0.50/0.75` 等权。先均匀选择 `α×100` 个物品作为可行见证；每行系数采样为 1–1000 的独立均匀整数，条件是见证满足该行约束，界为 `floor(α×行和)`。正收益采样为 1–1000，混合收益采样为 −500–1000。收益不参与见证或约束生成。每个基础实例产生两个收益变体，始终位于同一划分。

MDMKP 训练、验证、测试分别有 **9、18、54 个独立基础实例**；18、36、108 是包含收益变体的实例数。分析抽样不应把两个变体视为独立基础实例。这组数量是起步配置，不是统计功效保证；主实验还需要独立重复算法搜索。

标准补充集来自 CO-Bench 保存的 Taillard `tai50_20`、MDMKP `mdmkp_ct4` 的 q=5 变体、图着色 `gcol21–gcol30`、集合覆盖 `scp51–scp510`。其来源、配置文件与原始文件哈希均记录在 manifest。主数据生成分布与标准数据的分布一致性没有被假定。

部分标准实例已在任务开发期用于简单规则检查。因此，标准成绩作为补充基准；严格独立的主要评价使用新生成的主测试集。

数据已压缩保存为可移植 NPZ，运行无需访问外部数据目录或网络。加载时核对 SHA256；manifest 记录尺寸、内容身份、基础实例组、参考值类型、生成种子及准备环境。复现准备命令：

```bash
uv run python -m benchmarks.ahd_suite.prepare_data --source /home/fang/code/LLM4AD/data/CO-Bench
```

## 函数契约与外层求解器

完整可执行模板见 [templates.py](templates.py)，评价器的任务说明会给出实际数据数量、调用方式、预算和分数定义，供 TraceAAD 与基线读取。节点、作业、物品和集合编号均从 0 开始。每个实例重新执行候选程序，模块状态可在同一实例的多次调用之间保留，但不能跨实例保留。传给函数的数组均为副本。

### FSSP

```python
get_matrix_and_jobs(current_sequence, processing_times, n_machines, n_jobs)
# -> (perturbed_times, job_ids)
```

时间矩阵是 **作业×机器**，排列包含作业 ID。返回同形状、有限、非负的扰动矩阵及 1–5 个不同的作业 ID；这些是矩阵行号，不是当前排列中的位置。

外层从 NEH 开始。每轮先在真实加工时间下进行最多 3 次最优改进的交换／插入，再调用一次目标函数，在扰动矩阵上寻找涉及指定作业的最优改进邻居。默认 10 轮。全过程保留真实 makespan 最好的有效排列。扰动矩阵只改变搜索的评价，不改变真实加工时间。

### MDMKP

```python
score_moves(profits, upper_coefficients, upper_limits,
            lower_coefficients, lower_limits, selected, moves)
# -> shape (K,) 的有限分数
```

上界为 `A @ x <= b`，下界为 `G @ x >= d`，`x` 是二进制选择。`moves` 每行是 `(remove_item, add_item)`，`−1` 表示没有该项。候选包含加入、删除和一出一进交换，已通过全部可行性检查；分数最大者被执行。

默认进行 32 步，外层有 5 步物品 tabu。允许执行收益下降的移动，但最终返回遇到的最好可行解。生成实例从其可行见证开始；标准实例通过离线零目标可行性求解获得初解，不优化收益。

### 图着色

```python
score_coloring_moves(adjacency, colors, moves, phase)
# -> shape (K,) 的有限分数
```

邻接矩阵为对称布尔矩阵，颜色从 0 开始，`−1` 表示尚未着色；每个候选是 `(vertex_id, target_color)`。

`construct` 阶段，外层为每个未着色顶点提出其最小可用颜色，函数决定先处理哪个顶点。`repair` 阶段，外层尝试删除最高颜色类，函数选择冲突顶点的重着色移动。默认每次修复 100 步，最多 4 次减色，反向移动 tabu 为 7 步。中间允许冲突，只有全部冲突消失才接受减色；失败时返回此前的有效着色。

### 加权集合覆盖

```python
score_sets(costs, coverage, selected, uncovered)
# -> shape (n_sets,) 的有限分数
```

覆盖矩阵是 **元素×集合**。函数返回所有集合的分数，外层屏蔽已选集合与没有新增覆盖的集合，选择最高分，直至所有元素被覆盖。之后按成本从高到低删除冗余集合，并尝试最多 16 次移除和重新补全。补全过程调用同一函数，只有成本严格降低才接受新解。最终解必须覆盖全部元素。

这些是本项目实现的固定框架，不是相关论文实现的逐字复现。比较搜索方法时应在本框架上重新运行所有基线。

## 分数与时间预算

每项返回实例参考偏差的均值，单位为百分比，**越小越好**：

- FSSP、图着色、集合覆盖：`100 × (objective − reference) / |reference|`。
- MDMKP：`100 × (reference − profit) / |reference|`。

生成 FSSP 的参考是 NEH 可行解；图着色是确定性 DSATUR 构造；集合覆盖是 gain/cost 构造加冗余删除。它们是可行上界。生成 MDMKP 的参考是连续 LP 松弛上界，不是整数最优解。标准 FSSP 使用原始文件的已发表上界，其余标准任务使用 CO-Bench 发布的参考值。

因此这里称为“参考偏差”，不称为最优性 gap。负值不截断。参考值不传给候选函数。主数据与标准数据使用不同参考类型，分别报告，比较方法时使用相同数据与分数定义。

训练整套评价默认限时 60 秒。验证和测试按实例数比例放大总时限，保持相同的每实例平均时间预算；外层迭代次数在各阶段一致。主测试限时分别为 375、360、375、375 秒。候选超时、报错或违反返回契约时，本次评价失败。这是可运行的初始配置，正式实验前仍要检查算法开发空间和评价代价。

## 实验命令

检查单任务配置，不调用模型：

```bash
uv run python -m experiments.traceaad_v10_20.run --task fssp_gls --dry-run
```

运行单任务：

```bash
uv run python -m experiments.traceaad_v10_20.run --task fssp_gls --experiment traceaad_v10_20_co6 --run-name fssp_rep1 --repeat 1 --seed 0 --budget 1000
```

生成六任务计划（默认仅打印计划；加 `--launch` 才启动）：

```bash
uv run python -m experiments.traceaad_v10_20.launch_local --suite co6 --experiment traceaad_v10_20_co6 --batch co6_example --repeats 3 --budget 1000
```

基线使用相同注册任务：

```bash
uv run python -m experiments.launch --method eoh --suite co6 --batch co6_example --dry-run
```

搜索结束并完成验证集选择后，仅运行同规模主测试：

```bash
uv run python -m experiments.infra.evaluate experiments_result/traceaad_v10_20_co6/fssp_gls/fssp_rep1 --condition traceaad --primary
```

标准补充测试显式指定 `--units standard`。TSP/CVRP 的跨规模测试仍可通过 `--units` 指定。原五任务批次使用 `--suite legacy`；既有默认保持该组，以免旧批次意外增加任务。
