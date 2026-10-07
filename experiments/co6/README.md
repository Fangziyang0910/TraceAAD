# 六个组合优化任务：实验入口

本套主实验关注同规模独立测试。保留现有 TSP50 构造式和 CVRP50–ACO；新增 FSSP、MDMKP、图着色和集合覆盖。四项均固定外层求解器，只进化一个关键函数。[选择依据与研究目的](../../docs/04-研究认识与构想/2026-10-07-六个组合优化任务的设计.md)

## 代码与数据目录

一个任务对应一个目录。TSP、CVRP 沿用既有目录；四项新增任务使用同样的组织方式。

| 任务 | 问题说明与待进化函数 | 求解框架与评价器 | 数据规则与生成器 |
| --- | --- | --- | --- |
| FSSP | [template.py](../../benchmarks/fssp_gls/template.py) | [evaluation.py](../../benchmarks/fssp_gls/evaluation.py) | [dataset.py](../../benchmarks/fssp_gls/dataset.py) |
| MDMKP | [template.py](../../benchmarks/mdmkp_search/template.py) | [evaluation.py](../../benchmarks/mdmkp_search/evaluation.py) | [dataset.py](../../benchmarks/mdmkp_search/dataset.py) |
| 图着色 | [template.py](../../benchmarks/graph_colouring/template.py) | [evaluation.py](../../benchmarks/graph_colouring/evaluation.py) | [dataset.py](../../benchmarks/graph_colouring/dataset.py) |
| 集合覆盖 | [template.py](../../benchmarks/set_cover_construct/template.py) | [evaluation.py](../../benchmarks/set_cover_construct/evaluation.py) | [dataset.py](../../benchmarks/set_cover_construct/dataset.py) |

每项任务只包含模板、评价器和数据生成代码。`dataset.py` 定义规模、数量、生成分布与参考值计算。`benchmarks/tasks.py` 注册实验条件；共用代码只负责执行计分，以及固定种子和数据身份。

## 训练与独立测试数据

本次实验只使用训练集与独立测试集。流程为 **训练集上进化 → 按训练成绩选出一个最终程序 → 冻结程序并在独立测试集报告成绩**。所有方法采用相同规则，测试成绩不用于选择或继续修改程序。训练同分时取最早产生的有效程序；测试失败只记录失败，不替换为另一个程序。

TraceAAD 实验入口默认 `--final-selection training`，不执行验证集评价。最终程序及选择依据会落盘。旧五任务的历史验证条件仍可通过显式 `--final-selection validation` 复现，不用于本次六任务实验。

| task key | 固定规模 | 训练 | 同规模主测试 |
| --- | --- | ---: | ---: |
| `tsp_construct` | 50 节点 | 16 | 16 |
| `cvrp_aco` | 50 客户 | 10 | 64 |
| `fssp_gls` | 50 作业、20 机器 | 16 | 100 |
| `mdmkp_search` | 100 物品、10 上界、5 下界 | 18 | 108 |
| `graph_colouring` | 300 顶点、边密度约 0.5 | 16 | 100 |
| `set_cover_construct` | 200 元素、2000 集合、密度 0.02 | 16 | 100 |

主数据采用本项目明确定义的生成分布，训练与测试分别从独立种子流生成。四项新增任务在创建评价器时按固定种子生成训练或测试实例，在内存中保留。TSP 与 CVRP 的既有数据数量、随机种子和求解预算沿用原配置。

生成规则、规模、数量和参考值计算集中在各任务的 `dataset.py`。[选择依据与分布说明](../../docs/04-研究认识与构想/2026-10-07-六个组合优化任务的设计.md)

MDMKP 训练、测试分别有 **9、54 个独立基础实例**；18、108 是包含收益变体的实例数。分析抽样不应把两个变体视为独立基础实例。这组数量是起步配置，不是统计功效保证；主实验还需要独立重复算法搜索。

无需准备数据文件。创建评价器时，只生成指定划分的数据与参考值；它们在所有候选评价之间保持不变。初始化不计入候选评价时限。

总种子为 `20261007`。每个基础实例的随机流由 `[总种子, 任务流, 划分流, 基础实例编号]` 定义；任务流依次为 FSSP=0、MDMKP=1、图着色=2、集合覆盖=3，训练流为0、测试流为2。它们独立于算法搜索种子和全局 NumPy 随机状态。MDMKP 的两个收益变体共享基础实例，始终位于同一划分。

实例的内容哈希、种子和参考值参与评价身份计算；运行复现依靠固定生成代码和仓库的 `uv.lock` 依赖。数据协议为 `ahd-six-tasks-v4-seeded-runtime`，输入与参考值已逐项核对，与移除文件前的474个实例一致。

## 修改与评价范围

完整问题说明、输入输出与返回值消费方式保存在各任务的 `template.py`；固定求解代码在 `evaluation.py`。四项新增任务只进化模板中的目标函数：

| 任务 | 目标函数 | 固定外层 |
| --- | --- | --- |
| FSSP | `get_matrix_and_jobs` | NEH 初始化、真实局部改进、扰动引导移动 |
| MDMKP | `score_moves` | 可行加入／删除／交换、tabu、保留最好解 |
| 图着色 | `score_coloring_moves` | 构造、减色与冲突修复 |
| 集合覆盖 | `score_sets` | 构造、冗余删除、移除后补全 |

ID 均从0开始。每个实例重新执行候选程序，模块状态可在同一实例内保留；传给函数的数组均为副本。质量根据最终有效解计算。这些固定框架由本项目实现，所有搜索方法和基线在同一框架上重新运行。

## 分数与时间预算

每项返回实例参考偏差的均值，单位为百分比，**越小越好**：

- FSSP、图着色、集合覆盖：`100 × (objective − reference) / |reference|`。
- MDMKP：`100 × (reference − profit) / |reference|`。

生成 FSSP 的参考是 NEH 可行解；图着色是确定性 DSATUR 构造；集合覆盖是 gain/cost 构造加冗余删除。它们是可行上界。生成 MDMKP 的参考是连续 LP 松弛上界，不是整数最优解。

因此这里称为“参考偏差”，不称为最优性 gap。负值不截断。参考值不传给候选函数。比较方法时使用相同数据与分数定义。

训练整套评价默认限时 60 秒。测试按实例数比例放大总时限，保持相同的每实例平均时间预算；外层迭代次数在各阶段一致。主测试限时分别为 375、360、375、375 秒。候选超时、报错或违反返回契约时，本次评价失败。这是可运行的初始配置，正式实验前仍要检查算法开发空间和评价代价。

## 实验命令

检查单任务配置，不调用模型：

```bash
uv run python -m experiments.traceaad_v10_20.run --task fssp_gls --dry-run
```

运行单任务：

```bash
uv run python -m experiments.traceaad_v10_20.run --task fssp_gls --experiment traceaad_v10_20 --run-name fssp_rep1 --repeat 1 --seed 0 --budget 1000
```

生成六任务计划（默认仅打印计划；加 `--launch` 才启动）：

```bash
uv run python -m experiments.traceaad_v10_20.launch_local --suite co6 --experiment traceaad_v10_20 --batch co6_example --repeats 3 --budget 1000
```

基线使用相同注册任务：

```bash
uv run python -m experiments.launch --method eoh --suite co6 --batch co6_example --dry-run
```

搜索结束后冻结训练成绩最好的程序，再运行同规模主测试：

```bash
uv run python -m experiments.infra.evaluate experiments_result/traceaad_v10_20/fssp_gls/fssp_rep1 --condition traceaad --primary
```

TSP/CVRP 的跨规模测试仍可通过 `--units` 指定。原五任务批次使用 `--suite legacy`；既有默认保持该组，以免旧批次意外增加任务。


## 当前批次

四个新增任务的 V10.20 批次已于2026年10月7日在本机启动：`20261007_local_v1020_new4_seeded`。每项三路，种子0、1、2，每路1000次候选尝试。模型为 server3 两个端点的 `qwen3.8-27b-awq`，各分配六路；评价使用本机CPU。启动源码为 `bd4078dd`，完整条件与源码哈希保存在批次清单。

[实时训练页面](http://127.0.0.1:8765/#b=traceaad_v10_20)统一展示已有任务和新增任务，共9项、27路；其中新增四项各三路。批次清单为 `experiments_result/traceaad_v10_20/batch_20261007_local_v1020_new4_seeded.json`。查询当前状态：

```bash
uv run python -m experiments.infra.batch_status --manifest experiments_result/traceaad_v10_20/batch_20261007_local_v1020_new4_seeded.json
```

主测试在每路搜索结束、训练最佳程序冻结后执行。本次启动检查未评价独立测试集。
