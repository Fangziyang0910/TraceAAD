# 实验入口

这里的实验用于理解 LLM 在不同材料、目标与约束下怎样改进算法，并检验由这些认识形成的方法。研究问题与认识见[研究主线](../docs/04-研究认识与构想/研究主线与问题.md)。这里集中查找运行命令、状态查询和结果位置。

## 针对问题的检验

| 问题 | 实验 |
| --- | --- |
| 代码前的分析怎样影响生成？ | [格式配对](historical/format_study/README.md) |
| 目标与计算余量怎样影响结构发现？ | [固定父代提示对照](diagnosis_v1015_5/README.md) |
| 提供逐行耗时能改善修改吗？ | [耗时剖析重放](diagnosis_v1016_profile/README.md) |
| 训练进展能延续到独立实例吗？ | [前沿重测](diagnosis_v1016_frontier/README.md) |
| 初次生成的程序经少量改写后怎样变化？ | [同等开发](diagnosis_v1016_develop/README.md) |
| 落后的新设计得到开发后能否成立？ | [新设计开发搜索](traceaad_v10_18/README.md) |
| 改动在原算法上检验，能否让新计算成立？ | [改动检验搜索](traceaad_v10_19/README.md) |
| 保留当前规则、用它引导搜索，首版是否不劣于父代？ | [固定父代重放](diagnosis_v1020_replay/README.md) |
| ACO 训练成绩有多少是种子噪声？后期提升是否真实？ | [ACO 噪声审计](diagnosis_aco_noise/README.md) |
| 更正计算事实并增加 Deepen，能否把时限换成质量？ | [当前规则引导的搜索](traceaad_v10_20/README.md) |

## 搜索与独立测试

新实验默认在训练集上进化，冻结训练成绩最好的程序，再执行独立测试；不使用验证集筛选。旧五任务的验证条件可用 `--final-selection validation` 显式复现。

[六任务 AHD 实验](co6/README.md)提供 TSP、CVRP、FSSP、图着色、JSSP与OP的任务说明、函数契约、固定种子的数据生成规则和运行命令。启动计划用 `--suite co6`；通用评价用 `--primary` 只执行同规模主测试。FSSP、图着色与JSSP使用固定种子生成的训练集和同规模独立测试集；TSP/CVRP/OP沿用既有数据与跨规模条件。旧批次默认仍使用 `legacy`。

[当前规则引导的搜索](traceaad_v10_20/README.md)、[改动检验搜索](traceaad_v10_19/README.md)、[新设计开发搜索](traceaad_v10_18/README.md)、[短程改写搜索](traceaad_v10_17/README.md)、[形成路径搜索](traceaad_v10_15/README.md)、[尝试经验搜索](traceaad_v10_16/README.md)各自提供运行、选择和 held-out 命令，并共用[实验与结果实现](infra/SEARCH_FORMAT.md)。代码目录保留实现标识，文档按研究对象命名。V10.13–V10.14 的说明见[历史研究脚本](historical/README.md)，历史结果继续可视化。

模型、采样、时限与实例隔离见各入口的配置。评价使用单线程 BLAS/OpenMP，墙钟超时；装箱按实例重新执行候选程序，VRPTW 接口明确 depot 返回规则。

任务条件集中在 `benchmarks/tasks.py`。基线和 TraceAAD 的候选种子条件分别记录，详见[评价与保存说明](infra/SEARCH_FORMAT.md#评价条件与执行)。通用 held-out 命令为 `uv run python -m experiments.infra.evaluate <运行目录>`，额外条件通过 `--condition` 与 `--variant` 明确记录。

## 状态与结果

查询状态、等待完成和远端同步见[执行入口](infra/AGENT_OPERATIONS.md)。指定批次即可查询：

```bash
uv run python -m experiments.infra.batch_status --manifest <批次清单>
```

[结果概览](../docs/02-实验结果/01-搜索结果概览.md)用于查看成绩，[原始数据](../experiments_result/README.md)说明文件位置。结果存储与硬链接恢复见[存储说明](infra/RESULT_STORAGE.md)。
