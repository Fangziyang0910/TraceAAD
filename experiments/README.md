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
| Explore 反复提出失败方向，是因为看不到失败吗？ | [Explore 上下文重放](diagnosis_explore_context/README.md) |
| 起点提示能否让不同的决策结构进入搜索？ | [起点提示重放](diagnosis_init_prompt/README.md) |
| 更正计算事实并增加 Deepen，能否把时限换成质量？ | [当前规则引导的搜索](traceaad_v10_20/README.md) |
| 只交代问题与一条效率规则，能否减少超时、保持成绩？ | [精简提示与效率规则](traceaad_v10_21/README.md) |
| Explore 从精确输出出发分析，能否引入新的决策结构？ | [从精确输出出发的探索](traceaad_v10_22/README.md) |
| 按改动种类定义步骤、给各步相配的材料，搜索能否持续引入新计算？ | [按改动种类定义的步骤](traceaad_v10_23/README.md)；[行为检查](../docs/03-现象与检验/2026-10-09-V10.23行为检查.md) |
| 代码与开发记录能否帮助选择下一步算子？ | [决策模型的数据与训练](decision_model/README.md)：四台机器的独立环境、数据格式、LoRA 训练与恢复命令；监督信号来自同状态对照。 |

## 搜索与独立测试

所有实验遵守[实验准则](PROTOCOL.md)：六个任务、训练16例与同规模测试50例、每实例10秒、每方法每任务3路、按训练成绩冻结最终程序后测试。[六任务入口](co6/README.md)给出任务接口与命令。

[按改动种类定义的步骤](traceaad_v10_23/README.md)、[从精确输出出发的探索](traceaad_v10_22/README.md)、[精简提示与效率规则](traceaad_v10_21/README.md)、[当前规则引导的搜索](traceaad_v10_20/README.md)、[改动检验搜索](traceaad_v10_19/README.md)、[新设计开发搜索](traceaad_v10_18/README.md)、[短程改写搜索](traceaad_v10_17/README.md)、[形成路径搜索](traceaad_v10_15/README.md)、[尝试经验搜索](traceaad_v10_16/README.md)各自提供运行命令，并共用[实验与结果实现](infra/SEARCH_FORMAT.md)。代码目录保留实现标识，文档按研究对象命名。

任务条件集中在 `benchmarks/tasks.py`。通用测试命令为 `uv run python -m experiments.infra.evaluate <运行目录> --primary`。

## 状态与结果

查询状态、等待完成和远端同步见[执行入口](infra/AGENT_OPERATIONS.md)。指定批次即可查询：

```bash
uv run python -m experiments.infra.batch_status --manifest <批次清单>
```

[结果概览](../docs/02-实验结果/01-搜索结果概览.md)用于查看成绩，[原始数据](../experiments_result/README.md)说明文件位置。结果存储与硬链接恢复见[存储说明](infra/RESULT_STORAGE.md)。
