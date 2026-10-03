# 实验入口

这里的实验用于理解 LLM 在不同材料、目标与约束下怎样改进算法，并检验由这些认识形成的方法。研究问题与认识见[研究主线](../docs/04-研究认识与构想/研究主线与问题.md)。这里集中查找运行命令、状态查询和结果位置。

## 针对问题的检验

| 问题 | 实验 |
| --- | --- |
| 代码前的分析怎样影响生成？ | [格式配对](format_study/README.md) |
| 目标与计算余量怎样影响结构发现？ | [固定父代提示对照](diagnosis_v1015_5/README.md) |
| 提供逐行耗时能改善修改吗？ | [耗时剖析重放](diagnosis_v1016_profile/README.md) |
| 训练进展能延续到独立实例吗？ | [前沿重测](diagnosis_v1016_frontier/README.md) |
| 初次生成的程序经少量改写后怎样变化？ | [同等开发](diagnosis_v1016_develop/README.md) |

## 搜索、选择与测试

[形成路径搜索](traceaad_v10_15/README.md)、[尝试经验搜索](traceaad_v10_16/README.md)、[短路径搜索](traceaad_v10_14/README.md)各自提供运行、选择和 held-out 命令。代码目录保留实现标识，文档按研究对象命名。

模型、采样、时限与实例隔离见各入口的配置。评价使用单线程 BLAS/OpenMP，墙钟超时；装箱按实例重新执行候选程序，VRPTW 接口明确 depot 返回规则。

## 状态与结果

查询状态、等待完成和远端同步见[执行入口](infra/AGENT_OPERATIONS.md)。指定批次即可查询：

```bash
uv run python -m experiments.infra.batch_status --manifest <批次清单>
```

[结果概览](../docs/02-实验结果/01-搜索结果概览.md)用于查看成绩，[原始数据](../experiments_result/README.md)说明文件位置。结果存储与硬链接恢复见[存储说明](infra/RESULT_STORAGE.md)。
