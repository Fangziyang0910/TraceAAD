# 实验入口

这里的实验用于检验搜索轨迹中的反馈经验怎样改善后续决策，以及对应的算法收益与成本。研究问题与认识见[研究主线](../docs/04-研究认识与构想/研究主线与问题.md)。这里集中查找运行命令、状态查询和结果位置。旧本地模型结果已封存到[归档入口](../archive/local_models_20261011/README.md)，现用结果目录保留DeepSeek实验。

## 历史检验与实现入口

下表列出研究问题及对应实现。

| 问题 | 实验 |
| --- | --- |
| 重复提出同一计算时，从该计算已有的最好实现出发，能否提高重新实现的质量？ | [复提接续（V12.0）](traceaad_v12_0/README.md) |
| 真实程序轨迹能否帮助继承、替换和迁移计算？ | [V10.25 轨迹条件化搜索](traceaad_v10_25/README.md)；[原 V10.17 重跑](traceaad_v10_17/README.md#2026-10-10原机制在当前协议下重跑)。 |
| 落后的新设计得到开发后能否成立？ | [新设计开发搜索](traceaad_v10_18/README.md) |
| 改动在原算法上检验，能否让新计算成立？ | [改动检验搜索](traceaad_v10_19/README.md) |
| 更正计算事实并增加 Deepen，能否把时限换成质量？ | [当前规则引导的搜索](traceaad_v10_20/README.md) |
| 只交代问题与一条效率规则，能否减少超时、保持成绩？ | [精简提示与效率规则](traceaad_v10_21/README.md) |
| Explore 从精确输出出发分析，能否引入新的决策结构？ | [从精确输出出发的探索](traceaad_v10_22/README.md) |
| 作用问题、有效开发证据与共同开发机会能否提高固定预算的强程序产出？ | [有限开发搜索（V10.24）](traceaad_v10_24/README.md) |
| 按改动种类定义步骤、给各步相配的材料，搜索能否持续引入新计算？ | [按改动种类定义的步骤](traceaad_v10_23/README.md)；[行为检查](../docs/03-现象与检验/2026-10-09-V10.23行为检查.md) |
| 代码与开发记录能否帮助选择下一步算子？ | [决策模型的数据与训练](decision_model/README.md)：四台机器的独立环境、数据格式、LoRA 训练与恢复命令；监督信号来自同状态对照。 |

## 搜索与独立测试

模型、参数、任务、数据、评价与报告统一见[实验规范与共识](../docs/08-实验规范与共识/README.md)。本目录维护运行入口与状态查询；[六任务入口](co6/README.md)给出模板、评价器和命令链接。

2026-10-11起，新实验采用四项固定框架（TSP、CVRP、FSSP、OP）与两项完整求解器（图着色、JSSP）。后两项的 `solve` 可整体改写；旧评分函数批次仍按启动时的协议解释，不能用新评价器续跑或补测。

[按改动种类定义的步骤](traceaad_v10_23/README.md)、[从精确输出出发的探索](traceaad_v10_22/README.md)、[精简提示与效率规则](traceaad_v10_21/README.md)、[当前规则引导的搜索](traceaad_v10_20/README.md)、[改动检验搜索](traceaad_v10_19/README.md)、[新设计开发搜索](traceaad_v10_18/README.md)、[短程改写搜索](traceaad_v10_17/README.md)、[形成路径搜索](traceaad_v10_15/README.md)、[尝试经验搜索](traceaad_v10_16/README.md)各自提供运行命令，并共用[实验与结果实现](infra/SEARCH_FORMAT.md)。代码目录保留实现标识，文档按研究对象命名。

任务注册与统一预算在 `benchmarks/tasks.py`；任务专属的数据、配置、模板和评价器位于各自任务目录，共用数据与评价逻辑位于 `benchmarks/shared/`。通用测试命令为 `uv run python -m experiments.infra.evaluate <运行目录> --primary`。

当前仅保留上述六任务和旧版在线装箱（OBP）、VRPTW，共八个任务。CO-Bench适配器与专用任务预检入口已移除，历史结果保留。OBP、VRPTW沿用原设置。

## 状态与结果

2026-10-12起，新六任务批次使用平均函数CPU预算：四项固定框架任务平均2秒／实例，两项完整求解器平均10秒／实例。全部实例合法完成后检查平均CPU时间，单个实例超过均额不会直接淘汰；单实例墙钟安全上限统一120秒。训练、最终选择和测试共用该规则。配置记录`function_budget_scope=mean_instance`与新revision；旧批次不改写，独立测试保留其保存的评价规则。详见[评价规范](../docs/08-实验规范与共识/README.md#3-评价)。

查询状态、等待完成和远端同步见[执行入口](infra/AGENT_OPERATIONS.md)。指定批次即可查询：

```bash
uv run python -m experiments.infra.batch_status --manifest <批次清单>
```

[结果概览](../docs/02-实验结果/01-搜索结果概览.md)用于查看成绩，[原始数据](../experiments_result/README.md)说明文件位置。结果存储与硬链接恢复见[存储说明](infra/RESULT_STORAGE.md)。
