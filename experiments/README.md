# 实验入口

这里的实验用于检验搜索轨迹中的反馈经验怎样改善后续决策，以及对应的算法收益与成本。研究问题与认识见[研究主线](../docs/04-研究认识与构想/研究主线与问题.md)。这里集中查找运行命令、状态查询和结果位置。旧本地模型结果已封存到[归档入口](../archive/local_models_20261011/README.md)，现用结果目录保留DeepSeek实验。

## 当前实现入口

当前仅保留 [V12.0 复提接续](traceaad_v12_0/README.md)，提供关闭接续的基础组、完整接续组与随机展示诊断组。搜索、配置和提示均位于 `traceaad/v12_0/`，执行与存储工具位于 `traceaad/common/`。

V10.15–V10.25 与旧决策模型实验代码已移除。历史研究文档和实验结果保留；复现历史批次应使用其启动时的源码快照或 Git 历史。

## 搜索与独立测试

模型、参数、任务、数据、评价与报告统一见[实验规范与共识](../docs/08-实验规范与共识/README.md)。本目录维护运行入口与状态查询；[六任务入口](co6/README.md)给出模板、评价器和命令链接。

2026-10-11起，新实验采用四项固定框架（TSP、CVRP、FSSP、OP）与两项完整求解器（图着色、JSSP）。后两项的 `solve` 可整体改写；旧评分函数批次仍按启动时的协议解释，不能用新评价器续跑或补测。

V12.0 的运行命令见[实验入口](traceaad_v12_0/README.md)，结果读取共用[实验与结果实现](infra/SEARCH_FORMAT.md)。

任务注册与统一预算在 `benchmarks/tasks.py`；任务专属的数据、配置、模板和评价器位于各自任务目录，共用数据与评价逻辑位于 `benchmarks/shared/`。通用测试命令为 `uv run python -m experiments.infra.evaluate <运行目录> --primary`。

当前仅保留上述六任务和旧版在线装箱（OBP）、VRPTW，共八个任务。CO-Bench适配器与专用任务预检入口已移除，历史结果保留。OBP、VRPTW沿用原设置。

## 状态与结果

2026-10-12起，新六任务批次使用平均函数CPU预算：四项固定框架任务平均2秒／实例，两项完整求解器平均10秒／实例。全部实例合法完成后检查平均CPU时间，单个实例超过均额不会直接淘汰；单实例墙钟安全上限统一120秒。训练、最终选择和测试共用该规则。配置记录`function_budget_scope=mean_instance`与新revision；旧批次不改写，独立测试保留其保存的评价规则。详见[评价规范](../docs/08-实验规范与共识/README.md#3-评价)。

查询状态、等待完成和远端同步见[执行入口](infra/AGENT_OPERATIONS.md)。指定批次即可查询：

```bash
uv run python -m experiments.infra.batch_status --manifest <批次清单>
```

[结果概览](../docs/02-实验结果/01-搜索结果概览.md)用于查看成绩，[原始数据](../experiments_result/README.md)说明文件位置。结果存储与硬链接恢复见[存储说明](infra/RESULT_STORAGE.md)。
