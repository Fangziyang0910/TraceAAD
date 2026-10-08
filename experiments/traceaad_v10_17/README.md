# TraceAAD V10.17：科研实现

机制见[提出后的短程改写](../../docs/01-搜索方法/提出后的短程改写.md)。父代按训练质量乘尝试经验选择，Refine / Explore / Crossover 默认份额为 45:30:25。新的有效 Explore 程序中，按探索编号抽取 1/8，继续至多三步 Refine，每步从该探索达到的最好程序出发。失败的新程序至多修复一次，所有生成都计入预算。最终在独立选择集评价前五名，失败时按训练排名替补。

## 当前实现条件

2026-10-06 整理版标记为 `research-simple-20261006`，结果目录仍为 `experiments_result/traceaad_v10_17/`。运行名称与 revision 区分具体实验；原正式批次对应原解析与恢复规则。

整理版有两处行为变化：解析器提取并校验模型提交的最终程序，**不再从模板补入导入、常量或辅助函数**；完整尝试结束时统一保存，**中断时允许重做尚未提交的尝试**。

## 实现与记录

- `traceaad/v10_17/traceaad.py`：Explore 的抽取、短程改写和关闭规则。
- `traceaad/v10_17/config.py`、`prompts.py`：本版本的参数和提示差异。
- `traceaad/common/`：生成、一次 Repair、程序事实、评价、实测历史与最终选择，供 V10.15–V10.19 共用。
- `experiments/infra/search_*.py`：共用实验入口；版本命令继续可用。
- `experiments/infra/diagnose_search.py`：独立生成诊断统计，运行入口在搜索返回后调用它。

一条 `candidate` 记录保存本次 `attempt`、新产生的 `program`（重复或未交付时为 `null`）、逐种子 `evaluations` 与恢复 `state`。程序正文只在程序记录中保存；模型原始回复和实际采样信息保留在尝试的 `calls` 中。修复完成的结果仍归属原尝试的经验。

恢复保存 RNG、当前探索和待修复进度。完整记录提交后才推进恢复点；未完整提交的尾部在下一次写入时舍弃。源码文件变化不再阻止续跑，但配置和固定评价数据仍须一致。最终选择只计算并保存一次，摘要直接读取其结果。

## 运行

```bash
uv run python -m experiments.traceaad_v10_17.run --task tsp_construct --run-name trial_1 --budget 1000 --dry-run
uv run python -m experiments.traceaad_v10_17.run --task tsp_construct --run-name trial_1 --budget 1000
uv run python -m experiments.traceaad_v10_17.heldout --run-dir experiments_result/traceaad_v10_17/tsp_construct/trial_1
uv run python -m experiments.traceaad_v10_17.diagnose --run-dir experiments_result/traceaad_v10_17/tsp_construct/trial_1
```

`launch_server3.py --batch <name> [--launch]` 直接构造五个任务、三次重复的 15 路计划。训练、选择与测试的数据和时限沿用原 V10.17。参数默认值可直接修改，动作及其概率集中在 `Config.operators` 中。模块职责见[共用实现](../../traceaad/common/README.md)，结果格式见[实验与结果](../infra/SEARCH_FORMAT.md)。
