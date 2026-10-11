# TraceAAD V10.17：科研实现

本页维护探索后的有限续写及其结果记录的运行与复现入口。机制解释见[轨迹用途索引](../../docs/01-搜索方法/搜索轨迹的记录与使用.md)，启动与实现检查不代替算法收益。旧本地模型批次已归档，历史状态按日期读取；新实验遵守[当前规范](../../docs/08-实验规范与共识/README.md)，旧批次使用[历史身份](../../docs/08-实验规范与共识/历史实验身份.md)与源码快照。




机制见[提出后的短程改写](../../docs/01-搜索方法/提出后的短程改写.md)。父代按训练质量乘尝试经验选择，Refine / Explore / Crossover 默认份额为 45:30:25。新的有效 Explore 程序中，按探索编号抽取 1/8，继续至多三步 Refine，每步从该探索达到的最好程序出发。失败的新程序至多修复一次，所有生成都计入预算。最终在独立选择集评价前五名，失败时按训练排名替补。

## 2026-10-10：原机制在当前协议下重跑

用户明确要求机制与实验条件分开：恢复当时 V10.17 的机制，在当前六任务协议下重新搜索。新入口为 `run_current` / `launch_current`，机制身份 `v1017-original-current-protocol-1`，批次 `20261010_server3_v1017_original_co6`；不续跑历史目录。

历史来源为提交 `5a1afe12`，其 V10.17 方法及实验源码与 `20261003_server3_v1017` 清单中的文件哈希逐一一致。保留质量×尝试经验选择、原随机流、45:30:25 算子比例、8根／16次初始化、形成路径与尝试／全局进展的分工、原算子目标和 Analysis、1/8 Explore 提案接三步 Refine、完整程序交付和一次 Repair。恢复整理版删掉的模板依赖补齐、原上下文过长处理和15行／1500字符的运行错误摘要。没有采用 V10.25 的双差异历史和提示改写。

评价适配只在本实验目录：六任务各三路，种子0/1/2，每路1000候选；训练16例、每实例候选CPU2秒／整实例墙钟20秒；训练前五重测冻结后，独立测试50例。调用当前评价器和冻结流程，不使用 V10.21 的提示。保留旧版计算许可与计时反馈，因此本次按用户的原机制要求，例外于通用准则的“提示不提时间”；提示中的旧整批限时改为实际逐实例约束，OP等任务的分数含义按当前协议显示。持久化继续使用公共档案格式，恢复时核验机制、配置、任务和评价身份。

```bash
uv run python -m experiments.traceaad_v10_17.run_current --task tsp_construct --dry-run
uv run python -m experiments.traceaad_v10_17.launch_current --batch 20261010_server3_v1017_original_co6 --suite co6 --repeats 3 --budget 1000 --eval-workers 8 --scheduler-socket /tmp/traceaad-1005/scheduler.sock --launch
```

新批次已于东京时间2026-10-10 19:52启动。19:53:15核验18路均存活且已有候选，累计48次（44次新有效、1次重复、2次超时、1次运行错误），模型服务错误和上下文裁剪均为0。所有路次机制身份正确，144个实现文件与启动快照一致。每分钟由本地 `watch_v1017_co6` 同步。

- [批次清单](../../archive/local_models_20261011/traceaad_v10_17/batch_20261010_server3_v1017_original_co6.json)、[源码身份](../../archive/local_models_20261011/traceaad_v10_17/source_20261010_server3_v1017_original_co6.json)与[源码快照](../../archive/local_models_20261011/traceaad_v10_17/source_20261010_server3_v1017_original_co6.tar.gz)
- [首次健康核验](../../archive/local_models_20261011/traceaad_v10_17/health_20261010_server3_v1017_original_co6_initial.json)
- [历史等价核对](../../archive/local_models_20261011/diagnosis_v1017_restart/audit.json)：17个历史源文件哈希吻合；三路历史TSP档案展开150份提示，在替换评价事实后逐字一致；300次父代与参考选择一致。入口10项检查与已有49项相关回归通过。
- [训练看板](http://127.0.0.1:8765/#b=traceaad_v10_17&rb=20261010_server3_v1017_original_co6)

下文记载10月6日整理入口，不能用它代替本次恢复入口。历史实验的五任务、独立选择集和旧计分也不作为本次条件。

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
