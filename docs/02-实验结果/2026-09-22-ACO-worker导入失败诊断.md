# 2026-09-22：ACO worker 导入失败诊断

**恢复更新（同日 21:55 左右）**：用户授权后已修复源码、备份并回滚八路 ACO，从异常前状态绑定新冻结运行时续跑。八路均已产生新的有效分数，没有重现 worker 导入故障或异常超时。下面保留故障调查证据；修复、验证和恢复明细见文末。

已确认，当前 ACO 大面积 timeout 的直接原因是运行中的实验读取了被清理破坏的工作区代码：新建的 spawn worker 在模块导入阶段退出，进程池反复补建 worker，外层评价最终等到超时。它不是仅凭高负载推测出的资源不足，也不能作为候选算法陷入局部最优的证据。冻结源文件仍完整，但实际启动和续跑没有使用该冻结目录。

## 直接证据与故障链

1. **当前五路 ACO 的终端都有相同 worker 异常。** 从四路 CVRP 与 OP rep1 的 tmux pane 各保存最近 2000 行，每路包含 40–42 次 `IndentationError`。栈从 `multiprocessing.spawn.prepare → _fixup_main_from_name → experiments.traceaad_v10_12.run` 进入工作区，再通过 `llm4ad.__init__ → method.__init__` 自动导入 V10.10，在 `llm4ad/method/traceaad_v10_10/traceaad.py:40` 失败。
2. **V10.10 清理实际留下了语法错误。** 第 39、40 行连续为 `class TraceAADV1010(TraceAADV108):` 和 `class TraceAADV1010(TraceAADV1011):`，前一个 class 没有缩进主体。工作区 `ast.parse` 与新解释器 import 都直接复现该错误。它由提交 `4718b68a` 增加；文件 mtime 为 16:13:58，该提交时间为 16:32:18。
3. **还有一个更早的独立导入破坏。** `experiments/traceaad_v10_12/run.py:8` 仍导入 `TraceAADV1012RandCtx`，对应包已被删除。所谓解耦只在 main 尾部追加了 generic 参数赋值，没有删除顶部 import、参数和旧分支；`freeze.py` 的 sources 也仍包含已删包。run.py 的 mtime 为 15:55:40，与八路最后有效评价的 15:54:59–15:56:15 时间段一致。
4. **冻结副本存在，但没有隔离实际运行。** `manual_launch.py` 调用 freeze 后，使用 `cwd=ROOT` 执行工作区的 `python -m experiments.traceaad_v10_12.run`，没有切到 runtime 或设置相应 PYTHONPATH。当前五个搜索父进程的 cwd 也都是仓库根目录，没有 PYTHONPATH 设置。实际 worker traceback 同样指向工作区，排除了“虽然 cwd 在工作区但模块仍来自冻结目录”的解释。
5. **长驻父进程与新 worker 读取代码的时机不同。** 父进程在清理前已导入模块，仍能生成候选、写出预算记录。ACO 在构建候选 prior 后调用 `multiprocessing.get_context('spawn').Pool(...)`，新解释器重新导入入口，因上述错误退出。Python Pool 的 `_maintain_pool` 会补建退出 worker；任务结果无法返回，外层 `SecureEvaluator` 到 60 秒（OP）或 120 秒（CVRP）后写 `timeout`，掩盖了底层启动异常。

候选仍可能在构建 prior 时先触发自身的 runtime_error，所以不能把异常段的每个失败都说成 worker 导入错误。直接受阻的是 worker 内的 ACO job，不能把这里描述成“候选函数完全没有执行”。

## 隔离检验

检验不调用 LLM、不运行候选评价、不改变实验进程或任何冻结文件。父解释器先导入完整冻结版本，模拟清理前已经运行的主进程；随后只改变 spawn 子进程的源路径。所有对照一致限制诊断用 BLAS/OMP 线程，以降低探针开销，实验进程环境未改。

| 子进程读取的源码 | 启动结果 | 异常 |
| --- | --- | --- |
| 完整冻结副本 | 成功，exit code 0 | 无 |
| 临时复制冻结副本，仅删除 rand_ctx 包 | 失败，exit code 1 | `ModuleNotFoundError: No module named 'llm4ad.method.traceaad_v10_12_rand_ctx'` |
| 当前工作区 | 失败，exit code 1 | 与活跃 worker 相同的 `IndentationError` |

冻结 manifest 的 75 个文件全部通过 SHA256 校验。工作区的 `llm4ad/base/evaluate.py`、CVRP/OP 的 `evaluation.py` 和 V10.12 搜索主文件均与冻结副本一致；受破坏的是导入依赖和入口，而非本次查看的任务评价逻辑或 V10.12 搜索主文件。该检验已足以确认启动故障，因此没有额外消耗真实评价预算做候选复测。

## 时间边界与结论强度

- **直接确认**：当前活跃 ACO 的 spawn worker 因 V10.10 语法错误持续退出；当前工作区存在 rand_ctx 残留导入；实际启动路径没有隔离；两种导入故障均可复现。
- **高可信的起始解释**：15:55 的首次共同中断与 run.py 修改、rand_ctx 清理一致；仅删除该包就足以触发相同 spawn 启动失败机制。15:55 的最初 worker stderr 已不在现有 tmux 历史里，因此不将首次具体异常类型写成直接观察事实。
- **后续加重**：16:13:58 的 V10.10 语法错误让所有经 `llm4ad` 自动发现方法的全新导入失败；它会先于 rand_ctx 残留导入报错。
- **高负载的解释更新**：上一轮只读核查观察到 load 约 93–106 和大量线程；现在确认存在 worker 退出/补建循环，能解释额外进程启动开销。尚未量化它对全部负载的占比，不能将高负载认定为独立首因，也不必用调整线程数解释确定存在的导入错误。
- **为何其他任务还能继续**：外层安全评价在 Linux 使用 fork，继承已加载模块；五任务中 ACO 的内层评价显式使用 spawn Pool，额外重新导入入口。其他任务继续产生分数不能证明运行时未受代码变更影响。

21:20 快照中八路 ACO 自最后成功之后已有 895 次失败（776 次 timeout）；该数值是固定快照，活跃五路仍可能继续消耗预算。旧工作流水中“清理未影响在线运行”“CLI 已验证正常”的记录与当前文件、提交及实际 traceback 不符，不能继续用作安全依据。

## 修复与恢复应采用的顺序

本次范围为根因调查，未停止或重启活跃任务，也未修改机制、工作区方法代码或原始 journal。

1. 经用户授权后，对五路活跃 ACO 在安全检查点暂停并完整留存状态、日志、routing history；已完赛的 OP 三路也保留异常尾段证据。
2. 实验恢复优先使用已校验的冻结目录，显式控制 cwd 与导入路径，并验证 spawn 子进程实际加载位置；仅修工作区语法错误仍会暴露 rand_ctx 残留导入，不能当作完整修复。
3. 使用既有 `experiments/infra/evaluate.py` 在独立输出目录复核已知有效候选，保持训练数据、种子、ACO 参数和 timeout；通过后再决定恢复搜索。
4. 八路 ACO 的异常尾段已消耗正式预算，不能原地删除后假称原批自然完成。若决定回滚重跑，需保留原快照，并从异常前可验证的安全状态恢复，包括 RNG、父代分配和 journal 一致性；不能只把 budget_used 减去失败数。当前日志中的最后有效 E 只是诊断边界，不自动等于可直接恢复的完整检查点。
5. 工作区旧版本清理另外修复并验证。防回归检查至少包括：全新解释器导入、实际 `-m` 入口、spawn 子进程启动，以及运行进程确实指向冻结源码。只有方法单测通过无法覆盖这条故障链。

## 复核位置

- [启动对照脚本](../../experiments/traceaad_v10_12/analysis_20260922/aco_incident/probe_spawn_imports.py) 与 [三组完整输出](../../experiments/traceaad_v10_12/analysis_20260922/aco_incident/spawn_probe.json)。运行命令：`.venv/bin/python experiments/traceaad_v10_12/analysis_20260922/aco_incident/probe_spawn_imports.py`。
- [文件哈希、mtime 与实际进程路径](../../experiments/traceaad_v10_12/analysis_20260922/aco_incident/source_and_process_evidence.json)。
- [CVRP rep2 活跃 worker traceback](../../experiments/traceaad_v10_12/analysis_20260922/aco_incident/v1012_cvrp_r2_pane.txt)；同目录保存另外四路 pane。
- [原进展快照](../../experiments/traceaad_v10_12/analysis_20260922/progress_snapshot.json) 与 [V10.12 结果及失败边界](各版本实验记录/V10.12-实验结果.md)。

## 授权修复与实际恢复

用户在根因确认后明确要求修复代码，并从错误发生前的正确部分续跑。已仅暂停五路活跃 ACO，连同三路已耗尽预算的 OP，将八路完整目录、manifest 和终端错误保存到 `experiments/traceaad_v10_12/backup_before_aco_recovery_20260922_214100/`。模型服务、监控和其他实验进程未停止。

### 代码修复

- V10.12 run/launch/freeze：彻底移除已删除 rand_ctx 的导入、参数、分支和冻结来源，修复 launch.py 清理时留下的重复函数签名/关键字及括号错误。
- V10.10：回退不完整的继承改写，恢复清理前实现和所必需的 V10.3–V10.8 方法依赖（23 个文件）；没有恢复旧实验目录或结果，也没有把 V10.10 替换为 V10.11 的算法语义。恢复文件清单为 `restored_dependency_paths.txt`。
- manual launcher：创建冻结副本后校验全部文件哈希与语法，并进行真实 spawn 入口预检；tmux 显式使用 `-c <runtime>` 和 `-e PYTHONPATH=<runtime>`。不能仅依赖传给 tmux 客户端的环境变量。
- run 入口：正式搜索拒绝直接从无 runtime manifest 的可变工作区启动。`--help` 与导入仍可用于检查；`--no-freeze` 只表示复用已有且通过校验的 runtime。
- 新部署运行时为 `results/runtime_20260922_v1012_recovery/`。原冻结副本完整保留；新旧运行时的 40 个搜索机制、任务和评价器源文件哈希一致。改变的是启动/依赖隔离，不改变算法、任务、数据、种子、超时或真实预算。

### 恢复状态如何核验

选用共同安全边界 **2026-09-22 15:54:00 之前最后一个已结算候选**，早于首次共同故障；为避免误纳入在途影响，少量故障边界附近的正常调用也一并重做。其余 TSP/OBP/VRPTW 12 路已完成结果保持原样，本次恢复范围是直接确认受导入故障影响的八路 ACO。

| 任务 | rep1 | rep2 | rep3 | rep4 |
| --- | ---: | ---: | ---: | ---: |
| CVRP 恢复 E | 650 | 661 | 607 | 657 |
| OP 恢复 E | 744 | 866 | 974 | 917 |

历史中间 checkpoint 文件未逐次保留，因此从完整 journal 前缀重建该已结算状态，**没有直接沿用尾端 RNG，也没有只修改预算**：

1. 在对应 seed 的随机流中定位最终已存 RNG 状态，根据尾段实际调度的随机抽样次数恢复边界 RNG。
2. 使用未改变的 V10.12 `_schedule`，离线回放边界之后全部 **985 条候选记录**；逐条核验 operator、parent、donor、reference_ids、repair_of、父代累计次数和抽样概率。
3. 回放终点的 RNG、节点集合和父代计数与备份 checkpoint 完全一致，证明恢复的前缀状态能衔接实际历史。恢复的 step_counter、completed_attempts、invalid_streak、last_event 也由已结算记录重建。
4. 八路全部验证后才写入前缀；写入前检查会话已停止、原始文件仍与备份相同。events/nodes/llm_calls 保留原始字节前缀，旧完成 summary 留在备份，原终端日志保留并追加恢复标记。

共从活跃 journal 剔除 **955 次真实评价调用**对应的尾段，原件没有销毁。每路仍以 E1000 为逻辑总预算；重跑已丢弃调用意味着实际累计物理支出额外增加，不能把这次事故成本从资源统计中抹去。manifest 记录恢复点、来源哈希和备份；每路 routing history 追加 runtime 切换记录，后端映射保持原配置。

### 验证与实际续跑结果

- 方法测试、运行时回归测试与冻结任务契约：**25 passed**。
- 两个固定训练候选通过共享 `experiments.infra.evaluate` 暴露的评价类及 SecureEvaluator 复核：CVRP `-8.6427893553783`，OP `14.703999999999999`，与故障前记录一致（误差 <1e-9），耗时约 4.49 秒、1.62 秒；诊断评价独立记录，不写入正式搜索预算。这一步使用与最终部署机制/评价器哈希相同的候选冻结修复版本。
- 八路父进程均核验 cwd/PYTHONPATH 指向最终 runtime，均记录 `resumed=True`。全部八路已有新有效评价；观察窗口无 timeout 和 worker 导入异常。OP rep3 首个生成候选发生普通 runtime_error，按原一次修复规则在下一次评价得到有效分数，未改错误处理策略。
- 续跑后机器负载降至约 1–2，支持先前进程补建循环是主要异常来源，但此处不把负载下降当作算法质量证据。

复核产物均在 `experiments/traceaad_v10_12/analysis_20260922/aco_incident/`：

- `recover_checkpoints.py`、`restored_prefix/recovery_receipt.json`：恢复算法、逐路边界、RNG 校验与 journal 前缀哈希；大体积状态副本保留在本地。
- `validate_frozen_evaluation.py`、`evaluation_validation.json`：固定候选训练复核。
- `resume_verified_runs.py`、`resume_health.json`：限定八路恢复入口及恢复后首次有效评价证据。

后续应以恢复后的 journal 继续统计训练曲线；此前 21:20 的快照保留为事故诊断材料，不能与恢复后的记录混合计算最终评价次数或质量。
