# 搜索实验与结果格式

当前结果统一为 `traceaad-results-v2`。运行目录仍是 `experiments_result/<实验>/<任务>/<运行>/`，例如 `traceaad_v10_17/tsp_construct/<运行>/`。实验条件保存在 `run_config.json`，实现改动用 revision 区分，当前科研实现为 `research-minimize-v3-20261007`。

## 文件各存一类事实

| 文件 | 内容 |
| --- | --- |
| `run_config.json` | 任务、模型、种子、方法参数、评价协议、revision、预算及预算单位 |
| `events.jsonl` | 候选尝试、程序身份、逐种子评价、探索结束与阶段进度；按时间追加 |
| `programs.jsonl` | `key` 与源码正文；相同源码只存一次 |
| `calls.jsonl` | 实际模型请求、原始回复及调用元数据；结束后压缩为 `calls.jsonl.gz` |
| `resume.json` | 一个最新恢复点，含进度、RNG、固定协议和已提交文件位置；每次提交替换 |
| `summary.json` | 终局状态、计数、耗时及选中程序的身份与成绩；不重复源码 |
| `selection.json` | 独立选择的候选、评价和选中身份；没有独立选择的基线可省略 |
| `heldout.json` | 本路测试记录的平面列表；按 `(variant, scale)` 更新 |
| `best_program.py` | 最终程序的便捷导出；内容必须与选中 `key` 一致 |
| `.cache/history.json` | 可删除并重建的可视化投影；不保存源码、提示或回复 |

`diagnostics.json` 是离线分析产物，按需生成。基线的额外研究轨迹可以保留；普通可视化和成绩读取只依赖上表中的文件。`finalists.json`、分片样本文件和逐规模 held-out 文件不再重复保存同一结果。

## 事件与程序身份

`candidate` 事件记录 `candidate_id`、`budget_used`、`operator`、`status`、`fitness`、`valid`、`node_id` 和 `attempt`。`attempt` 保存父程序、参考程序、改动归属和 `call_ids`，通过调用编号查看原始请求和回复。新程序用 `program` 元数据引用 `programs.jsonl` 中的 `key`；重复尝试引用已有 `node_id`。评价和探索结束记录按发生时追加，不重写历史。

源码身份是实际保存文本的 SHA-256。当前 TraceAAD 在交付时先规范化源码，再计算身份；历史迁移不改写已经执行的程序。`fitness` 和 `score` 都是统一后的最小化目标值，越小越好。TSP、CVRP、VRPTW 用正路径长度，装箱用正箱数，OP 用负奖励，CO-Bench 用负归一化质量。`run_config.json` 的 `objective` 为 `min`，`native_objective` 另记原任务方向。改进量统一为原值减去新值，正数表示改善。选中程序的训练成绩与独立选择成绩分别保存。

预算和候选编号分别记录。当前 V10.15–V10.19 的初始化、失败、重复和 Repair 都消耗一次候选预算。旧实验的评价调用、预算槽位和样本次数仍保留原口径；迁移不把不同的计数改成同一种含义。有效率使用有效候选记录数 / 全部候选记录数。

## 提交与恢复

一次完整尝试先追加调用、程序与事件，再原子替换 `resume.json`。恢复点的 `files` 保存已提交的字节位置。读取事实和训练曲线时只读到这个位置；恢复写入时丢弃未提交尾部。未提交的模型调用或评价可以重做。

恢复要求配置与固定评价数据一致。模型遗漏依赖时，评价失败后进入一次 Repair；解析器不再补模板中的导入、常量或辅助函数。这些是当前 revision 的运行条件。历史未完成实验应使用原 Git 实现和原档案续跑，不能直接套用新运行条件。基线的 `resume.json` 是方法最新快照，不额外承诺完整续跑。

## 测试记录

`heldout.json` 中每条记录至少有 `task`、`variant`、`scale`、`fitness` 和 `verification`。带程序身份的结果还保存 `key`、`node_id` 与评价配置。不同评价条件使用不同 `variant`；当前方法默认空名称。通用评价入口默认使用 `shared:research-minimize-v3-20261007`；需要额外评价 TraceAAD 的种子条件时使用 `--condition traceaad`。旧 `shared` 成绩继续保留。

`verified` 表示成绩对应冻结的最终程序。`legacy` 表示历史记录缺少程序身份；保留并展示原成绩及此状态。程序或任务不匹配的结果不进入比较汇总。迁移不补造原实验没有记录的身份或评价。

## 评价条件与执行

五项任务的训练、独立选择和 held-out 条件集中在 `benchmarks/tasks.py`。任务入口从这里构造固定实例、规模、时限和 ACO 参数。数据集种子仍为训练 2024、测试 2025，独立选择的生成实例种子为 20260927。

- `shared`：基线条件，不额外重设候选的 Python/NumPy RNG，ACO 种子为 1234。
- `traceaad`：使用本路 `evaluation_seeds`（默认 730241）；评价前设定 Python/NumPy RNG，ACO 种子加上该评价种子。

两类条件保留各自含义，执行 revision 也进入协议身份。新通用评价器执行完整实例集合；TSP、VRPTW、OBP 顺序评价，`--workers` 控制 ACO 并行。旧通用 TSP 的逐实例进程和 ACO 私有求解入口已移除。新结果记录实际时限、种子和 revision，不覆盖旧条件的成绩。

统一入口负责设定种子、插入调用测量并执行实际候选。此前正式 TraceAAD 先执行模板，再执行候选，候选在这一层本来就只执行一次。本轮去掉模板预执行及 `source=` 绕行。OBP 为隔离实例状态仍逐实例重新执行程序。隔离进程、墙钟超时和终止子进程继续承担实际评价职责。

通用评价命令：

```bash
uv run python -m experiments.infra.evaluate <运行目录> --units 50,100,200
uv run python -m experiments.infra.evaluate <运行目录> --condition traceaad --output-dir <汇总目录>
```

OBP 的规模写为 `1k_100,5k_500,10k_500`。未完成搜索必须显式使用 `--allow-incomplete`；任意中途程序的结果不会作为冻结最终程序的成绩参与比较。

## 共用读取与基线记录

恢复、曲线和成绩读取共用 `committed_size` / `committed_rows`；监控、批次状态和 held-out 共用 `selected_program` / `heldout_identity`。提交边界与源码身份只解释一次。搜索计时以最后一个完整候选的提交时间为准，进入选择后只更新阶段，避免把选择耗时算进搜索吞吐。

基线记录器只接受 `run_dir` 和可选起始样本数，使用当前单个标量 fitness。删除随机目录、日志风格切换、多目标模式和未使用的恢复接口。种群记录共用一个实现，各方法自己的研究轨迹仍按实际需要保存。

模型模块负责请求重试与错误分类，SDK 自动重试关闭。搜索只累计候选预算及实际请求。每次物理请求的原始回复、耗时和错误都可以单独回查；基线调用记录也使用同一份请求事实。批次计划共用端点查询和 tmux 启动，容量、名称和端点来自一个后端配置。

已经结束的初始化与格式研究集中在 [historical](../historical/README.md)。当前搜索基础设施不依赖这些历史分析；研究证据与原输入名保留。

## 历史迁移与原档案

2026-10-07 已将本机 601 路已结束档案及 7,150 份诊断、汇总和历史数值文件切换为最小化。三路 V10.20 CVRP 的旧进程仍在运行；其 `.minimize/live.json` 标记只读最小化视图，原写入字节位置保持不变。后台 `traceaad_minimize_migration` 会在终局文件稳定后转换其事件、恢复点、选择与测试结果，再删除该标记。此次迁移只转换数值表示，没有重跑评价或重新选择程序。程序源码、模型原始请求与回复、任务实例、种子、预算、状态及身份保留原记录。过去实际送给模型的文字仍可从原始调用查看；新提示统一说明 Lower is better。原实验 revision 和协议身份继续指向当时条件，结果格式 v2 指明现在的数值方向。旧的未完成搜索不能把已转换的恢复点交给旧进程续写。

数值备份与核验记录位于 `experiments_result/.archive/minimize_20261007/`，每路有 `.minimize/receipt.json`。迁移器支持重复执行，已完成的文件不会再次取负：

```bash
uv run python -m experiments.infra.migrations.minimize_results --root experiments_result --auxiliary
```


本机 30 个批次、564 路训练结果，以及 25 份用于诊断的搜索副本已转换。原来的日志、树快照、样本分片和测试文件先压缩存入 `experiments_result/.archive/storage_20261006/`，核对曲线、预算、候选计数和 held-out 成绩后，再从工作目录删除。每路 `.conversion/receipt.json` 和归档中的 `report.json` 保存迁移记录；它们不参与日常结果读取。

历史字段兼容只在 `experiments/infra/migrations/`。`Facts`、训练可视化、成绩读取与批次状态只读当前格式。需要导入另一份旧档案时，可以显式运行：

```bash
uv run python -m experiments.infra.migrations.convert_results --batch <历史实验目录名> --cleanup
```

转换器只用于已有旧格式的档案，已经封存的路次会跳过。原格式可从对应 `*.tar.gz` 解包到单独目录；恢复说明见[结果存储](RESULT_STORAGE.md)。

迁移时仍在执行的两路 V10.19 保留旧进程写入的文件，由临时 `--watch` 进程每 15 秒转换快照。搜索进程结束后自动完成归档和清理，随后退出。它不改变正在进行的搜索机制或评价条件。

## 实验和分析入口

V10.15–V10.19 的 `run.py`、`heldout.py`、`heldout_batch.py`、`launch_batch.py` 和主机启动入口调用共用的 `experiments/infra/search_*.py`。批次计划按清单生成，held-out 按实际路次生成；跨主机复制后用清单所在目录定位运行目录。

```bash
uv run python -m experiments.traceaad_v10_17.diagnose --run-dir <运行目录>
uv run python -m experiments.infra.batch_status --manifest <批次清单>
uv run python -m experiments.monitor --host 0.0.0.0 --port 8765 --experiment traceaad_v10_17
```

可视化按追加字节刷新曲线，打开程序详情时才读取源码。列表、最近候选、有效率和预算共用一个投影。页面、样式和交互分别位于 `monitor.html`、`monitor.css`、`monitor.js`。
