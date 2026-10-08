# 搜索实验与结果格式

当前结果统一为 `traceaad-results-v2`。运行目录是 `experiments_result/<实验>/<任务>/<运行>/`，例如 `traceaad_v10_17/tsp_construct/<运行>/`。实验条件保存在 `run_config.json`，实现改动用 revision 区分，当前科研实现为 `research-minimize-v3-20261007`。

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

源码身份是实际保存文本的 SHA-256。当前 TraceAAD 在交付时先规范化源码，再计算身份。`fitness` 和 `score` 都是统一后的最小化目标值，越小越好。TSP、CVRP、VRPTW 用正路径长度，装箱用正箱数，OP 用负奖励，CO-Bench 用负归一化质量。`run_config.json` 的 `objective` 为 `min`，`native_objective` 另记原任务方向。改进量统一为原值减去新值，正数表示改善。选中程序的训练成绩与独立选择成绩分别保存。

预算和候选编号分别记录。初始化、失败、重复和 Repair 都消耗一次候选预算。有效率使用有效候选记录数 / 全部候选记录数。

## 提交与恢复

一次完整尝试先追加调用、程序与事件，再原子替换 `resume.json`。恢复点的 `files` 保存已提交的字节位置。读取事实和训练曲线时只读到这个位置；恢复写入时丢弃未提交尾部。未提交的模型调用或评价可以重做。

恢复直接使用当前配置、机制和评价器，继续读取已有尝试，不要求保持原实验条件。活跃运行的 `run_config.json` 同步当前参数。模型遗漏依赖时，评价失败后进入一次 Repair；解析器不补模板中的导入、常量或辅助函数。基线的 `resume.json` 是方法最新快照。

## 测试记录

`heldout.json` 中每条记录至少有 `task`、`variant`、`scale`、`fitness` 和 `verification`。带程序身份的结果还保存 `key`、`node_id` 与评价配置。两个测试入口默认共用同一冻结程序、同一规模的已有成绩；新记录默认空名称。明确传入 `--variant` 时才使用另外的测试记录。

`verified` 表示成绩对应冻结的最终程序。

## 评价条件与执行

评价条件见[实验准则](../PROTOCOL.md)，任务设置集中在 `benchmarks/tasks.py`。六任务的基线与当前 TraceAAD 共用逐实例执行器，统一候选种子、实例隔离与时限；已有测试成绩直接读取，不因评价器变化而重测或改写。

通用评价命令：

```bash
uv run python -m experiments.infra.evaluate <运行目录> --units 50,100,200
uv run python -m experiments.infra.evaluate <运行目录> --condition traceaad --output-dir <汇总目录>
```

OBP 的规模写为 `1k_100,5k_500,10k_500`。未完成搜索必须显式使用 `--allow-incomplete`；任意中途程序的结果不会作为冻结最终程序的成绩参与比较。

## 共用读取与基线记录

恢复、曲线和成绩读取共用 `committed_size` / `committed_rows`；监控、批次状态和 held-out 共用 `selected_program` / `heldout_identity`。提交边界与源码身份只解释一次。搜索计时以最后一个完整候选的提交时间为准，进入选择后只更新阶段，避免把选择耗时算进搜索吞吐。

基线记录器只接受 `run_dir` 和可选起始样本数，使用当前单个标量 fitness。删除随机目录、日志风格切换、多目标模式和未使用的恢复接口。种群记录共用一个实现，各方法自己的研究轨迹仍按实际需要保存。

六任务的七种基线结束时，重评训练前五个不同源码的程序，冻结其中最好的有效结果；全部失败时明确记录失败。重评不消耗候选预算，评价次数单独记录。固定框架自己逐实例执行源码，安全入口不再预执行一次。调度器故障中止搜索并保留错误与模型调用记录。

模型模块负责请求重试与错误分类，SDK 自动重试关闭。搜索只累计候选预算及实际请求。每次物理请求的原始回复、耗时和错误都可以单独回查；基线调用记录也使用同一份请求事实。批次计划共用端点查询和 tmux 启动，容量、名称和端点来自一个后端配置。

已经结束的初始化与格式研究集中在 [historical](../historical/README.md)。当前搜索基础设施不依赖这些历史分析；研究证据与原输入名保留。

## 实验和分析入口

V10.15–V10.19 的 `run.py`、`heldout.py`、`heldout_batch.py`、`launch_batch.py` 和主机启动入口调用共用的 `experiments/infra/search_*.py`。批次计划按清单生成，held-out 按实际路次生成；跨主机复制后用清单所在目录定位运行目录。

```bash
uv run python -m experiments.traceaad_v10_17.diagnose --run-dir <运行目录>
uv run python -m experiments.infra.batch_status --manifest <批次清单>
uv run python -m experiments.monitor --host 0.0.0.0 --port 8765 --experiment traceaad_v10_17
```

可视化按追加字节刷新曲线，打开程序详情时才读取源码。列表、最近候选、有效率和预算共用一个投影。页面、样式和交互分别位于 `monitor.html`、`monitor.css`、`monitor.js`。
