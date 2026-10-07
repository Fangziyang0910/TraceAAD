# Agent 执行入口

检查实验进度、等待完成、同步远端档案时，先读本文。这里保存稳定入口；进程、资源和完成数量以当次查询为准。

## 批次状态

批次清单是查询范围的依据。指定清单，不遍历所有版本，也不以已同步的路次代替整个批次。

```bash
uv run python -m experiments.infra.batch_status --manifest experiments_result/traceaad_v10_16/batch_20261003_server3_v1016.json
```

默认返回总数、搜索 ETA，以及需要注意的路次。`--details` 展开每路，`--json` 返回结构化数据。待 held-out 只包括已完成且选定程序哈希和节点均通过核验的路次；无效结果、程序不匹配、历史身份缺失和未评价结果分别计数。此入口读取统一的逐路结果，不依赖版本号猜文件格式。

每个未完成路次最多读取 1 MiB 日志尾部；完整路次直接读总结。尾部缺少完整记录时返回未知，不扫描整个日志补齐。`recorded` 表示存在进度记录，不能据此确认进程存活；`stale` 是日志长期未更新，需要进一步检查。ETA 只估算搜索，不包含独立选择及 held-out。本地副本只说明已同步状态。

## server3

2026-10-03 已使用并核实的入口：

| 对象 | 入口 |
| --- | --- |
| SSH 主机别名 | `B3-server3`（不是模型后端名称 `server3`） |
| 远端仓库 | `/home/fzy/code/LLM4AD/TraceAAD` |
| 远端 Python | 仓库内 `.venv/bin/python` |
| 实验档案 | 仓库内 `experiments_result/` |

通过本地命令执行一次远端查询；读取器经 stdin 发送，不需部署新文件。远端需要当前 `benchmarks.tasks`、`monitor_timing`、`traceaad.common.storage` 模块，并已转换为当前结果格式。旧格式主机需先完成迁移，再使用此查询入口。

```bash
uv run python -m experiments.infra.batch_status --ssh B3-server3 --repo /home/fzy/code/LLM4AD/TraceAAD --manifest experiments_result/traceaad_v10_16/batch_20261003_server3_v1016.json
```

连接失败时再检查 `ssh -G B3-server3`；仓库或解释器不存在时再定位环境。已知入口可用时无需重新搜索目录、检查全部依赖或读取 SSH 配置。

## 等待、检索和同步

训练可视化使用本地 `8765` 端口（`http://127.0.0.1:8765/`）。先检查该端口；服务未运行时，用 `uv run python -m experiments.monitor --host 0.0.0.0 --port 8765` 启动。不指定 `--experiment` 时默认展示最近更新的批次；版本对比默认选中该 TraceAAD 批次和全部基线方法。查看其他批次用链接中的 `#b=<批次>`。页面读取本地档案，运行中路次显示最近同步的快照；文件变化会刷新缓存。核验 `/api/state?batch=traceaad_v10_16` 的路次数、曲线和 ETA，并用 `/api/compare?cohorts=traceaad_v10_16` 核验测试结果。

V10.15–V10.19 共用生成、评价和结果保存，各版本仍写到原版本目录。日常读取只使用 `events.jsonl`、`programs.jsonl`、`resume.json`、`summary.json`、`selection.json` 和平面的 `heldout.json`。原始模型请求与回复单独保存于 `calls.jsonl`，结束后压缩。诊断入口为 `uv run python -m experiments.traceaad_v10_17.diagnose --run-dir <目录>`，其他当前版本同样提供。格式、恢复条件和历史迁移见[实验与结果](SEARCH_FORMAT.md)。

历史版本已经转换为同一种事件结构。横轴单位来自 `run_config.json` 的 `budget_axis`，候选编号和实际预算分别保留。曲线显示迄今最佳训练成绩，并延伸到最后一条已记录的预算位置。列表、最近候选、有效率和预算共用一个增量投影；有效率的分母是全部候选记录数。

详情显示搜索最优和最终选中程序。源码按内容哈希从 `programs.jsonl` 读取。带身份的测试结果须与冻结的最终程序一致；不一致或无法核验的记录不进入比较。原实验没有身份记录的成绩保留 `legacy` 状态，与原来的比较口径一致。

`.cache/history.json` 可以删除并重建。缓存只保存小型事件投影、程序元数据和读取位置，不重复源码、请求或回复。首次读取扫描轻量事件，后续只读追加字节；未完成的尾行下次重试。配置、恢复点、事件、程序、选择及批次清单变化会刷新 API 缓存；未完成路次的时间状态最多每 15 秒重算一次。页面每分钟更新批次列表，切换批次立即取消旧请求。`--experiment` 可固定默认批次，链接中的 `#b=` 优先。修改 Python 后重启服务，修改 HTML、CSS 或 JS 后刷新页面。

2026-10-06 迁移时，两路 V10.19 仍由旧进程执行。临时快照转换器读取它们的旧日志，搜索完成后自动归档、清理并退出。需要排查时检查 `/tmp/traceaad-storage-migration-watch.pid` 和同名 `.log`；不重启搜索进程。

等待状态变化使用 `--wait-seconds 45`，默认每 10 秒在同一进程中检查，变化或到期后返回一次。时间戳和 ETA 自身变化不会唤醒调用方。每次最多等待 60 秒，之后可汇报状态或处理其他工作。普通快速查询直接等待返回；后台长任务按进度检查，避免每秒调用 `write_stdin`。

读取会话日志先抽样确认结构，再按日期、工作目录和消息类型筛选。先输出计数和少量样例，需要完整证据时再展开。文件检索先用 `rg --files` 定位，再用 `rg -n` 和局部读取；文件枚举顺序不代表时间顺序。

独立的本地读取和远端查询用 `Promise.allSettled` 并行，并检查每项结果。编辑、启动、同步和核验按依赖顺序执行。复杂远端 Python 通过 stdin 或脚本文件传递，命令参数用 shell quoting；避免层层嵌套引号。

同步前固定已完成路次清单，再用 `rsync --files-from` 统一同步，随后对冻结文件核验哈希。旧格式档案先同步到独立暂存实验目录，显式运行迁移工具后再纳入本地读取；不要覆盖已转换的运行目录。新完成路次加入下一轮。运行中的日志可继续变化，应与冻结档案分开处理；不要因同步时日志增长而反复重传已冻结档案。批次全局汇总可能持续变化，不作为冻结路次的内容哈希依据。

2026-10-07 已统一为最小化目标。三路运行中的 V10.20 CVRP 暂由 `.minimize/live.json` 标记只读转换，旧进程继续按原条件运行。完成迁移的后台会话为 `traceaad_minimize_migration`，日志在 `/tmp/traceaad-minimize-live.log`；可用 `tmux has-session -t traceaad_minimize_migration` 检查。它只在搜索终局文件稳定后改写分数和恢复字节位置，不中断实验。监控已重新加载；五个任务均标记为 `min`。


2026-10-07，新增四任务的12路实验已归入 `experiments_result/traceaad_v10_20/`，清单为 `batch_20261007_local_v1020_new4_seeded.json`。训练页统一显示9个任务、27路：`http://127.0.0.1:8765/#b=traceaad_v10_20`。

运行进程仍持有旧路径，临时别名连接到新的实际目录。`traceaad_v10_20_co6/` 当前只承担旧路径转接；V10.20 任务目录内带旧名称的符号链接也是写入别名，不是额外实验。监控不计入这些别名。自动清理会话为 `traceaad_v1020_path_cleanup`，读取 `/tmp/traceaad-v1020-path-aliases.json`；对应原进程退出后删除其运行别名，全部结束后删除旧路径容器并退出。运行期间不要手动删除这些链接。原启动命令保留在清单的 `original_command`，当前规范路径与续跑命令使用清单的 `run_dir` 和 `command`。

2026-10-07任务替换：主套件 `co6` 现为TSP、CVRP、FSSP、图着色、JSSP与OP。原新增批次的背包、集合覆盖各三路已经停止，`summary.status=stopped` 与清单中记录原因，检查点保留。状态查询和监控将其显示为已停止，不列为运行故障或估算剩余运行时间。原FSSP、图着色六路继续运行。JSSP与OP的替换批次统一归入 `traceaad_v10_20`，清单为 `batch_20261007_local_v1020_replacement.json`；原六路和替换六路合计两个模型端点各六路。运行中路径别名仍由现有清理会话管理。
