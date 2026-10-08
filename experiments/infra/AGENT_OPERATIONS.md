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

通过本地命令执行一次远端查询；读取器经 stdin 发送，不需部署新文件。远端需要当前 `benchmarks.tasks`、`monitor_timing`、`traceaad.common.storage` 模块。

```bash
uv run python -m experiments.infra.batch_status --ssh B3-server3 --repo /home/fzy/code/LLM4AD/TraceAAD --manifest experiments_result/traceaad_v10_16/batch_20261003_server3_v1016.json
```

连接失败时再检查 `ssh -G B3-server3`；仓库或解释器不存在时再定位环境。已知入口可用时无需重新搜索目录、检查全部依赖或读取 SSH 配置。

## 本机模型服务

后端 `local`（`http://127.0.0.1:8001/v1`，模型名 `Qwen3.8-27B`，3 槽）由 tmux 会话 `qwen3_8-27b-mtp` 中的 llama-server 提供：`~/models/gguf/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf`，上下文 98304（3 槽各 32768），MTP 草稿 3 个 token，KV 缓存 q8_0，日志 `/tmp/llama-qwen38-restart.log`。先用 `curl -s --noproxy '*' 127.0.0.1:8001/health` 检查；会话不存在时按 `~/models/gguf/qwen3.8-27b/update_ud_q4_k_xl.sh` 末尾的同一组参数在该会话中启动。

## 本机基线排队

基线在本机排队运行：同时最多 3 路（本机模型的 3 个槽位），一路结束补下一路；每路一次评价一个实例，核心向本机 CPU 调度器申请，与 TraceAAD 同一核心池。调度池只含每个物理性能核的一个线程（`--cpus 0,2,4,6,8,10,12,14`）；本机 16–31 号是能效核，llama-server 用 `taskset -a -cp 16-31 <pid>` 固定在能效核，重启模型服务后要重新执行；能效核同时是平台对照中的一个评价平台，本机模型服务有请求时会与它争用。server3 的调度器在其 tmux 会话 `traceaad_scheduler` 中运行（socket `/tmp/traceaad-1005/scheduler.sock`，0–51 号各一个物理核）；同步代码前的工作区备份在 server3 的 `~/backups/`。基线的超时率读运行目录的 `evaluations.jsonl`，`events.jsonl` 不区分失败类型。排队器在 tmux 会话中运行，日志在 `experiments_result/<方法>/launch_logs/`。

```bash
uv run python -m experiments.infra.local_queue --method funsearch --suite co6 --batch 20261008_local --slots 3
```

排队器只在内存中记录在跑的路次；它中途退出时，等这些路次结束后再重启。

## 等待、检索和同步

训练可视化使用本地 `8765` 端口（`http://127.0.0.1:8765/`）。先检查该端口；服务未运行时，用 `uv run python -m experiments.monitor --host 0.0.0.0 --port 8765` 启动。不指定 `--experiment` 时默认展示最近更新的批次；版本对比默认选中该 TraceAAD 批次和全部基线方法。查看其他批次用链接中的 `#b=<批次>`。页面读取本地档案，运行中路次显示最近同步的快照；文件变化会刷新缓存。核验 `/api/state?batch=traceaad_v10_16` 的路次数、曲线和 ETA，并用 `/api/compare?cohorts=traceaad_v10_16` 核验测试结果。

V10.15–V10.19 共用生成、评价和结果保存，各版本仍写到原版本目录。日常读取只使用 `events.jsonl`、`programs.jsonl`、`resume.json`、`summary.json`、`selection.json` 和平面的 `heldout.json`。原始模型请求与回复单独保存于 `calls.jsonl`，结束后压缩。诊断入口为 `uv run python -m experiments.traceaad_v10_17.diagnose --run-dir <目录>`，其他当前版本同样提供。格式、恢复条件和历史迁移见[实验与结果](SEARCH_FORMAT.md)。

横轴单位来自 `run_config.json` 的 `budget_axis`，候选编号和实际预算分别保留。曲线显示迄今最佳训练成绩，并延伸到最后一条已记录的预算位置。列表、最近候选、有效率和预算共用一个增量投影；有效率的分母是全部候选记录数。

详情显示搜索最优和最终选中程序。源码按内容哈希从 `programs.jsonl` 读取。带身份的测试结果须与冻结的最终程序一致；不一致或无法核验的记录不进入比较。

`.cache/history.json` 可以删除并重建。缓存只保存小型事件投影、程序元数据和读取位置，不重复源码、请求或回复。首次读取扫描轻量事件，后续只读追加字节；未完成的尾行下次重试。配置、恢复点、事件、程序、选择及批次清单变化会刷新 API 缓存；未完成路次的时间状态最多每 15 秒重算一次。页面每分钟更新批次列表，切换批次立即取消旧请求。`--experiment` 可固定默认批次，链接中的 `#b=` 优先。修改 Python 后重启服务，修改 HTML、CSS 或 JS 后刷新页面。

等待状态变化使用 `--wait-seconds 45`，默认每 10 秒在同一进程中检查，变化或到期后返回一次。时间戳和 ETA 自身变化不会唤醒调用方。每次最多等待 60 秒，之后可汇报状态或处理其他工作。普通快速查询直接等待返回；后台长任务按进度检查，避免每秒调用 `write_stdin`。

读取会话日志先抽样确认结构，再按日期、工作目录和消息类型筛选。先输出计数和少量样例，需要完整证据时再展开。文件检索先用 `rg --files` 定位，再用 `rg -n` 和局部读取；文件枚举顺序不代表时间顺序。

独立的本地读取和远端查询用 `Promise.allSettled` 并行，并检查每项结果。编辑、启动、同步和核验按依赖顺序执行。复杂远端 Python 通过 stdin 或脚本文件传递，命令参数用 shell quoting；避免层层嵌套引号。

server3 是一次性的执行机：不保留 git、历史版本、备份或已拉回的结果，只保留统一配好的 `.venv` 和正在运行的批次。用 `uv run python -m experiments.infra.remote` 管理：`push` 把本机工作区镜像到 server3（删除那边多出的代码，保留 `.venv` 与 `experiments_result`；依赖文件变化时执行 `uv sync --frozen`）；`pull --batch <批次>` 拉回该批次的运行目录、清单和启动日志；`prune --batch <批次>` 删除 server3 上已结束、已有 held-out 且与本地逐字节一致的运行，最后一路删掉时连同清单和启动日志；`watch --batch <批次>` 每 60 秒拉取并清理，批次在 server3 上清空后退出。启动 server3 批次前先 `push`；运行期间在本机 tmux 中 `watch`。

同步前固定已完成路次清单，再用 `rsync --files-from` 统一同步，随后对冻结文件核验哈希。新完成路次加入下一轮。运行中的日志可继续变化，应与冻结档案分开处理；不要因同步时日志增长而反复重传已冻结档案。批次全局汇总可能持续变化，不作为冻结路次的内容哈希依据。
