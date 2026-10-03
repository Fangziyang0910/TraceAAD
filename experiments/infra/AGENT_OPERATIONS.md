# Agent 执行入口

检查实验进度、等待完成、同步远端档案时，先读本文。这里保存稳定入口；进程、资源和完成数量以当次查询为准。

## 批次状态

V10.15/16 的批次清单是查询范围的依据。指定清单，不遍历所有版本，也不以已同步的路次代替整个批次。

```bash
uv run python -m experiments.infra.batch_status --manifest experiments_result/traceaad_v10_16/batch_20261003_server3_v1016.json
```

默认返回总数、搜索 ETA，以及需要注意的路次。`--details` 展开每路，`--json` 返回结构化数据。待 held-out 只包括已完成且选定程序哈希和节点均通过核验的路次；无效结果、程序不匹配和缺失结果分别计数。此入口针对 V10.15/16 原生逐路结果，不合并旧版批次结果。

每个未完成路次最多读取 1 MiB 日志尾部；完整路次直接读总结。尾部缺少完整记录时返回未知，不扫描整个日志补齐。`recorded` 表示存在进度记录，不能据此确认进程存活；`stale` 是日志长期未更新，需要进一步检查。ETA 只估算搜索，不包含独立选择及 held-out。本地副本只说明已同步状态。

## server3

2026-10-03 已使用并核实的入口：

| 对象 | 入口 |
| --- | --- |
| SSH 主机别名 | `B3-server3`（不是模型后端名称 `server3`） |
| 远端仓库 | `/home/fzy/code/LLM4AD/TraceAAD` |
| 远端 Python | 仓库内 `.venv/bin/python` |
| 实验档案 | 仓库内 `experiments_result/` |

通过本地命令执行一次远端查询；读取器经 stdin 发送，不需部署新文件。远端需具备现有的 `monitor_results` 和 `monitor_timing` 模块。

```bash
uv run python -m experiments.infra.batch_status --ssh B3-server3 --repo /home/fzy/code/LLM4AD/TraceAAD --manifest experiments_result/traceaad_v10_16/batch_20261003_server3_v1016.json
```

连接失败时再检查 `ssh -G B3-server3`；仓库或解释器不存在时再定位环境。已知入口可用时无需重新搜索目录、检查全部依赖或读取 SSH 配置。

## 等待、检索和同步

等待状态变化使用 `--wait-seconds 45`，默认每 10 秒在同一进程中检查，变化或到期后返回一次。时间戳和 ETA 自身变化不会唤醒调用方。每次最多等待 60 秒，之后可汇报状态或处理其他工作。普通快速查询直接等待返回；后台长任务按进度检查，避免每秒调用 `write_stdin`。

读取会话日志先抽样确认结构，再按日期、工作目录和消息类型筛选。先输出计数和少量样例，需要完整证据时再展开。文件检索先用 `rg --files` 定位，再用 `rg -n` 和局部读取；文件枚举顺序不代表时间顺序。

独立的本地读取和远端查询用 `Promise.allSettled` 并行，并检查每项结果。编辑、启动、同步和核验按依赖顺序执行。复杂远端 Python 通过 stdin 或脚本文件传递，命令参数用 shell quoting；避免层层嵌套引号。

同步前固定已完成路次清单，再用 `rsync --files-from` 统一同步，随后对冻结文件核验哈希。新完成路次加入下一轮。运行中的日志可继续变化，应与冻结档案分开处理；不要因同步时日志增长而反复重传已冻结档案。批次全局汇总可能持续变化，不作为冻结路次的内容哈希依据。
