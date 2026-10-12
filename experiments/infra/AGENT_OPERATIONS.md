# 实验执行入口

实时状态以批次清单、日志和实际进程为准。实验条件见[规范](../../docs/08-实验规范与共识/README.md)，成绩见[结果](../../docs/02-实验结果/01-搜索结果概览.md)。

## 状态查询

```bash
uv run python -m experiments.infra.batch_status --manifest <批次清单> --details
uv run python -m experiments.infra.batch_status --ssh B3-server3 --repo /home/fzy/code/LLM4AD/TraceAAD --manifest <批次清单> --details
```

`--json` 返回结构化数据，`--wait-seconds 45` 等待状态变化，每次最多等待 60 秒。`recorded` 只表示存在记录，`stale` 需要检查进程；本地副本只表示已同步状态。ETA 仅包含搜索。预算读取方法事件和恢复点，不能用有效候选数代替。

## server3

| 对象 | 入口 |
| --- | --- |
| SSH | `B3-server3` |
| 仓库 | `/home/fzy/code/LLM4AD/TraceAAD` |
| Python | 仓库内 `.venv/bin/python`，不解析 symlink |
| CPU 调度器 | tmux `traceaad_scheduler`，socket `/tmp/traceaad-1005/scheduler.sock`，物理核 0–51 |
| 实验数据 | `experiments_result/` |

运行设置 `HF_HUB_OFFLINE=1`，凭据从远端 `.env` 读取。连接失败时检查 `ssh -G B3-server3`。调度器重启前确认旧进程已退出，再删除遗留 socket；结束 tmux 不会自动删除 socket。

server3 用作执行机，保留统一 `.venv` 和在跑数据。代码同步使用现有入口，不在搜索进程运行时覆盖公共源码：

```bash
uv run python -m experiments.infra.remote push
uv run python -m experiments.infra.remote pull --experiment <方法> --batch <批次>
uv run python -m experiments.infra.remote watch --experiment <方法> --batch <批次>
uv run python -m experiments.infra.remote prune --experiment <方法> --batch <批次>
```

`push` 镜像代码，删除远端多余代码但保留 `.venv` 与结果；依赖变化时执行 `uv sync --frozen`。`watch` 每 60 秒拉取并清理已完成档案。`prune` 仅清理已完成、已有独立测试且与本地逐字节一致的数据。冻结档案先固定清单，再同步并核验哈希；增长中的日志不用于冻结档案的一致性判断。

## 看板与结果读取

```bash
uv run python -m experiments.monitor --host 0.0.0.0 --port 8765
```

看板地址 `http://127.0.0.1:8765/`。`#b=<方法>&rb=<批次>` 固定批次，`rb=latest` 跟随最新批次。API 为 `/api/state?batch=<方法>&run_batch=<批次>`，跨方法结果为 `/api/compare?cohorts=eoh,reevo`。修改 Python 后重启服务，修改 HTML、CSS、JS 后刷新页面。

日常读取 `events.jsonl`、`programs.jsonl`、`resume.json`、`summary.json`、`selection.json`、`heldout.json`；原始请求在 `calls.jsonl` 或其 gzip 档案。基线失败类型与超时率读 `evaluations.jsonl`。测试结果须与最终选中程序的哈希和节点一致。`.cache/history.json` 可删除重建。格式见[SEARCH_FORMAT](SEARCH_FORMAT.md)。

<a id="中断后的基线恢复"></a>

## 基线恢复

恢复入口支持 EoH、ReEvo、FunSearch、MCTS-AHD、CALM、PathWise 和 ShinkaEvo。在执行机仓库根目录运行：

```bash
# 只读核对配置、预算与搜索状态
HF_HUB_OFFLINE=1 .venv/bin/python -m experiments.infra.resume_baseline --run-dir <运行目录>
# 缺少评分向量的 FunSearch 旧档案，先做 CPU 重评
HF_HUB_OFFLINE=1 .venv/bin/python -m experiments.infra.resume_baseline --run-dir <运行目录> --prepare-profiles
# 单路或整批续跑
HF_HUB_OFFLINE=1 .venv/bin/python -m experiments.infra.resume_baseline --run-dir <运行目录> --execute
HF_HUB_OFFLINE=1 .venv/bin/python -m experiments.infra.resume_baseline --manifest <批次清单> --execute
```

批次入口复用原 tmux 会话，跳过已存在的会话和已冻结路次。满预算但未冻结的路次仅做最终选择。日志为 `logs/resume_search.log`；`launched` 仅确认会话创建，仍须核对进程和新增记录。

| 方法 | 恢复状态与边界 |
| --- | --- |
| EoH | 最近种群和后续候选，从下一算子继续 |
| ReEvo | 最近完整种群、当前最好程序、长期反思，从新一代继续 |
| MCTS-AHD | 按事件和真实父代提示重建搜索树，从新一轮继续 |
| FunSearch | 岛归属与逐实例评分向量，从新提示批继续 |
| CALM | 自动检查点保存候选、种子、评分向量、提示回访与数值微调状态 |
| PathWise | 自动检查点保存初始化、种群、节点、父代与反思状态 |
| ShinkaEvo | 自动检查点保存岛副本、archive、embedding、bandit、meta 与代数 |

前三种方法及 FunSearch 的旧日志恢复为 `journal_boundary`，原随机状态与执行栈未保存，按原 seed 重新开始边界后的执行。后三种方法的新实验保存 `search_checkpoint.json.gz`，恢复 Python／NumPy 随机状态；执行栈不保存。未完成阶段从保存边界继续，已有候选支出与 token 累计不退回。旧档案缺少必需状态时拒绝恢复，不能只凭最好程序补造状态。

FunSearch 已有评分向量时直接读取；缺失时用重评分数与向量重建数据库，保留岛归属。失败程序排除，最终训练排序使用重评分数，数据分别保存在 `recovery_profiles.jsonl` 和 `recovery_evaluations.jsonl`。空岛、父代不唯一或岛重置来源缺失时拒绝启动。CPU 补全结束不会自动恢复模型请求，准备状态见 `recovery_ready.json`。

新启动与恢复共用 `.recovery.lock`，进程退出后释放；带 `writer_lock` 的遗留 running 状态可再次恢复。旧进程没有该标记时，先核实退出再修正状态。批次更新用 `.batch.lock` 串行写入。未完成 JSONL 尾行在写入前移到 `logs/interrupted_tails/`，压缩档案原子替换；只读核对不修改原件。恢复事实与代码哈希在 `recovery.jsonl`。

## 本地模型与排队

本地后端为 `http://127.0.0.1:8001/v1`、`Qwen3.8-27B`，tmux `qwen3_8-27b-mtp`，3 槽。健康检查：

```bash
curl -s --noproxy '*' 127.0.0.1:8001/health
uv run python -m experiments.infra.local_queue --method funsearch --suite co6 --batch <批次> --slots 3
```

服务启动参数见 `~/models/gguf/qwen3.8-27b/update_ud_q4_k_xl.sh`。本地评价使用物理性能核 `0,2,4,6,8,10,12,14`；模型线程用 `taskset -a -cp 16-31 <pid>` 固定到能效核，重启后重新绑定。排队器只在内存跟踪在跑任务，中途退出后等现有任务结束再重启。旧实验见[归档目录](../../archive/local_models_20261011/README.md)。
