# 实验操作

运行前核对[实验条件](../PROTOCOL.md)。本页的主机与服务位置用于查找；实时状态需要检查进程、日志和健康接口。

## 查询状态

命令在仓库根目录执行。将 `MANIFEST_PATH` 替换为批次清单路径，将 `RUN_DIR` 替换为运行目录。

```bash
# 本地查询
uv run python -m experiments.infra.batch_status \
  --manifest MANIFEST_PATH --details

# 远端查询：清单路径按远端位置填写
uv run python -m experiments.infra.batch_status \
  --ssh B3-server3 --repo /home/fzy/code/LLM4AD/TraceAAD \
  --manifest MANIFEST_PATH --details

# 本地看板
uv run python -m experiments.visualization --host 127.0.0.1 --port 8765
```

| 输出或选项 | 含义 |
| --- | --- |
| `--json` | 输出结构化状态。 |
| `--wait-seconds 45` | 最多等待 45 秒，检查状态变化。 |
| `recorded` | 找到记录，不等于已确认进程仍在运行。 |
| `stale` | 需要进一步检查进程。 |
| `unknown` | 运行目录存在，但没有可用的摘要或事件。 |
| `absent` | 本机没有该运行目录。可能未在本机启动，也可能远端已清理。远端清理后，应读取本地同步的清单与结果。 |

`heldout` 列只统计主比较测试（`PRIMARY_SPLITS`）。`missing` 表示主测试缺失，会影响 `ready_for_heldout` 与需要关注的判断。

看板地址为 `http://127.0.0.1:8765/`。本地数据只反映已同步的状态。生成预算读取事件与恢复点，不用有效程序数代替。文件说明见[结果格式](SEARCH_FORMAT.md)。

## server3 入口

| 对象 | 位置 |
| --- | --- |
| SSH | `B3-server3` |
| 仓库 | `/home/fzy/code/LLM4AD/TraceAAD` |
| Python | 仓库内 `.venv/bin/python`；保留该路径，不解析符号链接。 |
| CPU 调度器 | 原独立调度器已停；迁移批次使用 server4 的公共调度器。 |
| 实验数据 | 仓库下 `experiments_result/`。 |

远端运行设置 `HF_HUB_OFFLINE=1`，凭据从远端环境加载。连接配置用 `ssh -G B3-server3` 检查。重启调度器前确认旧进程已退出；结束 tmux 后可能仍有遗留 socket。

## server4 入口

server4 统一执行迁移批次的搜索与 CPU 评价，并调度 server3、server1 和本机的模型请求。仓库位于 `/home/fzy/code/LLM4AD/TraceAAD`，Python 为仓库内 `.venv/bin/python`。模型鉴权由该目录的 `.env` 加载。

```bash
ssh B3-server4
cd /home/fzy/code/LLM4AD/TraceAAD
source .env.server4
uv run --frozen python -m experiments.infra.scheduler --status
uv run --frozen python -m experiments.traceaad.run \
  --task tsp_construct --backend server3 \
  --scheduler-socket "$TRACEAAD_SCHEDULER_SOCKET" \
  --eval-workers 8 --dry-run
```

`.env.server4` 设置 CPU 调度器 `TRACEAAD_SCHEDULER_SOCKET`、模型调度器 `TRACEAAD_MODEL_SCHEDULER_SOCKET`、离线模式及单线程 BLAS。两个变量均指向 `/tmp/traceaad-1003/scheduler.sock`。调度器在 tmux `traceaad_scheduler` 中运行；资源池以 `--status` 的实时输出为准。TraceAAD 使用 `--scheduler-socket`，基线入口通过上述环境变量接入。

当前资源池为 128 个物理 CPU 核和 27 个模型请求槽位：server3 的 8000、8001 端点各 9 个，server1 的 8080 端点 6 个，本机 GGUF 端点 3 个请求名额。10月25日按用户要求恢复为 3 个独立 KV 槽，与公共调度器的 3 个请求名额一致；公共调度器未重启。原机器上的实验驱动、队列及独立调度器已停止，模型服务继续运行。用户已要求所有 GPU 来源等价：每次模型请求使用任意空闲端点，不绑定实验原来的机器、模型别名或量化；单个端点仍受自己的槽位上限约束。CPU 评价也共用同一资源池，单次 `eval_workers` 是请求上限，不是独占核心。其他绕过公共调度器的调用不受该池限制。

本机通过 systemd 用户服务 `traceaad-local-model-tunnel.service` 建立反向 SSH 隧道，将 server4 的 `127.0.0.1:18001` 转发到本机 `127.0.0.1:8001`。模型由 `qwen38-llama.service` 管理：关闭 MTP、关闭共享 KV、总上下文 147456、三个并发槽（单路 49152 tokens），超额请求在服务端排队。模型异常退出自动重启，隧道断开自动重连。使用 `systemctl --user status qwen38-llama.service traceaad-local-model-tunnel.service` 查询；配置位于本机 `~/.config/systemd/user/`，模型日志为 `/tmp/llama-qwen38-recovered.log`。原两个 tmux 会话已由这些服务替代。server4 的两张 GPU 尚未部署模型服务。

迁移控制器在 server4 的 tmux `traceaad_migration_controller` 中运行。原 139 项计划已加入 TraceAAD 12.1 从零重跑的 18 路，控制器共管理 157 项。控制器仅保留 `shared` 一个组，当前最多运行 35 路实验；其余保持 `pending`，一路搜索及主测试结束后补入一路。上限从 37 收到 35 时不打断既有实验，先自然收敛，再补位。各路穿插 CPU 评价，因此搜索进程数可以超过模型槽位数；模型请求最多同时 27 个，CPU 实例最多同时 128 个。

补齐批次为 `20261022_server4_backfill_co6`：MCTS-AHD、DeltaEvolve 各六任务的种子 2；HSEvo、CALM、PathWise、ShinkaEvolve、RefineEvo 各六任务三路。源码冻结于 `TraceAAD_runs/20261022_server4_baseline_backfill`，配置预览、校验值、入队和启动验证位于 `experiments_result/queues/20261022_baseline_backfill/`。HSEvo 原取消批次的六路部分记录已按用户要求从本机与 server1 删除，新十八路从零开始。

TraceAAD 12.1 重跑批次为 `20261025_server4_qwen_co6_restart`，结果目录为 `traceaad_v12_1`：六任务各运行种子 0、1、2，每路 1000 次预算。使用原版设计核验、匹配和补全机制，源码为 `TraceAAD_runs/20261025_server4_traceaad_v12_1`；公共调度、内存保护、任务说明及最终选择按本次保存的统一条件执行。原 Qwen 十二路及批次清单已按用户要求从本机与 server3 删除，没有保留结果备份。新批次清单、十八份配置预览、源码校验、条件适配、删除回执及真实启动证据位于 `experiments_result/queues/20261025_traceaad_v12_1_restart/`。十路曾在搜索开始前遇到调度器连接拥挤，随后按三秒间隔错峰启动，没有沿用旧十二路的任何搜索状态。

需要 embedding 的三种方法复用 server3 的 `qwen3-embedding-4b-fp8`。server4 的 tmux `traceaad_embedding_tunnel` 将本机 `127.0.0.1:18002` 转发到 server3 的 `127.0.0.1:8002`；专用 SSH 密钥只允许该转发。`.env.server4` 的 `SHINKA_EMBEDDING_*` 指向此接口。隧道需保持运行；重启命令保存在补齐记录的 `embedding_tunnel_command.txt`。该服务与生成模型的 27 请求槽分开。

进度和恢复依据见 `experiments_result/migrations/20261022_server4/plan.json`、`recovery_checks.json` 与 `controller.log`。首次扩容见 `experiments_result/scaleup/20261022_fill27/`；随后解除来源绑定的停点、旧结果包、新源码及验证见 `experiments_result/reschedule/20261022_shared_pool/`。切换时保留了 1 路已完成结果，36 路从各自停点续跑。

`run_config.json` 的 `model_routing`、`routing_history` 记录共享池生效边界；原 `llm` 字段保留名义配置。后续条件应按逐请求的 `scheduler.endpoint`、`scheduler.model` 和实际 `model` 读取，不能再把整路解释为纯 GGUF 或纯 AWQ。本机 tmux `traceaad_server4_results_sync` 每 60 秒拉取十四批结果和调度记录，不清理远端数据。控制器完成全部搜索及主测试后，同步进程再补拉一轮各批结果再退出。

服务器重启后，确认旧调度器已退出，再执行 `bash experiments_result/migrations/20261022_server4/scheduler_command.txt`。实验恢复前还需核对控制器与各运行锁。环境初始验证保存在 `experiments_result/environment_setup/20261022_server4/`；该目录的旧 17 槽命令已被迁移命令替代。

## 同步与清理

将 `METHOD_NAME` 替换为实际结果目录名，将 `BATCH_NAME` 替换为批次名。当前 TraceAAD 默认结果目录名为 `traceaad`，历史批次按原目录填写。

```bash
uv run python -m experiments.infra.remote pull \
  --experiment METHOD_NAME --batch BATCH_NAME

uv run python -m experiments.infra.remote sync \
  --experiment METHOD_NAME --batch BATCH_NAME
```

上述命令默认指向 server3。迁移后的四批使用 `--host B3-server4`；仓库路径不同时再指定 `--repo`。

| 动作 | 实际作用 |
| --- | --- |
| `pull` | 拉取一次批次结果。 |
| `sync` | 默认每 60 秒拉取一次；清单内所有搜索达到终止状态后停止。失败终止也算终止。 |
| `push` | 镜像本地代码，删除远端多余代码；保留排除的环境、结果与 `archive/`。 |
| `prune` | 先拉取，再按完成状态、测试文件存在性和校验结果删除远端运行目录。 |
| `watch` | 循环拉取并执行 `prune`。 |

同步代码的命令为：

```bash
uv run python -m experiments.infra.remote push
```

远端搜索使用公共源码时，不要覆盖它。`push` 检测到依赖文件变化后，会在远端执行 `uv sync --frozen`。具体排除与同步规则见 [remote.py](remote.py)。

需要清理时才使用 `prune` 或 `watch`。默认检查 `heldout.json` 存在，但不验证测试内容；`--no-heldout` 会跳过存在性检查。删除前使用 rsync 校验远端文件是否已在本地一致保存，排除 `.cache/`。这不是对结果科学有效性的检查。

增长中的日志副本仍可能变化。复查实验时，以已提交的事件和对应配置为准。

## 恢复搜索

先只读检查基线的恢复条件，再加 `--execute` 续跑：

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m experiments.infra.resume_baseline \
  --run-dir RUN_DIR

HF_HUB_OFFLINE=1 .venv/bin/python -m experiments.infra.resume_baseline \
  --run-dir RUN_DIR --execute
```

支持的方法与恢复边界见 [resume_baseline.py](resume_baseline.py)。CALM、PathWise、ShinkaEvo 和 DeltaEvolve 读取完整检查点；其他基线从当前日志中保存的种群、树或数据库记录恢复。日志恢复不还原完整随机状态。FunSearch 使用已记录的逐实例评分；缺失时停止恢复，不补评旧数据。DeltaEvolve 保存待评价程序与待完成 embedding 节点；恢复不重新生成待评价程序，也不重复评价已提交的待 embedding 节点。

TraceAAD 使用原入口、原目录和相同参数读取恢复点。操作前核对源码与配置，确认旧进程已退出。历史运行的评价规则以对应源码快照为准。恢复继续已提交的状态；最好程序不能重建缺失的搜索状态。锁与日志尾部由现有入口处理。

## 模型与计算资源

后端标签、端点和请求容量见 [base.py](base.py) 的 `BACKENDS`。通过现有启动器分配请求槽，不在文档中复制容量表。

提供调度器 socket 时，实例评价申请 CPU 核心；未提供时，按配置的 worker 数并发执行。DeepSeek 直接请求网关，提供 socket 时仍可使用调度器分配 CPU。

| 资源 | 配置线索 |
| --- | --- |
| 本机常用 CPU 核 | `0,2,4,6,8,10,12,14`。 |
| server3 常用物理核 | `0–51`。 |
| 本地生成服务 | `http://127.0.0.1:8001/v1`；`systemctl --user restart qwen38-llama.service`。 |

实际可用资源需要现场核对。本地基线排队入口为 `experiments.infra.local_queue`，先用 `--dry-run` 查看计划。它只在内存跟踪任务，跳过已有运行目录与活动会话，不自动创建重试目录。失败运行通过恢复入口处理。排队器中断后，先检查已启动的任务，再重新排队。

## 历史数据与修复

旧实验数据保存在本机 `archive/local_models_20261011/`。需要核对指定历史事实时，再定向读取其中的元数据与转换记录。分析或恢复前复制到新位置；已有硬链接在写入前需要分离共享 inode。

相对评分转换前的原件通常位于运行目录的 `logs/raw_search_originals/`。仅有平均相对改进，不能精确还原原始目标均值。`score_source` 区分换算与重新评价；放宽 CPU 或墙钟限制的结果保留其条件。

旧格式转换、评分换算和专项归档脚本已清理。相关历史结果与处理记录保留在对应实验目录中，当前读取器按程序个体编号读取结果。
