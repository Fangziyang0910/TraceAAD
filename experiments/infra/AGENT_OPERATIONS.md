# 实验操作

运行前核对[实验条件](../PROTOCOL.md)。本页的主机与服务位置用于查找；实时状态需要检查进程、日志和健康接口。

北京时间11月16日00:05，早期V3冻结批次`20261116_000118_evotrace_v3_full_co4`（提交 `a28b1ed7` 快照）在server4启动，四任务TSP、CVRP、FSSP、OP各seed0/1/2，共12路，每路1000次实际评价。结果目录`evotrace_v3_full`，源码冻结于`/home/fzy/code/LLM4AD/TraceAAD_runs/20261116_000118_evotrace_v3_full_co4`（见[启动核验](../../experiments_result/evotrace_v3_full/launch_20261116_000118_evotrace_v3_full_co4/README.md)）。代表覆盖版批次 `20261116_165500_evotrace_v3_rep_cover_co4`（提交 `d3cbb242` 机制）使用减法前的反思提示与生成上下文，论文第3节与附录F的数据目前来自该批次；该批次于北京时间11月17日按用户要求停止，TSP、CVRP 各路停在 797–874 次评价，FSSP、OP 已先跑满 1000 次（停止记录 `experiments_result/evotrace_v3_rep_cover/stop_20261117.json`，在 server4）。11月17日，主线经[V3减法](../../docs/记录/2026-11-17-EvoTrace-V3减法.md)（提交 `1e6552b3`）后，精简版批次 `20261117_132027_evotrace_v3_lean_co4` 在server4启动，后因反思写成规则、Explore 挂接错误于同日停止并作废（每路 91–137 次评价）。随后按[事实反思](../../docs/记录/2026-11-17-EvoTrace-V3事实反思.md)修订（源码已随 `190fe372` 合入主线），批次 `20261117_153134_evotrace_v3_factual_co4` 于北京时间11月17日15:32在server4启动（结果目录 `evotrace_v3_factual`，源码冻结于 `/home/fzy/code/LLM4AD/TraceAAD_runs/20261117_153134_evotrace_v3_factual_co4`），跑至每路 253–321 次评价（共 3567 次）后按用户要求停止（停止记录 `experiments_result/evotrace_v3_factual/stop_20261117.json`）。同日18:19，完成根反思存活基准锚点、`Explore` 剥离局部谱系经验及分配去工程化（未提交）后，定型版批次 `20261117_181830_evotrace_v3_clean_co4` 在server4启动，四任务各seed0/1/2，共12路，每路1000次实际评价；结果目录 `evotrace_v3_clean`，源码冻结于 `/home/fzy/code/LLM4AD/TraceAAD_runs/20261117_181830_evotrace_v3_clean_co4`，为当前默认看板实验。同日的 `20261117_114412_evotrace_v3_context_co4`（结果目录 `evotrace_v3_context`）已停止并作废，不作对照。

当前保留的主对照基线批次：
- **EvoTrace V1**：Qwen批次 `20261112_165634_evotrace_v1_co4`（四任务12路）与 DeepSeek 批次 `20261112_165634_evotrace_v1_deepseek_none_co4`（四任务各seed0），源码快照 `/home/fzy/code/LLM4AD/TraceAAD_runs/20261112_165634_evotrace_v1_co4`，均已完成搜索与独立测试（[启动核验](../../experiments_result/evotrace_v1/launch_20261112_165634_evotrace_v1_co4/latest_verification.json)）。
- **TraceAAD Base**：详见下方“正式 Base 批次”。
- 已停止的中间迭代批次（`evotrace_v3_revised`、`evotrace_v3` 及各历史 `evotrace_v2*`）均已退出进程并移出活动同步名单，原始结果目录与 `stop_*.json` 保留于本地 `experiments_result/`，阶段记录封存于本地 `archive/`。

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
uv run python -m experiments.visualization --host 0.0.0.0 --port 8765
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

看板地址为 `http://10.150.68.218:8765/`，本机也可访问 `http://127.0.0.1:8765/`。本地数据只反映已同步的状态。生成预算读取事件与恢复点，不用有效程序数代替。文件说明见[结果格式](SEARCH_FORMAT.md)。

看板展示由主实验名单中的`visualization_methods`单独控制；当前默认展示“EvoTrace V3”的四任务、三种子12路实验，同时保留V1、V2及其他历史入口。`visualization_reference`指定Base为默认比较参照。展示范围不删除历史运行，也不改变同步/检查使用的`methods`清单。浏览器时间统一为北京时间，刷新提示表示读取本地同步副本。

本机看板由 systemd 用户服务 `traceaad-dashboard.service` 启动；结果同步由 `traceaad-results-sync.service` 按[主实验名单](../../experiments_result/primary_runs.json)每60秒从server4拉取。两个服务开机后自动启动。

`traceaad-watch.timer` 每三分钟运行 `python -m experiments.infra.watch_server4`。它不写批次名：server4 上有清单、且主实验名单中有运行在本机尚未完成（搜索未到终止状态，或已完成但缺 `heldout.json`）的批次，就是要检查的批次。新批次的运行加入主实验名单后自动纳入，被替换的批次随之移出；不在名单中的批次（如消融）不检查。每次检查：失败的 `traceaad-*` 用户服务、本机模型、server4 到本机模型的转发、tmux 会话、各批次的远端状态，以及本机副本是否落后远端 20 次候选以上。新异常通过本机桌面通知；最近一次检查保存在 `~/.cache/traceaad/server4-watch.json`。同一模块的 `sync` 动作（`python -m experiments.infra.watch_server4 sync`）按同样的规则每 60 秒同步这些批次，可以用一个服务替代按版本的同步服务。查询示例：

```bash
systemctl --user status traceaad-dashboard.service traceaad-results-sync.service
systemctl --user status traceaad-watch.timer
journalctl --user -u traceaad-watch.service -n 5 --no-pager
```

## 正式 Base 批次（11月9日纯 Boltzmann 版）

当前默认比较参照为 `20261109_base_boltzmann_co6`（结果目录 `experiments_result/traceaad_base_boltzmann_20261109`，清单 [`batch_20261109_base_boltzmann_co6.json`](../../experiments_result/traceaad_base_boltzmann_20261109/batch_20261109_base_boltzmann_co6.json)），源码冻结于 `/home/fzy/code/LLM4AD/TraceAAD_runs/20261109_traceaad_base_v23_2`。该批次采用当前 Base 机制（`static_algorithm_context_v1` + `boltzmann_fitness_v1`，`selection_temperature=1.0`，每路 1000 次评价，JSON Schema 严格约束 `analysis`、`design`、`code` 三个非空字符串），四任务（TSP、CVRP、FSSP、OP）× 3 种子共 12 路均已完成搜索与独立测试。更早的 10 月 31 日排名折扣版 Base（`20261031_base_schema_co6`）保留在 `primary_runs.json` 的 `methods.traceaad_base` 历史六任务清单中，不再作为当前看板默认参照。


## server1 GPU1：llama.cpp

11月15日起，server1 的 GPU1（RTX 4080 SUPER 32 GB，UUID `GPU-c875982a-4776-123f-64f2-1c8bcae78de6`）上的 8080 端点由原 vLLM（Qwen3.8 AWQ，6 槽）改为 llama.cpp v1.0.0，配置与本机一致：`Qwen3.8-27B-UD-Q4_K_M.gguf`、非思考模式、MTP（`--spec-draft-n-max 4`）、主/草稿 KV 均为 `f8_e4m3`，采样温度0.7、top-p 0.8、top-k 20、min-p 0、presence 1.5；3 槽、每槽65536（总196608）。服务名 `Qwen3.8-27B`，API key 沿用 `~/vllm031/api_key.txt`（`--api-key-file`）。显存占用约26.7 GB。

| 对象 | 位置 |
| --- | --- |
| 启动脚本 | server1 `~/llama/serve_llama_gpu1.sh`，tmux 会话 `llama_gpu1`，失败后10秒重试 |
| 日志 | `~/logs/llama-q38-gpu1-8080.log` |
| 程序与源码 | `~/app/llama-v1.0.0`（安装）、`~/app/llama.cpp`（源码，CUDA 12.6、架构89编译） |
| 模型文件 | `~/models/gguf/qwen3.8-27b-q4km/`（主模型与 `mtp-Qwen3.8-27B-Q4_0.gguf`） |
| 原 vLLM | 启动脚本 `~/vllm031/serve_gpu1.sh` 保留；该会话 `qwen38_gpu1` 已停止 |

实测（代码生成提示）：单请求约85 tok/s，3并发每路约58–74 tok/s；三路同时60124 token 提示均正常返回。GPU1 余约6 GB，尚未试 `--spec-draft-n-max` 大于4。该端点按 `Qwen3.8-27B` 提供服务，请求中的模型名不影响 llama.cpp 路由。11月15日已重启 server4 调度器（命令见 `experiments_result/migrations/20261115_server1_llama/scheduler_command.txt`）：server1 改为 3 槽、本机端点改为 2 槽，server3 两个端点各 9 槽不变，并从 server4 对 server1 与本机端点各完成一次真实请求。同日把调度器监听队列从 socketserver 默认的 5 调为 1024（`core/scheduling.py` 的 `request_queue_size`）：此前同一瞬间超过约 5 个连接会收到 `Resource temporarily unavailable`，而 `SchedulerError` 不在 `generate` 的可重试范围内，该次生成会直接失败。调整后在 server4 上用 `ScheduledLLM`（每线程一个实例）做突发测试：60 个生成请求与 100 个 CPU 申请同一瞬间到达，160 个连接全部成功，四个端点都被用到，同时在途数均未超过各自槽数（9、9、3、2），结束后调度器无残留任务。调度器进程的真实参数以 `--status` 为准；tmux 服务器进程的命令行仍显示旧参数（`server1=6`），只是进程标题，不代表当前配置。

## server3：官方 FP8（11月15日）

server3 的 GPU0、GPU1 上的 8000、8001 端点由 AWQ-INT4 改为官方 `Qwen/Qwen3.8-27B-FP8`（`~/models/Qwen3.8-27B-FP8`），vLLM 0.31.0，服务名 `qwen3.8-27b-fp8`。启动脚本 `~/vllm031/serve_fp8.sh <gpu> <port> <mtp> <显存比例> <最大并发序列>`，当前为 `0 8000 2 0.97 8` 与 `1 8001 2 0.97 8`，tmux 会话 `vllm031_fp8_gpu0_8000`、`vllm031_fp8_gpu1_8001`，日志 `~/logs/vllm031-q38-fp8-<端口>.log`。参数：仅语言模型、`--max-model-len 49152`、`--kv-cache-dtype fp8`、MTP 2 个草稿 token、显存比例 0.97。原 AWQ 脚本 `~/vllm031/serve.sh` 保留。

嵌入服务 `qwen3-embedding-4b-fp8`（端口 8002）从 GPU0 移到 GPU3（`~/vllm031/serve_embedding_fp8_gpu3.sh`，tmux 会话 `vllm031_embedding_fp8_gpu3_8002`），与用户 `wkz` 的训练任务共用该卡；server4 隧道与接口不变，已用 server4 的隧道验证。GPU2、GPU3 属共享资源，随时可能被他人占用。

vLLM 启动日志给出每卡 KV 空间 14.43 GiB、317,773 个 token，48K 上下文下最坏并发 6.47 倍，据此每路设 6 槽。权重加载占 28.49 GiB，激活峰值 2.85 GiB。此前按 AWQ 日志的每 token 显存估算为 8 倍，高估；实测每 token 约 47.6 KiB，AWQ 日志为约 36.6 KiB。

实测（代码生成提示、非思考）：单请求约 61 tok/s，6 并发每路 38–63 tok/s，MTP 接受率约 91%；6 个同时 43063 token 的提示在 26–30 秒内全部答对，无抢占。调度器已重启（命令见 `experiments_result/migrations/20261115_server3_fp8/scheduler_command.txt`），server4 上的 `base.py` 同步改为 FP8 模型名与 6 槽，突发测试（60 个生成请求与 100 个 CPU 申请同一瞬间）全部成功，各端点同时在途数未超过槽数。AWQ 与 FP8 的输出质量与速度尚未在同一批提示上对比。

## server3 入口

| 对象 | 位置 |
| --- | --- |
| SSH | `B3-server3` |
| 仓库 | `/home/fzy/code/LLM4AD/TraceAAD` |
| Python | 仓库内 `.venv/bin/python`；保留该路径，不解析符号链接。 |
| CPU 调度器 | 原独立调度器已停；搜索使用 server4 的公共调度器。 |
| 实验数据 | 仓库下 `experiments_result/`。 |

远端运行设置 `HF_HUB_OFFLINE=1`，凭据从远端环境加载。连接配置用 `ssh -G B3-server3` 检查。重启调度器前确认旧进程已退出；结束 tmux 后可能仍有遗留 socket。

## server4 入口

server4 统一执行搜索与 CPU 评价，并调度 server3、server1 和本机的模型请求。仓库位于 `/home/fzy/code/LLM4AD/TraceAAD`，Python 为仓库内 `.venv/bin/python`。模型鉴权由该目录的 `.env` 加载。

```bash
ssh B3-server4
cd /home/fzy/code/LLM4AD/TraceAAD
source .env.server4
uv run --frozen python -m experiments.infra.scheduler --status
.venv/bin/python -m experiments.traceaad.run \
  --method base --task tsp_construct --run-name base_dry --batch base_dry \
  --scheduler-socket "$TRACEAAD_SCHEDULER_SOCKET" --env-file .env.server4 --dry-run
```

`.env.server4` 设置 CPU 调度器 `TRACEAAD_SCHEDULER_SOCKET`、模型调度器 `TRACEAAD_MODEL_SCHEDULER_SOCKET`、离线模式及单线程 BLAS。两个变量均指向 `/tmp/traceaad-1003/scheduler.sock`。调度器在 tmux `traceaad_scheduler` 中运行；资源池以 `--status` 的实时输出为准。TraceAAD 使用 `--scheduler-socket`，基线入口通过上述环境变量接入。

当前资源池为 128 个物理 CPU 核和 17 个模型请求槽位：server3 的 8000、8001 端点（官方 FP8）各 6 个，server1 的 8080 端点（llama.cpp）3 个，本机 GGUF 端点 2 个。原机器上的实验驱动、队列及独立调度器已停止，模型服务继续运行。所有 GPU 来源等价：每次模型请求使用任意空闲端点，不绑定实验原来的机器、模型别名或量化；单个端点仍受自己的槽位上限约束。CPU 评价也共用同一资源池，单次 `eval_workers` 是请求上限，不是独占核心。TraceAAD/EvoTrace 的整任务评价（`experiments/traceaad/parallel_evaluation.py`）每次最多申请 8 个核，实例并行，总时限按实例秒之和计量；申请时只取当前空闲的核，至少 1 个。其他绕过公共调度器的调用不受该池限制。

本机通过 systemd 用户服务 `traceaad-local-model-tunnel.service` 建立反向 SSH 隧道，将 server4 的 `127.0.0.1:18001` 转发到本机 `127.0.0.1:8001`。模型由 `qwen38-llama.service` 管理（配置位于 `~/.config/systemd/user/`，日志 `/tmp/llama-qwen38-recovered.log`），异常退出自动重启，隧道断开自动重连，用 `systemctl --user status qwen38-llama.service traceaad-local-model-tunnel.service` 查询。当前服务运行 llama.cpp v1.0.0（build 12359，安装在 `~/app/llama-v1.0.0`，由 `LD_LIBRARY_PATH` 加载），加载 unsloth `Qwen3.8-27B-UD-Q4_K_M.gguf`（`~/models/gguf/qwen3.8-27b-q4km/`），非思考模式（`--chat-template-kwargs '{"enable_thinking":false}'`），开启 MTP（`--spec-type draft-mtp`，草稿模型 `mtp-Qwen3.8-27B-Q4_0.gguf`，`--spec-draft-n-max 4`），主 KV 与草稿 KV 均为 `f8_e4m3`，`--load-mode mlock`，并发 2 槽、每槽 65536 tokens（总 131072），显存约 23.8 GB（24 GB 卡）；实测单请求约 141 tok/s，2 并发每路约 118 tok/s，两路同时 63523 token 提示可正常处理。server4 的两张 GPU 尚未部署模型服务。

需要 embedding 的基线复用 server3 的 `qwen3-embedding-4b-fp8`。server4 的 tmux `traceaad_embedding_tunnel` 将本机 `127.0.0.1:18002` 转发到 server3 的 `127.0.0.1:8002`；专用 SSH 密钥只允许该转发。`.env.server4` 的 `SHINKA_EMBEDDING_*` 指向此接口。隧道需保持运行；重启命令保存在补齐记录的 `embedding_tunnel_command.txt`。该服务与生成模型的 17 个请求槽分开。

`/tmp/tmux-1003` 存 tmux socket，`/tmp/traceaad-1003` 存调度器 socket，清理 `/tmp` 时不能删。tmux socket 丢失后，服务进程仍在：先重建目录（权限700），再向 `tmux: server` 进程发 `SIGUSR1`。

服务器重启后，确认旧调度器已退出，再执行 `bash experiments_result/migrations/20261115_server3_fp8/scheduler_command.txt`。实验恢复前还需核对各运行锁。环境初始验证保存在 `experiments_result/environment_setup/20261022_server4/`；该目录的旧 17 槽命令已被迁移命令替代。


## 同步与清理

将 `METHOD_NAME` 替换为实际结果目录名，将 `BATCH_NAME` 替换为批次名。结果目录名按批次清单中的 `experiment` 填写。

```bash
uv run python -m experiments.infra.remote pull \
  --experiment METHOD_NAME --batch BATCH_NAME

uv run python -m experiments.infra.remote sync \
  --experiment METHOD_NAME --batch BATCH_NAME
```

上述命令默认指向 server3。server4 上的批次使用 `--host B3-server4`；冻结快照中的批次再用 `--repo` 指定快照路径。

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

TraceAAD 使用原入口、原目录和相同参数加 `--resume` 续跑。操作前确认旧进程已退出。历史运行按其源码快照的规则恢复。

## 模型与计算资源

后端标签、端点和请求容量见 [base.py](base.py) 的 `BACKENDS`。通过现有启动器分配请求槽，不在文档中复制容量表。

提供调度器 socket 时，实例评价申请 CPU 核心；未提供时，按配置的 worker 数并发执行。

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
