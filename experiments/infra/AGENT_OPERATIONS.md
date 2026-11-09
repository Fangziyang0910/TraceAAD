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

看板展示由主实验名单中的`visualization_methods`单独控制，固定为正式Base（11月5日）、V21、V22、V23与12个对照方法；默认监控V22，`visualization_reference`指定Base为默认比较参照。展示范围不删除历史运行，也不改变同步/检查使用的`methods`清单。浏览器时间统一为北京时间，刷新提示表示读取本地同步副本。

本机看板由 systemd 用户服务 `traceaad-dashboard.service` 启动；结果同步由 `traceaad-results-sync.service` 按[主实验名单](../../experiments_result/primary_runs.json)每60秒从server4拉取。两个服务开机后自动启动。

`traceaad-watch.timer` 每三分钟运行 `python -m experiments.infra.watch_server4`。它不写批次名：server4 上有清单、且主实验名单中有运行在本机尚未完成（搜索未到终止状态，或已完成但缺 `heldout.json`）的批次，就是要检查的批次。新批次的运行加入主实验名单后自动纳入，被替换的批次随之移出；不在名单中的批次（如消融）不检查。每次检查：失败的 `traceaad-*` 用户服务、本机模型、server4 到本机模型的转发、tmux 会话、各批次的远端状态，以及本机副本是否落后远端 20 次候选以上。新异常通过本机桌面通知；最近一次检查保存在 `~/.cache/traceaad/server4-watch.json`。同一模块的 `sync` 动作（`python -m experiments.infra.watch_server4 sync`）按同样的规则每 60 秒同步这些批次，可以用一个服务替代按版本的同步服务。查询示例：

```bash
systemctl --user status traceaad-dashboard.service traceaad-results-sync.service
systemctl --user status traceaad-watch.timer
journalctl --user -u traceaad-watch.service -n 5 --no-pager
```

## 正式 Base 批次（10月31日）

`20261031_base_schema_co6` 在server4从零启动18路，每路1000次评价。生成与修复使用明确的JSON Schema，要求 `analysis`、`design`、`code` 三个字符串并禁止额外字段；非空要求由解析器检查。提示、历史窗口、父代选择、算子、数据与评价时限沿用分析轨迹批次。

源码冻结于 `/home/fzy/code/LLM4AD/TraceAAD_runs/20261031_traceaad_base_schema_final`，609个文件中7个变化、602个一致，远端60项检查通过。结果入口为 `experiments_result/traceaad_base`（server4主仓库中的符号链接，指向快照内的 `traceaad_base_schema` 目录），看板和同步使用该批次。它使用11月5日修改前的 Base 代码。原 `traceaad_base`（`20261030_base_compact_co6`）改名为 `traceaad_base_compact`；此前的记录中，`experiments_result/traceaad_base` 指紧凑批次。上一批分析实验的18路已正常停止并保留结果；更早批次继续按各自状态运行。[启动记录](../../experiments_result/traceaad_base/launch_20261031/README.md)保存配置、源码核对、停止与启动证据。


## Base 与 V21 对照批次（11月5日）

`20261105_base_co6` 与 `20261105_v21_co6` 于北京时间11月5日13:00在server4同一快照、同一调度器下启动，各18路，每路1000次评价，训练后自动做独立主测试。源码冻结于 `/home/fzy/code/LLM4AD/TraceAAD_runs/20261105_traceaad_base_v21`，结果入口为 `experiments_result/traceaad_base_20261105` 与 `experiments_result/traceaad_v21`（server4主仓库中的符号链接）。

`20261105_v21_co6` 于13:41停止，共815次评价，不再使用。`20261105b_v21_co6` 使用精简后的V21，于13:45启动18路，继续以 `20261105_base_co6` 为对照。源码冻结于 `/home/fzy/code/LLM4AD/TraceAAD_runs/20261105b_traceaad_v21`，结果入口为 `experiments_result/traceaad_v21b`。

`20261105b_v21_co6` 于17:33停止，共3971次评价，不再使用。`20261105c_v21_co6` 使用按算子分配材料的V21，继续以 `20261105_base_co6` 为对照。源码冻结于 `/home/fzy/code/LLM4AD/TraceAAD_runs/20261105c_traceaad_v21`，结果入口为 `experiments_result/traceaad_v21c`。

## V22 正式批次（11月7日）

旧实现已删除。下文保留历史启动与结果位置，复查和恢复使用该批次冻结源码。

`20261107_v22_co6`于北京时间11月7日13:09在server4启动，六任务×种子0、1、2，共18路，每路1000次评价，初始化5个有效程序，搜索后自动执行独立主测试。V22为V21加机制1，包含预测尺度±3倍的稳健残差更新、最多29个第二步候选和严格恢复校验，不含经验、反思或嵌入。

源码冻结于`/home/fzy/code/LLM4AD/TraceAAD_runs/20261107_traceaad_v22`，结果入口为`experiments_result/traceaad_v22`（server4主仓库中的符号链接）。411份源码和文档按SHA-256核验；67份Base、V21、benchmarks与core的Python文件与V21快照逐文件一致。远端156项测试通过；六任务的预算、方法参数、任务定义、评价执行、模型采样和路由配置与`20261105c_v21_co6`一致。训练16例共享160秒，主测试50例共享500秒，内存2 GiB。

启动检查在13:10确认18路会话及搜索全部存活，均已完成真实评价，已提交62次评价，模型请求错误为0；12次已提交演化的算子、参考和尺度记录对应正确。这只是启动状态。V22已加入主实验名单，现有结果同步和看板自动读取。配置预览、源码哈希、远端测试和启动检查保存在`experiments_result/traceaad_v22/launch_20261107/`。

## V23 正式对照批次（11月7日）

旧实现已删除。下文保留历史启动与结果位置，复查和恢复使用该批次冻结源码。

`20261107_v23_co6`于北京时间11月7日13:33在server4启动18路，六任务×种子0/1/2，每路1000次评价，初始化5个有效程序，训练后自动做独立主测试。V23为Base加机制1，完整保留Base的生成流程和`analysis → design → code`输出，与V22共用选择和学习代码、分别维护状态。

源码冻结于`/home/fzy/code/LLM4AD/TraceAAD_runs/20261107_traceaad_v23`，结果入口为`experiments_result/traceaad_v23`。416份文件按SHA-256核验；62份Base、benchmarks和core的Python文件与正式Base快照一致，机制1选择器与运行中的V22逐字节一致。六任务的预算、完整方法参数、任务定义、评价、模型采样及路由与`20261105_base_co6`一致。远端167项测试通过，6条既有fork弃用提示。

13:34:49检查确认18路存活且运行，全部完成真实评价，共42次已提交评价，模型请求错误为0；4次已提交演化的动作、参考和评价计数正确，实际提示均使用Base输出顺序。V22的18路继续运行，冻结源码未改动。V23已加入主实验名单，现有同步、看板与检查服务自动识别。配置、源码、测试与启动证据见[启动记录](../../experiments_result/traceaad_v23/launch_20261107/README.md)。

## V22.1 与 V23.1 正式批次（11月8日）

`20261108_v22_1_co6` 与 `20261108_v23_1_co6` 于北京时间11月8日19:22在server4同一冻结快照启动，各六任务×种子0/1/2，共18路，每路1000次评价，初始化5个有效程序，训练后自动做独立主测试。V22.1为V21加最新版机制1，V23.1为Base加同一机制1：直接学习真实冠军改善，只在近期且质量状态可比的历史内借用，并按经验强度修正Base联合概率，单动作乘子封顶4倍。

源码冻结于`/home/fzy/code/LLM4AD/TraceAAD_runs/20261108_traceaad_v22_1_v23_1`，424份文件按SHA-256核验。结果入口为`experiments_result/traceaad_v22_1`与`experiments_result/traceaad_v23_1`（server4主仓库中的符号链接）。两批六任务配置分别与`20261105c_v21_co6`、`20261105_base_co6`逐字段差异为零。远端193项测试通过，1项进程竞态测试首跑失败、隔离复跑通过。训练16例共享160秒，主测试50例共享500秒，内存2 GiB。

19:27–19:29启动检查确认36路全部存活且running，均已完成真实评价，模型请求错误为0；已提交演化的动作、记录字段、评价计数与提示顺序通过只读核对。两批已加入本机与server4主实验名单。启动证据见`experiments_result/traceaad_v22_1/launch_20261108/`与`experiments_result/traceaad_v23_1/launch_20261108/`。

## V24 正式批次（11月9日）

`20261109_v24_co6` 于北京时间11月9日11:48在server4启动18路，六任务×种子0/1/2，每路1000次评价，初始化5个有效程序，训练后自动做独立主测试。V24为V21加机制3，对照为`20261105c_v21_co6`。反思请求经同一调度器发出，不占评价预算，token 汇总在 `summary.json` 的 `auxiliary.reflect`。

源码冻结于`/home/fzy/code/LLM4AD/TraceAAD_runs/20261109b_traceaad_v24`，结果入口为`experiments_result/traceaad_v24`（server4主仓库中的符号链接）。Base、V21、benchmarks与core的Python文件与V21正式快照逐文件一致；远端80项测试通过。同一快照的`experiments_result/replay/`保存反思回放，`20261109_traceaad_v24`快照保存旧提示的TSP冒烟。启动证据见[启动记录](../../experiments_result/traceaad_v24/launch_20261109/README.md)。

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

当前资源池为 128 个物理 CPU 核和 27 个模型请求槽位：server3 的 8000、8001 端点各 9 个，server1 的 8080 端点 6 个，本机 GGUF 端点 3 个请求名额。10月25日按用户要求恢复为 3 个独立 KV 槽，与公共调度器的 3 个请求名额一致；公共调度器未重启。原机器上的实验驱动、队列及独立调度器已停止，模型服务继续运行。用户已要求所有 GPU 来源等价：每次模型请求使用任意空闲端点，不绑定实验原来的机器、模型别名或量化；单个端点仍受自己的槽位上限约束。CPU 评价也共用同一资源池，单次 `eval_workers` 是请求上限，不是独占核心。其他绕过公共调度器的调用不受该池限制。

本机通过 systemd 用户服务 `traceaad-local-model-tunnel.service` 建立反向 SSH 隧道，将 server4 的 `127.0.0.1:18001` 转发到本机 `127.0.0.1:8001`。模型由 `qwen38-llama.service` 管理：关闭 MTP、关闭共享 KV、总上下文 147456、三个并发槽（单路 49152 tokens），超额请求在服务端排队。模型异常退出自动重启，隧道断开自动重连。使用 `systemctl --user status qwen38-llama.service traceaad-local-model-tunnel.service` 查询；配置位于本机 `~/.config/systemd/user/`，模型日志为 `/tmp/llama-qwen38-recovered.log`。原两个 tmux 会话已由这些服务替代。server4 的两张 GPU 尚未部署模型服务。

需要 embedding 的基线复用 server3 的 `qwen3-embedding-4b-fp8`。server4 的 tmux `traceaad_embedding_tunnel` 将本机 `127.0.0.1:18002` 转发到 server3 的 `127.0.0.1:8002`；专用 SSH 密钥只允许该转发。`.env.server4` 的 `SHINKA_EMBEDDING_*` 指向此接口。隧道需保持运行；重启命令保存在补齐记录的 `embedding_tunnel_command.txt`。该服务与生成模型的 27 请求槽分开。

`/tmp/tmux-1003` 存 tmux socket，`/tmp/traceaad-1003` 存调度器 socket，清理 `/tmp` 时不能删。tmux socket 丢失后，服务进程仍在：先重建目录（权限700），再向 `tmux: server` 进程发 `SIGUSR1`。

服务器重启后，确认旧调度器已退出，再执行 `bash experiments_result/migrations/20261022_server4/scheduler_command.txt`。实验恢复前还需核对各运行锁。环境初始验证保存在 `experiments_result/environment_setup/20261022_server4/`；该目录的旧 17 槽命令已被迁移命令替代。


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


## 新 V23.1 纯净 Base 对照（11月9日）

`20261109_v23_1_static_co6` 于北京时间11月9日15:37在server4启动，六任务×种子0/1/2，各1000次评价。冻结于 `/home/fzy/code/LLM4AD/TraceAAD_runs/20261109_traceaad_v23_1_static`，以11月8日共用快照为底，只调整Base演化提示和V23.1接入及生成协议识别，任务、数据、机制1与评价实现保持一致。初始化和修复保持原样，18路实际条件已核对。

旧 `20261108_v23_1_co6` 剩余6路已停止，12路完成结果与旧源码保留。主仓库 `experiments_result/traceaad_v23_1` 指向新快照的结果目录，新目录保留旧结果软链接。本机与server4主实验名单均使用新批次；原有同步与检查服务继续接管。新日志带 `static_algorithm_context_v1`，旧日志按旧冻结源码恢复。

远端102项检查通过。15:39复查18路存活且均已真实评价，共66次，模型请求错误0。V22.1与V24继续运行。详见[启动记录](../../docs/记录/2026-11-09-V23.1纯净Base对照启动.md)与[原始证据](../../experiments_result/traceaad_v23_1/launch_20261109_static/README.md)。


## 纯净 Base 正式批次与 V24 停止（11月9日）

`20261109_base_static_co6` 于北京时间11月9日15:52启动18路，各1000次评价，结果入口为 `experiments_result/traceaad_base_static_20261109`。它与新V23.1直接共用 `/home/fzy/code/LLM4AD/TraceAAD_runs/20261109_traceaad_v23_1_static`，恢复使用该快照内批次清单记录的原入口命令；当前仓库的入口重命名不改变冻结运行。六任务预览及18路实际配置逐项核对，含任务、预算、评价、模型及源码路径均一致；只由选择实现区分机制1。

V24批次 `20261109_v24_co6` 的18路已停止，结果保留。从活动同步/检查名单移出V24，保留其历史展示；新Base加入主名单并成为默认比较参照。V22.1和新V23.1继续运行。15:57复查Base18路均有真实评价，共169次，模型请求错误0。见[启动记录](../../docs/记录/2026-11-09-Base正式对照启动与V24停止.md)与[证据](../../experiments_result/traceaad_base_static_20261109/launch_20261109_static/README.md)。
