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

本机看板由 systemd 用户服务 `traceaad-dashboard.service` 启动；V16、V17 的结果同步分别由 `traceaad-results-sync@v16.service` 和 `traceaad-results-sync@v17.service` 管理。三个服务已启用，开机后自动启动，失败时自动重启。同步日志写入各批次启动目录的 `sync.log`（V16 为 `launch_20261028/`；V17 首批放弃后，`traceaad-results-sync@v17.service.d/rerun.conf` 把服务改为同步重跑批次，日志在 `launch_20261028_rerun/`）；全部搜索与主测试结束后，同步脚本正常退出。

`traceaad-watch.timer` 每三分钟运行 `python -m experiments.infra.watch_server4`。它不写批次名：server4 上有清单、且[主实验名单](../../experiments_result/primary_runs.json)中有运行在本机尚未完成（搜索未到终止状态，或已完成但缺 `heldout.json`）的批次，就是要检查的批次。新批次的运行加入主实验名单后自动纳入，被替换的批次随之移出；不在名单中的批次（如消融）不检查。每次检查：失败的用户服务、本机模型、server4 到本机模型的转发、tmux 会话、各批次的远端状态，以及本机副本是否落后远端 20 次候选以上。新异常通过本机桌面通知；最近一次检查保存在 `~/.cache/traceaad/server4-watch.json`。同一模块的 `sync` 动作（`python -m experiments.infra.watch_server4 sync`）按同样的规则每 60 秒同步这些批次，可以用一个服务替代按版本的同步服务。查询示例：

```bash
systemctl --user status traceaad-dashboard.service 'traceaad-results-sync@v16.service' 'traceaad-results-sync@v17.service'
systemctl --user status traceaad-watch.timer
journalctl --user -u traceaad-watch.service -n 5 --no-pager
```

## V19.3 轨迹反思批次（11月1日）

`20261101_v19_3_co6` 在server4启动六任务×种子0、1、2，共18路，每路1000次实际评价。使用现有Qwen共享模型池和公共调度器，初始化5个有效不同算法，生成／反思输出上限16384／2048 tokens，`trajectory`反思。上下文窗口49152，tokenizer来自实际服务模型；训练160秒、主测试500秒、内存2 GiB。每路自动完成搜索、固定训练赢家、独立主测试。

源码冻结于 `/home/fzy/code/LLM4AD/TraceAAD_runs/20261101_traceaad_v19_3_final`，110个文件经SHA-256核对。远端35项针对性检查通过；六任务真实预检完成36次评价与21次反思，均已形成经验并用于后续生成。正式18路主进程、模型回复和实际评价均已核实。公共结果入口为 `experiments_result/traceaad_v19_3`，清单为 `batch_20261101_v19_3_co6.json`。主实验名单、默认看板、每60秒同步和原有巡检已接入。配置及核验见[启动记录](../../experiments_result/traceaad_v19_3/launch_20261101/README.md)。

## 独立 Base JSON Schema 批次（10月31日）

`20261031_base_schema_co6` 在server4从零启动18路，每路1000次评价。生成与修复使用明确的JSON Schema，要求 `analysis`、`design`、`code` 三个字符串并禁止额外字段；非空要求由解析器检查。提示、历史窗口、父代选择、算子、数据与评价时限沿用分析轨迹批次。

源码冻结于 `/home/fzy/code/LLM4AD/TraceAAD_runs/20261031_traceaad_base_schema_final`，609个文件中7个变化、602个一致，远端60项检查通过。结果入口为 `experiments_result/traceaad_base_schema`，看板和同步使用该批次。上一批分析实验的18路已正常停止并保留结果；更早批次继续按各自状态运行。[启动记录](../../experiments_result/traceaad_base_schema/launch_20261031/README.md)保存配置、源码核对、停止与启动证据。

## 独立 Base 分析轨迹批次（10月31日）

`20261031_base_analysis_co6` 于北京时间09:22–09:23在server4从零启动18路，每路1000次评价。模型依次输出 `analysis`、`design`、`code`，分析随个体保存，祖先及已有子代按父子来源、改动分析与评价呈现局部改进脉络。修复同步更新分析、设计与代码；历史窗口、父代公式、算子配比和评价条件沿用上一批。

源码冻结于 `/home/fzy/code/LLM4AD/TraceAAD_runs/20261031_traceaad_base_analysis_final`，609个文件经过校验，本地和远端各59项检查通过。结果入口为 `experiments_result/traceaad_base_analysis`；本批18路于10:01正常停止，由JSON Schema批次替代，源码与结果保留。上一批 `20261031_base_prompt_co6` 的18路于09:22正常停止，无遗留进程，源码与结果保留，仍可在看板查看停止状态；更早的紧凑输出批次继续运行。本批启动及真实分析上下文证据见[启动记录](../../experiments_result/traceaad_base_analysis/launch_20261031/README.md)。

## 独立 Base 提示修订批次（10月31日）

`20261031_base_prompt_co6` 于北京时间09:00–09:01在server4启动六任务 × 种子0、1、2，共18路，09:22按用户要求停止并由分析轨迹批次替代。该批提示先生成代码，再用自足摘要说明实际计算；Refine、交叉与修复明确保留核心计算对输出的作用。每路1000次评价、初始化5个有效个体、一次修复、温度1.0、输出上限16384 tokens，训练160秒、主测试500秒。父代公式、算子配比、参考选择及历史窗口沿用紧凑输出批次。

源码冻结于 `/home/fzy/code/LLM4AD/TraceAAD_runs/20261031_traceaad_base_prompt_final`；609个文件中仅提示词和两份说明变化。公共新结果入口为 `experiments_result/traceaad_base_prompt`，原 `traceaad_base` 继续指向紧凑输出批次，旧实验继续运行。主实验名单保留旧方法并新增18路，自动同步已发现新批次，看板默认显示“TraceAAD Base · 新提示”。本地与远端各54项检查通过，18个主进程、实际请求及评价均已核实。配置、差异及证据见[启动记录](../../experiments_result/traceaad_base_prompt/launch_20261031/README.md)。

## 独立 Base 正式批次（10月30日）

10月30日重跑批次为 `20261030_base_compact_co6`，按用户要求缩短模型输出并从头重跑18路。统一要求设计说明为单段、2–3句、最多200字符，代码不含注释或文档字符串；每实例时间额度10秒，训练总时限160秒、主测试500秒。新源码冻结于 `/home/fzy/code/LLM4AD/TraceAAD_runs/20261030_traceaad_base_compact_final`，公共结果入口已切换，看板与自动同步跟随新批次。原批次18路已停止，无遗留进程，原源码、结果及本地副本保留。配置、停止与启动证据和实际输出检查见[重跑记录](../../experiments_result/traceaad_base/launch_20261030_compact/README.md)。

`20261030_base_co6` 于北京时间22:44–22:45启动六任务 × 种子0、1、2，共18路。每路1000次完整评价、5个有效初始个体，温度1.0、输出上限16384 tokens。搜索与主测试使用任务全部实例的总时限，各次评价从公共CPU池申请一个核；模型继续使用现有共享池。入口为 `experiments.traceaad_base.run` 和 `experiments.traceaad_base.launch_host`，说明见[独立Base实验](../traceaad_base/README.md)。

源码冻结于 `/home/fzy/code/LLM4AD/TraceAAD_runs/20261030_traceaad_base_final`，609个文件经过SHA-256核对。六任务真实预检各6次评价及六路主测试通过，18个正式主进程和实际请求、评价均已核实。公共结果入口 `experiments_result/traceaad_base/` 链接到冻结快照，清单为 `batch_20261030_base_co6.json`；原始记录在每路 `native/` 下，外层文件接入现有看板。主实验名单、默认看板与自动同步均已接入，旧批次继续运行。[启动记录](../../experiments_result/traceaad_base/launch_20261030/README.md)保存配置、校验与检查结果。

## V18 正式批次（10月29日）

`20261029_v18_co6` 为六任务 × 种子0、1、2，共18路，每路1000次模型请求。V18 的预算包含初始化、发散、接口故障、无效程序及重复；同龄竞赛与有限偏差搜索的实现和恢复约束见 [V18 README](../../traceaad/v18/README.md)。输出上限8192，评价 worker 上限8，其余训练与主测试条件沿用公共协议。

最终源码冻结于 `/home/fzy/code/LLM4AD/TraceAAD_runs/20261029_traceaad_v18_final`，启动前校验454个文件；全量398项测试及六任务各8次请求的真实预检通过。较早的 `20261029_traceaad_v18_frozen` 仅用于首次预检；正式运行使用 `final` 快照，其中提示已包含评价器的实际运行限制。

后续提示审计确认：该批额外展示实例数与时间预算，偏离公共提示协议。工作区已修订为 `v18-context-2`，去除这些信息并改进路径与尝试反馈，独立小样本对照保存于 `TraceAAD_runs/20261029_v18_context2_probe`。按用户重跑要求，原18路已停止，源码、结果与恢复点全部保留，本机副本在停止后通过 rsync 校验和核对。新旧提示不可混接同一恢复点，详见 [V18 README](../../traceaad/v18/README.md)。

第二批 `20261029_v18_context2_co6` 已按用户要求停止，18路恢复点与结果保留于 `/home/fzy/code/LLM4AD/TraceAAD_runs/20261029_traceaad_v18_context2_frozen`，停止后完成本机拉取及文件校验和核对。公共调度器和模型服务没有重启。该批暴露了重复起点重新取得保护、连续复制反复获配的问题，详细诊断见 [机制分析](../../experiments_result/analysis/v18_mechanism_20261029/README.md)。

当前批次为 `20261029_v18_loopfix_co6`，10月29日15:42（北京时间）从零启动六任务 × 种子0、1、2，每路1000次请求上限。源码冻结于 `/home/fzy/code/LLM4AD/TraceAAD_runs/20261029_traceaad_v18_loopfix_final`，454个文件经过 SHA-256 核对；全量422项测试和六任务各8次真实预检通过。使用 `v18-context-3`、`v18-repeat-guard-1`、`max_no_change=3`，拒绝重复源码建根，同一生成条件连续无新候选后关闭；首个有效根之前保留原16次初始化额度。计数跨状态与恢复保留，不能将旧批次检查点混接新机制。模型池、采样、同龄持平、LDS深度和评价条件保持一致。

规范结果入口为公共仓库的 `experiments_result/traceaad_v18/`，现链接到 `loopfix_final` 快照；旧结果仍在各自冻结快照及本机目录中。当前清单为 `batch_20261029_v18_loopfix_co6.json`，启动、停止旧批及核验记录在 [launch_20261029_loopfix](../../experiments_result/traceaad_v18/launch_20261029_loopfix/README.md)。每路仍自动执行“搜索 → 冻结结果检查 → 独立主测试”；冻结检查允许请求用满，或明确以 `no_eligible_generation` 提前结束，要求已有有效选择且无待处理请求。主实验名单与看板的 V18 已切换到新批次，其他方法名单不变。本机 `traceaad-results-sync.service` 每60秒同步主实验名单中尚未完成的 server4 批次，保留远端结果；该服务及原有 `traceaad-watch.timer` 均已启用。原看板仍使用 `0.0.0.0:8765`，默认显示 V18。

## V18.1 正式批次（10月29日）

`20261029_v18_1_co6` 于北京时间23:45开始启动六任务 × 种子0、1、2，共18路。每路1000次基础生成回复，额外BirthCheck独立记录成本；同龄通道比例0.25、每思想12次保护、8代后退出。初始化、算子、提示和评价沿用基础TraceAAD；不使用V18任务树、LDS或Diverge。旧V18 loopfix继续运行。

源码冻结于 `/home/fzy/code/LLM4AD/TraceAAD_runs/20261029_traceaad_v18_1_birthfix_final`，版本 `v18.1-birth-age-2`，468个文件校验。全量616项测试、最终提示194项针对性检查通过；六任务各8次生成的真实预检全部完成，48次生成及36次审核共84次模型调用，六路主测试有效。首轮预检发现出生被误解为全球原创性，原记录保留于 `20261029_traceaad_v18_1_final`，正式运行只使用明确本轮登记范围的新快照。

公共结果入口 `experiments_result/traceaad_v18_1/` 链接到新快照，清单为 `batch_20261029_v18_1_co6.json`。每路仍自动执行搜索、冻结检查、独立主测试。主实验名单加入V18.1并设为看板默认，保留其他版本；原自动同步和巡检按名单发现新批次，无须重启服务。[启动记录](../../experiments_result/traceaad_v18_1/launch_20261029/)保存源文件校验、验证、预检、清单与运行核对。

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


## V14 正式批次（10月27日）

新批次 `20261027_v14_co6` 为六任务 × 种子 0／1／2，共十八路，已全部启动。每路 1000 次候选机会、输出上限 8192、评价 worker 上限 8；V14 不设单独的模型请求上限。机制与默认值见 [V14 README](../../traceaad/v14/README.md)。

源码冻结在 `/home/fzy/code/LLM4AD/TraceAAD_runs/20261027_traceaad_v14_frozen`，启动前核对 406 个文件的 SHA-256。公共仓库的 `experiments_result/traceaad_v14/` 链接到冻结快照的结果目录。清单为 `batch_20261027_v14_co6.json`，三秒错峰，使用共享模型池与公共 CPU 调度器。每路执行“搜索 → 冻结结果检查 → 独立主测试”。启动脚本 `launch_batch.py` 需在快照根目录以 `PYTHONPATH=.` 运行，不带 `--launch` 时只写预览。

真实模型冒烟测试位于 `/home/fzy/code/LLM4AD/TraceAAD_runs/20261027_traceaad_v14_preflight`（TSP，14 次候选，停滞窗口 3，发育 2 步），确认发散提示只含设计描述、发育期以发散期最好成员为父代、结束后回到普通搜索。本机 tmux `traceaad_v14_results_sync` 每 60 秒同步。主实验名单已将 TraceAAD 切换至 V14，切换前名单保存在 `launch_20261027/primary_runs_before.json`。

## V15 正式批次与 2×2 消融（10月27日）

V15 在 V14 上加入曝光策略的两条规则，机制与开关见 [V15 README](../../traceaad/v15/README.md)。V14 继续运行，它是 2×2 消融中两条规则都关闭的一格，不停止、不替换。

| 实验目录 | 批次 | 规则 | 范围 |
| --- | --- | --- | --- |
| `traceaad_v15` | `20261027_v15_co6` | `allocation=effort`、`exposure=info` | 六任务 × 种子 0／1／2 |
| `traceaad_v15_who` | `20261027_v15_who` | `allocation=effort`、`exposure=code`（只用规则一） | TSP、CVRP、OP × 种子 0／1／2 |
| `traceaad_v15_form` | `20261027_v15_form` | `allocation=fitness`、`exposure=info`（只用规则二） | TSP、CVRP、OP × 种子 0／1／2 |

源码冻结在 `/home/fzy/code/LLM4AD/TraceAAD_runs/20261027_traceaad_v15_frozen`，启动前核对 416 个文件的 SHA-256。三个批次共用这一快照与启动脚本 `experiments_result/traceaad_v15/launch_20261027/launch_batch.py`。在快照根目录以 `PYTHONPATH=.` 运行，`--cell main|who|form` 选择批次；不带 `--launch` 时只写预览。脚本核对路数、规则组合与目录不存在后，以三秒错峰启动；每路执行“搜索 → 冻结结果检查（含 `summary.json` 的 `v15` 字段）→ 独立主测试”，主测试输出写入 `logs/heldout_primary.json`。公共仓库的 `experiments_result/<实验目录>` 链接到快照中的结果目录。每路 1000 次候选机会、输出上限 8192、评价 worker 上限 8，使用共享模型池与公共 CPU 调度器。

主批次 18 路于 10月27日 11:14 启动。两格消融原定由 server4 的 tmux `traceaad_v15_ablation_waiter` 在主批次只剩图着色与 JSSP 时启动。10月27日 22:40，用户取消了这一消融，等待会话在启动任何一格前停止，日志为快照内 `launch_20261027/ablation_waiter.log`。`traceaad_v15_who`、`traceaad_v15_form` 只有预览，没有运行。

真实模型冒烟位于 `/home/fzy/code/LLM4AD/TraceAAD_runs/20261027_traceaad_v15_preflight`。TSP 冒烟（24 次候选，3 个初始程序，发育 2 步，停滞窗口 6）确认了按最少投入交替发育、在检查点按冠军曲线暂停、停滞后新建谱系、谱系内抽样父代、信息形式的 Crossover 与回访重问；后续初始化只见设计描述，回复字段顺序为 `analysis`、`code`、`design`。图着色冒烟在主批次覆盖该任务后停止。

V15 主批次已完成，最后一次同步包括全部主测试记录。主实验名单加入 `traceaad_v15`；两格消融的条目已随消融删除。修改前名单保存在 `experiments_result/traceaad_v15/launch_20261027/primary_runs_before.json`。

## V16 正式批次（10月27日–28日）

V16 是论文中的 TraceAAD。它在 V15 上加入比较点的乐观估计、修改说明和跨谱系参考程序，机制与开关见 [V16 README](../../traceaad/v16/README.md)。V14 与 V15 主批次继续运行，不停止、不替换。

| 实验目录 | 批次 | 设置 | 范围 | 状态 |
| --- | --- | --- | --- | --- |
| `traceaad_v16` | `20261028_v16_co6` | 默认，三个开关都开 | 六任务 × 种子 0／1／2 | 10月27日 23:06 启动 |
| `traceaad_v16_noproj` | `20261028_v16_noproj` | `projection=False`（论文的“无乐观估计”） | 六任务 × 种子 0／1／2 | 只写了预览，未排队 |

源码冻结在 `/home/fzy/code/LLM4AD/TraceAAD_runs/20261028_traceaad_v16_frozen`，424 个文件的 SHA-256 写入 `source_hashes.json`，启动前逐一核对。启动脚本为 `experiments_result/traceaad_v16/launch_20261028/launch_batch.py`：在快照根目录以 `PYTHONPATH=.` 运行，`--cell main|noproj` 选择批次，不带 `--launch` 时只写预览。脚本另外核对 `projection`、`changes`、`foreign_reference` 的取值。冻结结果检查仍要求 `summary.json` 含 `v15` 字段，因为 V16 沿用这一字段。每路 1000 次候选机会、输出上限 8192、评价 worker 上限 8，使用共享模型池与公共 CPU 调度器。18 路三秒错峰直接启动，启动时另有 15 路 V14、V15 在运行，共 33 路；模型请求超过 27 个槽位时在公共调度器中排队。批次名中的日期取本机日期，北京时间仍为 10月27日。

**放弃的首批 `20261027_v16_co6`。** 它于 10月27日 22:40 启动，约 20 分钟后由用户决定放弃。原因是本机 GGUF 端点不执行 JSON 模式：它的回复有 34%–54% 把完整的 JSON 对象包在一层 ```json 围栏中，旧解析器把这些回复记为解析失败，约占全部记录的 5%。三个 AWQ 端点没有出现这种回复。V14、V15 也有同样的损失（分别为 585、681 条）。修复后，解析器先去掉包住整个回复的一层围栏，再解析 JSON；用历史回复回放，V14、V15、V16 的围栏回复分别有 577／585、678／682、28／28 条可以解析，余下的是真正格式错误的回复。首批 18 路的 43 个进程已停止，停止回执为旧快照内 `experiments_result/traceaad_v16/stop_20261028.json`。数据保留在 `/home/fzy/code/LLM4AD/TraceAAD_runs/20261027_traceaad_v16_frozen`，公共仓库的 `experiments_result/traceaad_v16` 链接已改指新快照。首批的启动脚本、等待脚本与日志保留在 `launch_20261027/`。

真实模型冒烟位于 `/home/fzy/code/LLM4AD/TraceAAD_runs/20261027_traceaad_v16_preflight`（TSP，30 次候选，3 个初始程序，发育 2 步，停滞窗口 6），使用修复前的源码。修改操作的回复都按 `analysis`、`code`、`design`、`change` 四个字段解析成功；提示中除任务模板外只有修改对象的代码；形成路径与此前尝试以 `Change` 加成绩出现；Crossover 的参考程序来自其他谱系，指令要求保留当前主决策规则；乐观估计触发一次。冒烟中检查脚本命名为 `inspect_smoke.py`：放在源码根目录的 `inspect.py` 会遮蔽标准库，使搜索无法启动。

本机 `traceaad-results-sync@v16.service` 运行 `experiments_result/traceaad_v16/launch_20261028/sync_batch.py`，每 60 秒同步主批次，全部完成主测试后退出。本地 `experiments_result/traceaad_v16/` 中仍有首批已同步的运行目录；主实验名单只列新批次，看板因此只显示新批次。主实验名单加入 `traceaad_v16` 并设为默认；修改前名单保存在 `experiments_result/traceaad_v16/launch_20261028/primary_runs_before.json`。

## V17 正式批次（10月28日）

V17 = V16 的上下文 + V15 的同龄比较（`projection=False`）+ 组件导入（`imports`）：Crossover 改为导入其他谱系中带来改进的修改。另有默认关闭的 `progress_rule`（冠军以外的谱系按自身是否进步继续开发），前两批用的是它。机制与开关见 [V17 README](../../traceaad/v17/README.md)。V14、V15、V16 继续运行，不停止、不替换。

| 实验目录 | 批次 | 设置 | 范围 | 状态 |
| --- | --- | --- | --- | --- |
| `traceaad_v17` | `20261028_v17_sameage` | 默认：同龄比较、`imports` 开 | 六任务 × 种子 0／1／2 | 10月28日 14:08 启动 |
| `traceaad_v17_progress` | `20261028_v17_progress` | `progress_rule=True` | 六任务 × 种子 0／1／2 | 只定义在启动脚本中，未预览、未排队 |
| `traceaad_v17_noimports` | `20261028_v17_noimports` | `imports=False`（无乐观估计的 V16） | 六任务 × 种子 0／1／2 | 只定义在启动脚本中，未预览、未排队 |

源码冻结在 `/home/fzy/code/LLM4AD/TraceAAD_runs/20261028_traceaad_v17_sameage_frozen`，436 个文件的 SHA-256 写入 `source_hashes.json`，启动前逐一核对。启动脚本为 `experiments_result/traceaad_v17/launch_20261028_sameage/launch_batch.py`：在快照根目录以 `PYTHONPATH=.` 运行，`--cell main|progress|noimports` 选择批次，不带 `--launch` 时只写预览。脚本核对 `projection=False`、`changes`、`foreign_reference` 为真，以及两个 V17 开关与所选格一致。冻结结果检查要求 `summary.json` 同时含 `v15` 与 `v17` 字段。每路 1000 次候选机会、输出上限 8192、评价 worker 上限 8，使用共享模型池与公共 CPU 调度器。18 路三秒错峰直接启动；启动时另有 V16 四路在运行，共 22 路。公共仓库的 `experiments_result/traceaad_v17` 链接到新快照的结果目录。

**停止的重跑 `20261028_v17_rerun`（进步规则）。** 它于 12:23 启动，14:07 由用户决定停止，当时各路做到 81–370 条记录。进步规则按设计执行，但冠军以外的谱系几乎总有一个在自身记录上改进：372 次活跃尝试中 311 次给了同龄比较会暂停的谱系，换来 4 次冠军更替。V17 在原处改为默认使用同龄比较。18 路的 44 个进程已停止，停止回执为旧快照内 `experiments_result/traceaad_v17/stop_20261028_rerun.json`，本地副本在 `launch_20261028_rerun/`。数据保留在 `/home/fzy/code/LLM4AD/TraceAAD_runs/20261028_traceaad_v17_rerun_frozen`，停止后的最终状态已同步到本地；它可作为 `progress_rule=True` 在相同记录数下的对照。启动脚本、同步日志与修改前名单保留在 `launch_20261028_rerun/`。

**放弃的首批 `20261028_v17_co6`。** 它于 10月28日 10:29 启动，12:19 由用户决定放弃，当时各路做到 82–360 条记录。原因是发散几乎不发生：发散只排在没有活跃谱系之后，而冠军以外的谱系几乎总有一个在自身记录上改进，相同记录数下只发散了 6 次（V14–V16 为 40–47 次）。V17 在原处修正：探索打开且满足 V15 的发散条件时先发散。首批 18 路的 44 个进程已停止，停止回执为旧快照内 `experiments_result/traceaad_v17/stop_20261028.json`，本地副本在 `launch_20261028/`。数据保留在 `/home/fzy/code/LLM4AD/TraceAAD_runs/20261028_traceaad_v17_frozen`；首批的启动脚本、同步日志与修改前名单保留在 `launch_20261028/`。检查脚本与结果在 `experiments_result/analysis/v17_progress_20261028/`。

真实模型冒烟位于 `/home/fzy/code/LLM4AD/TraceAAD_runs/20261028_traceaad_v17_preflight`（TSP 与 JSSP，各 40 次候选，3 个初始程序，发育 2 步，停滞窗口 4），使用修正前的源码。TSP 冒烟用完预算：40 条记录全部有效，没有解析失败；发育、冠军、活跃谱系三种分配都出现，一个在比较点落后的谱系靠自身进步继续开发并接管了冠军；4 次 Import 中 2 次改进父代，结果与组件程序的代码距离中位数为 0.88。修正只改变分配顺序，发散请求沿用 V14 的提示，所以没有再做冒烟；新顺序由单元测试覆盖。检查脚本为快照内 `inspect_smoke.py`。第三批 `20261028_v17_sameage` 的默认路径由已运行过的部件组成（V15 的同龄比较、V16 的上下文、前两批的 Import），由单元测试覆盖组合，也没有另做冒烟。

本机 `traceaad-results-sync@v17.service` 经 drop-in `rerun.conf` 运行 `experiments_result/traceaad_v17/launch_20261028_sameage/sync_batch.py`，每 60 秒同步新批次，日志为同目录 `sync.log`，全部完成主测试后退出。新批次运行在主实验名单中，所以 `watch_server4` 自动检查它。本地 `experiments_result/traceaad_v17/` 中仍有前两批已同步的运行目录；主实验名单只列新批次（标签 “TraceAAD V17”），默认方法仍为 `traceaad_v16`。修改前名单保存在 `launch_20261028_sameage/primary_runs_before.json`。

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
