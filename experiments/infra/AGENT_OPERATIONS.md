# Agent 执行入口

检查实验进度、等待完成、同步远端档案时，先读本文。这里保存稳定入口；进程、资源和完成数量以当次查询为准。

## 批次状态

旧本地模型结果已归档到`archive/local_models_20261011/`，对应摘要与清单status为archived；不列作活跃实验、等待队列或待补测。EoH与ReEvo已恢复的CVRP、FSSP完成搜索，图着色与JSSP继续；FunSearch四路已按用户要求启动恢复。MCTS-AHD保持暂停。实时进度以批次查询和进程核验为准。

批次清单是查询范围的依据。指定清单，不遍历所有版本，也不以已同步的路次代替整个批次。

```bash
uv run python -m experiments.infra.batch_status --manifest experiments_result/eoh/batch_20261011_server3_deepseek_off_pilot.json
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
uv run python -m experiments.infra.batch_status --ssh B3-server3 --repo /home/fzy/code/LLM4AD/TraceAAD --manifest experiments_result/eoh/batch_20261011_server3_deepseek_off_pilot.json
```

连接失败时再检查 `ssh -G B3-server3`；仓库或解释器不存在时再定位环境。已知入口可用时无需重新搜索目录、检查全部依赖或读取 SSH 配置。

## 本机模型服务

后端 `local`（`http://127.0.0.1:8001/v1`，模型名 `Qwen3.8-27B`，3 槽）由 tmux 会话 `qwen3_8-27b-mtp` 中的 llama-server 提供：`~/models/gguf/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf`，上下文 98304（3 槽各 32768），MTP 草稿 3 个 token，KV 缓存 q8_0，日志 `/tmp/llama-qwen38-restart.log`。先用 `curl -s --noproxy '*' 127.0.0.1:8001/health` 检查；会话不存在时按 `~/models/gguf/qwen3.8-27b/update_ud_q4_k_xl.sh` 末尾的同一组参数在该会话中启动。

本机旧实验的两个调度器已在空闲状态停止，旧队列不再启动；server3当前调度器与DeepSeek实验继续运行。现用看板在tmux `results_monitor`中，日志为`logs/results_monitor.log`。

## 本机基线排队

基线在本机排队运行：同时最多 3 路（本机模型的 3 个槽位），一路结束补下一路；每路一次评价一个实例，核心向本机 CPU 调度器申请，与 TraceAAD 同一核心池。调度池只含每个物理性能核的一个线程（`--cpus 0,2,4,6,8,10,12,14`）；本机 16–31 号是能效核，llama-server 用 `taskset -a -cp 16-31 <pid>` 固定在能效核，重启模型服务后要重新执行；能效核同时是平台对照中的一个评价平台，本机模型服务有请求时会与它争用。server3 的调度器在其 tmux 会话 `traceaad_scheduler` 中运行（socket `/tmp/traceaad-1005/scheduler.sock`，0–51 号各一个物理核，两个模型端点各 9 个槽位；结束会话不会删除 socket 文件，确认没有调度器进程后先删掉它再重启）；同步代码前的工作区备份在 server3 的 `~/backups/`。基线的超时率读运行目录的 `evaluations.jsonl`，`events.jsonl` 不区分失败类型。排队器在 tmux 会话中运行，日志在 `experiments_result/<方法>/launch_logs/`。

```bash
uv run python -m experiments.infra.local_queue --method funsearch --suite co6 --batch 20261008_local --slots 3
```

排队器只在内存中记录在跑的路次；它中途退出时，等这些路次结束后再重启。

## 等待、检索和同步

训练可视化使用本地 `8765` 端口（`http://127.0.0.1:8765/`）。先检查该端口；服务未运行时，用 `uv run python -m experiments.monitor --host 0.0.0.0 --port 8765` 启动。不指定 `--experiment` 时默认展示最近更新的批次；版本对比默认选中该 TraceAAD 批次和全部基线方法。查看其他方法用链接中的 `#b=<方法目录>`。运行监控默认筛选该方法的最新启动批次，也可选全部或具体批次；`#b=<方法目录>&rb=<启动批次>`固定条件，`rb=latest`跟随最新批次。API用`/api/state?batch=<方法目录>&run_batch=<启动批次或latest>`筛选，不传run_batch则返回全部。页面读取本地档案，运行中路次显示最近同步的快照；文件变化会刷新缓存。核验 `/api/state?batch=eoh` 的路次数、曲线和 ETA，并用 `/api/compare?cohorts=eoh,reevo` 核验测试结果。

V10.15–V10.19 共用生成、评价和结果保存，各版本仍写到原版本目录。日常读取只使用 `events.jsonl`、`programs.jsonl`、`resume.json`、`summary.json`、`selection.json` 和平面的 `heldout.json`。原始模型请求与回复单独保存于 `calls.jsonl`，结束后压缩。诊断入口为 `uv run python -m experiments.traceaad_v10_17.diagnose --run-dir <目录>`，其他当前版本同样提供。格式、恢复条件和历史迁移见[实验与结果](SEARCH_FORMAT.md)。

横轴单位来自 `run_config.json` 的 `budget_axis`，候选编号和实际预算分别保留。基线实际预算读取方法事件与恢复点，不用保存的有效候选数代替；曲线延伸到已记录的实际预算位置。已停止且用完预算、尚未完成最终收尾的路次显示“搜索完成 · 待收尾”。列表、最近候选、有效率和预算共用一个增量投影；有效率的分母是全部候选记录数。

V10.24详情还显示开发问题、工作版／单元最好版、分段实际支出、逐次结果，以及edit/full评价、缓存和交付失败统计。显式停止的批次以清单停止标记为准，不被旧运行总结覆盖。

详情显示搜索最优和最终选中程序。源码按内容哈希从 `programs.jsonl` 读取。带身份的测试结果须与冻结的最终程序一致；不一致或无法核验的记录不进入比较。

`.cache/history.json` 可以删除并重建。缓存只保存小型事件投影、程序元数据和读取位置，不重复源码、请求或回复。首次读取扫描轻量事件，后续只读追加字节；未完成的尾行下次重试。配置、恢复点、事件、程序、选择及批次清单变化会刷新 API 缓存；未完成路次的时间状态最多每 15 秒重算一次。页面每分钟更新批次列表，切换批次立即取消旧请求。`--experiment` 可固定默认批次，链接中的 `#b=` 优先。修改 Python 后重启服务，修改 HTML、CSS 或 JS 后刷新页面。

等待状态变化使用 `--wait-seconds 45`，默认每 10 秒在同一进程中检查，变化或到期后返回一次。时间戳和 ETA 自身变化不会唤醒调用方。每次最多等待 60 秒，之后可汇报状态或处理其他工作。普通快速查询直接等待返回；后台长任务按进度检查，避免每秒调用 `write_stdin`。

读取会话日志先抽样确认结构，再按日期、工作目录和消息类型筛选。先输出计数和少量样例，需要完整证据时再展开。文件检索先用 `rg --files` 定位，再用 `rg -n` 和局部读取；文件枚举顺序不代表时间顺序。

独立的本地读取和远端查询用 `Promise.allSettled` 并行，并检查每项结果。编辑、启动、同步和核验按依赖顺序执行。复杂远端 Python 通过 stdin 或脚本文件传递，命令参数用 shell quoting；避免层层嵌套引号。

server3 是一次性的执行机：不保留 git、历史版本、备份或已拉回的结果，只保留统一配好的 `.venv` 和正在运行的批次。用 `uv run python -m experiments.infra.remote` 管理：`push` 把本机工作区镜像到 server3（删除那边多出的代码，保留 `.venv` 与 `experiments_result`；依赖文件变化时执行 `uv sync --frozen`）；`pull --batch <批次>` 拉回该批次的运行目录、清单和启动日志；`prune --batch <批次>` 删除 server3 上已结束、已有 held-out 且与本地逐字节一致的运行，最后一路删掉时连同清单和启动日志；`watch --batch <批次>` 每 60 秒拉取并清理，批次在 server3 上清空后退出。启动 server3 批次前先 `push`；运行期间在本机 tmux 中 `watch`。

同步前固定已完成路次清单，再用 `rsync --files-from` 统一同步，随后对冻结文件核验哈希。新完成路次加入下一轮。运行中的日志可继续变化，应与冻结档案分开处理；不要因同步时日志增长而反复重传已冻结档案。批次全局汇总可能持续变化，不作为冻结路次的内容哈希依据。


## 已暂停：EoH／ReEvo／FunSearch／MCTS-AHD 六任务关闭思考批次

批次`20261011_server3_deepseek_off_pilot`，EoH和ReEvo各6任务×1路，共12路；seed=0，1000候选／路，每路最多8个实例worker，共用`/tmp/traceaad-1005/scheduler.sock`的52核心CPU池。生成使用DeepSeek-V4.1-Flash，原生思考关闭、不发送reasoning_effort，temperature=1.0、top_p=0.95。DeepSeek版ReEvo初始化增量设为0，作为明确记录的模型适配；原始类默认+0.3保留。EoH、ReEvo输出上限各16384。

远端Python使用`/home/fzy/code/LLM4AD/TraceAAD/.venv/bin/python`原路径，不解析symlink；启动设置`HF_HUB_OFFLINE=1`，使用已部署的tokenizer缓存。凭据只从远端.env读取。源码哈希与实际命令在批次清单，逐次实际请求与API用量在calls.jsonl，token_usage.json实时累计。反思等非候选调用也记录。

清单分别为`experiments_result/eoh/batch_20261011_server3_deepseek_off_pilot.json`、`experiments_result/reevo/batch_20261011_server3_deepseek_off_pilot.json`。本机`sync_20261011_server3_deepseek_off_pilot_eoh`、`sync_20261011_server3_deepseek_off_pilot_reevo`每60秒复用remote watch同步；按现有流程，完成独立测试且本地核验一致后才清理远端。

```bash
uv run python -m experiments.infra.batch_status --ssh B3-server3 --repo /home/fzy/code/LLM4AD/TraceAAD --manifest experiments_result/eoh/batch_20261011_server3_deepseek_off_pilot.json
uv run python -m experiments.infra.batch_status --ssh B3-server3 --repo /home/fzy/code/LLM4AD/TraceAAD --manifest experiments_result/reevo/batch_20261011_server3_deepseek_off_pilot.json
```

旧`20261011_server3_deepseek_high_pilot`保持停止，标为diagnostic_excluded_from_main；旧同步watch已停止。high／low／off诊断已结束，临时脚本与输出已删除，结论记录在基线审查文档，不纳入正式方法比较。

FunSearch已加入同批次，六任务各1路，seed=0、1000候选、每路最多8个实例worker，共用上述52核心CPU池。关闭思考、temperature=1.0、top_p=0.95、max_tokens=16384。保留方法默认10个岛、每提示4次采样、2个父代与4小时岛重置周期。实际配置及token记录位于各运行目录；源码哈希与启动命令见`experiments_result/funsearch/batch_20261011_server3_deepseek_off_pilot.json`。本机tmux `sync_20261011_server3_deepseek_off_pilot_funsearch`每60秒同步。当前共18路。FunSearch复用已部署且核对一致的方法代码，未向server3覆盖在跑实验的公共源码。

MCTS-AHD于10月11日15:12加入同批次，六任务各1路，seed=0、1000候选、每路最多8个实例worker，共用上述52核心CPU池；生成设置与上相同，方法参数为默认值（初始4、种群10、λ0=0.1、α=0.5、4个采样线程）。同样复用server3已部署且核对一致的代码，未push；源码哈希与启动命令见`experiments_result/mcts_ahd/batch_20261011_server3_deepseek_off_pilot.json`。本机tmux `sync_20261011_server3_deepseek_off_pilot_mcts_ahd`每60秒同步。当前共24路。

2026-10-11按用户要求因API额度不足中断：本批次四种方法的运行进程已全部停止，清单标记`stopped_by_user`，原因`api_quota_insufficient`。结果已拉回本机保留，四个同步watch已停止；不会自动重启。下文历史启动与同步说明仅用于定位批次，不表示当前正在运行。

## 中断后的基线恢复

`experiments.infra.resume_baseline`支持EoH、ReEvo、FunSearch、MCTS-AHD、CALM、PathWise、ShinkaEvo七种基线。本次旧批次涉及前四种方法。默认只核对配置、程序来源、预算和搜索状态，不调用模型、不做任务评价；核对结果在各方法的`recovery_check_20261011_server3_deepseek_off_pilot.json`。六路已满1000候选，只需最终训练重评与独立测试；其余18路按已花预算继续，不能重新初始化或重置token累计。

```bash
# server3仓库根目录；单路只读核对
HF_HUB_OFFLINE=1 .venv/bin/python -m experiments.infra.resume_baseline --run-dir <运行目录>
# FunSearch缺少历史评分向量时，只做CPU补全，不调用模型
HF_HUB_OFFLINE=1 .venv/bin/python -m experiments.infra.resume_baseline --run-dir <运行目录> --prepare-profiles
# 仅在用户要求恢复后执行；不足预算的路次会调用模型
HF_HUB_OFFLINE=1 .venv/bin/python -m experiments.infra.resume_baseline --run-dir <运行目录> --execute
# 整个方法批次：默认只读核对；加--execute在原tmux会话中启动各路恢复
HF_HUB_OFFLINE=1 .venv/bin/python -m experiments.infra.resume_baseline --manifest <批次清单> --execute
```

批次入口支持上述七种基线。在执行机仓库根目录运行，按清单所在目录定位各路，不沿用另一台机器的绝对路径。已冻结路次和已存在的tmux会话自动跳过；满预算但未冻结的路次只做最终训练选择。后台日志写入各路`logs/resume_search.log`，CPU补全使用`--prepare-profiles`及`logs/recovery_profiles.log`。打印launched只表示会话已启动，须继续核对日志、进程与批次状态；结果同步仍使用既有`remote watch`入口。

新启动与恢复入口共用`.recovery.lock`，阻止同一路重复写入；锁随进程退出释放。新入口写入的running状态带`writer_lock`标记，强制终止后可直接再次执行恢复，无需手改状态。未升级的旧进程没有这个标记，仍需核实进程已经停止后修正旧状态，不能仅凭锁空闲判断其存活。批次清单更新另用`.batch.lock`串行写入。

预算未满的中断记录为stopped或interrupted，不提前冻结最终程序；方法原有的自然结束规则仍可正常完成。预算已满后恢复FunSearch最终选择，仍使用已保存的重评分数与失败排除结果。后续候选已有逐实例向量时直接读取，标记`recorded_database`；缺失的旧向量仍需重评，标记`reevaluated_database`。本轮改动只涉及记录、恢复和启动，不改变基线的生成、父代选择、反思、树搜索或候选预算规则。当前已启动的server3进程仍使用启动时源码，本机改动未推送。

前四种方法的旧日志恢复是`journal_boundary`：旧随机状态和执行栈未保存，重新设定原seed。EoH从下一算子开始；ReEvo保留最近完整种群、当前最好程序及最后成功的长期反思，从新一代开始；MCTS按事件与实际父代提示恢复树，从新一轮树搜索开始；FunSearch恢复岛归属和评分向量，从新提示批开始。未完成的阶段不承诺逐步重现；已有候选支出不退回。恢复事实及代码哈希追加到`recovery.jsonl`，原始日志与配置不覆盖。

CALM、PathWise和ShinkaEvo的新实验会自动保存`search_checkpoint.json.gz`，无需额外开关。它是压缩JSON，通过临时文件原子替换；只保存搜索对象和随机状态，不保存模型客户端或线程池。恢复核对原模型、任务、参数、预算身份以及事件和源码文件的完整性，标记为`search_checkpoint`。

| 方法 | 保存与恢复的状态 | 恢复边界 |
| --- | --- | --- |
| CALM | 候选与种子、逐实例评分、提示试验与回访关系、注入组件、重复检测、数值微调尝试、停滞计数、已准备消息及独立RandomState | 种子评价／数值微调完成、准备好的轮次或完整轮次结束 |
| PathWise | 初始化进度与候选、完整种群、节点档案与最好节点、父代信息、反思历史、轮次与节点编号 | 初始化候选结束或外层迭代结束 |
| ShinkaEvo | 全部岛程序与副本身份、archive及亲子计数、embedding、bandit统计与随机流、meta记忆与待总结程序、已完成代数 | 种子或整代（含迁移）结束 |

三个方法均恢复检查点时的Python／NumPy随机状态。Shinka从保存代数的下一代继续，不把代数重置为1；CALM已有消息不重新调度提示，种子评价后的检查点继续初始数值微调。初始化、解析失败、patch拒绝和数值微调的预算按各方法原有口径保留，不用有效候选记录数代替已花预算。CALM在批次评价和数值微调开始时记录原来已经预留的评价预算。

执行栈不保存。检查点之后未完成的阶段从保存边界重新开始，该阶段已记录的支出仍计入总预算，不退还；PathWise跳过已花预算的初始化位置，未完成外层迭代使用新编号，避免复用日志中的节点身份。原调用与评价记录保留。恢复本身不调用模型、不重新评价旧程序或请求旧embedding；后续搜索和最终训练选择按原规则正常调用。旧CALM、PathWise、Shinka档案没有完整检查点时，预算未满的恢复会明确拒绝；不能仅凭最好程序猜测剩余搜索状态。

写入前先在持有运行锁的情况下保存并移除JSONL未完成尾行，片段保留在`logs/interrupted_tails/*.partial`，完整记录和其他硬链接副本不变。调用压缩也先形成完整临时档案，再原子替换，避免二次中断破坏旧gzip。上述处理只由写入入口执行，只读核对不修剪文件。恢复或核对失败时关闭评价线程、生成／meta客户端及embedding客户端。

FunSearch按用户确认的方式重新评价旧程序，使用重评后的分数与逐实例向量重建数据库。数据独立写`recovery_profiles.jsonl`与`recovery_evaluations.jsonl`，旧调用、候选和评价记录不覆盖。原岛归属保留，聚类及岛内最好分数按新结果更新，不要求与旧值相同。重评失败的程序保留失败记录并排除出重建数据库；候选预算不退回。最终训练候选排序也使用重建后的分数，避免新旧评分混用。空岛、父代身份不唯一或旧岛重置来源缺失仍拒绝启动模型请求。

恢复方式明确记为`reevaluated_database`，不是中断前状态的精确还原。后续FunSearch评价保存逐实例向量，不再依赖补测恢复这些结果。原预算、候选编号与token累计保留，恢复重评成本单独记录。

FunSearch的CPU补全在tmux `recover_profiles_20261011_funsearch_<task>`中进行，每路一次程序、8个实例worker，日志为各路`logs/recovery_profiles.log`。CVRP已补齐；FSSP、图着色、JSSP继续重评，分数变化不会中止。`recovery_ready.json`分别记录preparing、ready或实际错误。准备完成不会自动恢复LLM请求，正式搜索全部保持暂停。用户要求恢复后，再用独立tmux会话包装--execute并恢复相应remote watch。已冻结路次拒绝重复选择最终程序。

按用户最新要求，先恢复EoH四路：CVRP从653／1000、FSSP从774／1000、图着色从432／1000、JSSP从399／1000继续。采用原批次、原运行目录与原tmux会话，通过resume_baseline --execute恢复；模型、采样与评价条件保持原配置。本机`sync_20261011_server3_deepseek_off_pilot_eoh`恢复每60秒同步。TSP与OP已满预算，不新增LLM请求。

随后按用户要求恢复ReEvo四路：CVRP从738／1000、FSSP从796／1000、图着色从391／1000、JSSP从414／1000继续。恢复最近种群、全局最好程序和长期反思，从新一代开始，先成对反思再生成候选。原模型与评价设置不变，token累计连续；本机`sync_20261011_server3_deepseek_off_pilot_reevo`恢复同步。TSP与OP已满预算，不新增模型请求。

随后按用户要求恢复FunSearch四路：CVRP从702／1000、FSSP从782／1000、图着色从592／1000、JSSP从466／1000继续。四路CPU重评均已ready，使用重评分数与逐实例向量重建数据库，从新提示批继续；原岛归属、已花预算和token累计保留。复用原tmux会话，恢复日志为各路`logs/resume_search.log`；本机`sync_20261011_server3_deepseek_off_pilot_funsearch`每60秒同步。TSP与OP实际预算均为1000，不新增模型请求。
