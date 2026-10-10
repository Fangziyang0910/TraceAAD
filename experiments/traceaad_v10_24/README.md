# 围绕求解器作用的有限开发

V10.24 已实现，并于2026-10-10完成成组编辑与开发机制重构。Refine、Crossover、Explore共用四次一段、最多两段；新开与续投交替，待续投按等待顺序恢复原worktip或失败源码。单一候选预算，没有Plan准入、预留或名义槽。成熟程序默认SEARCH/REPLACE，初始化与整体重组保留完整Code；只评价原子应用后的完整结果。

[最终设计与完整提示](../../docs/04-研究认识与构想/2026-10-09-从求解行为到开发决策的V10.24全局设计.md)解释推导、来源、公式与边界。[实验准则](../PROTOCOL.md)保持不变。2026-10-10已启动首个18路真实搜索批次，评价与冻结条件保持原协议。

## 当前运行批次

2026-10-10按用户指示停止`20261009_server3_v1023c`的18路搜索，停止记录累计11,925次已提交候选，各路581—861次；已有结果已同步本地，未作为完整预算结果。模型服务、CPU/GPU调度器和决策模型采集保持运行。

首批`20261010_server3_v1024`暴露了具体修改被固定为持续目标及旧片段edit的问题。停止时保留585次候选（各路23—38次），以独立Question、全单元紧凑结果、当前基底重申修正；精确匹配及评价条件不变。

新版批次：`20261010_server3_v1024b`，server3，co6×3，种子0/1/2，每路1000候选，共18,000候选。每次评价最多8个worker，复用`/tmp/traceaad-1005/scheduler.sock`；候选每实例2秒CPU、整实例20秒墙钟上限不变。

- [批次清单](../../experiments_result/traceaad_v10_24/batch_20261010_server3_v1024b.json)保存18路路径、命令、采样、评价条件和实现哈希。
- [源码哈希快照](../../experiments_result/traceaad_v10_24/source_20261010_server3_v1024b.json)与同目录`source_20261010_server3_v1024b.tar.gz`固定本次未提交工作区的134个实现文件；部署后逐一核对一致。
- 训练看板 [8765](http://127.0.0.1:8765/#b=traceaad_v10_24&rb=latest) 默认选最新启动批次；可切换旧批次，详情显示开发单元和真实支出。
- 本地tmux会话`watch_v1024`每60秒同步结果。查询当前状态用下列命令，不能把本地快照当作实时进程状态。

```bash
uv run python -m experiments.infra.batch_status --ssh B3-server3 \
  --repo /home/fzy/code/LLM4AD/TraceAAD \
  --manifest experiments_result/traceaad_v10_24/batch_20261010_server3_v1024b.json --details
```

首轮核验：18个新进程、预算与协议身份均匹配；东京时间01:28:45共94次候选，无服务错误或预算／状态不一致。此截点主要仍在初始化，不作为修正后算法收益证据。健康记录为[重启后检查](../../experiments_result/traceaad_v10_24/health_20261010_v1024b_followup.json)。

## 运行入口

先检查配置；该命令不调用模型：

```bash
uv run python -m experiments.traceaad_v10_24.run --task tsp_construct --dry-run
```

单路搜索使用独立运行名：

```bash
uv run python -m experiments.traceaad_v10_24.run \
  --task tsp_construct --backend server3 --seed 0 --repeat 1 \
  --run-name v1024_tsp_rep1 --budget 1000 --eval-workers 8
```

六任务批次入口复用现有主机调度：

```bash
uv run python -m experiments.traceaad_v10_24.launch_host \
  --suite co6 --batch <批次名> --eval-workers 8 \
  --scheduler-socket /tmp/traceaad-1005/scheduler.sock
```

恢复使用相同运行名、配置、任务、评价协议和交付协议。当前交付条件为`v1024-atomic-edit-1`，开发条件为`v1024-shared-development-3`，上下文为`v1024-context-3`。旧full-only或首版预留机制的检查点均拒绝续接，必须使用新运行目录。不要复用 V10.23 目录。任务结束后，按原协议另行评价冻结程序：

```bash
uv run python -m experiments.infra.evaluate \
  experiments_result/traceaad_v10_24/<任务>/<运行> --primary --condition traceaad
uv run python -m experiments.traceaad_v10_24.diagnose \
  --run-dir experiments_result/traceaad_v10_24/<任务>/<运行>
```

主协议仍为训练16、同规模测试50、三路种子0/1/2、每路1000候选、训练前五重评后冻结。评价器、模型服务、函数约束与测试读取规则沿用公共设施。修改这些条件不是本版默认配置。

## 怎样读取结果

- `events.jsonl`：每次候选保留 `unit_id`、`block_id`、真实父程序、声明 Base、回退原因、actual_diff、元字段、材料 ID、prompt_hash，并新增delivery_mode、edit_base_id、entered_evaluation与delivery_cost。失败的edit提交保留为未应用证据。`development` 增量保存受影响的单元、块与调度状态。
- `resume.json`：`state.v1024` 是可由事件增量重建的最新检查点，包含所有段、单元、下一种资源决定和最小基底超长记录。已有三个 RNG 一起恢复。
- `calls.jsonl[.gz]`：实际请求／回复与模型调用成本；源码复用现有程序档案，评价包含逐实例分数与 CPU 数据。
- `diagnostics.json` 的 `explorations`：在 V10.24 中是有限问题的诊断，`kind=finite_development_requests`。显示新开／续投与来源成本、实际开题数、Explore率、分配原因、上下文裁剪与关闭原因、各单元事件链、段收益及 `gain_accounting_error`。其中delivery按edit/full汇总尝试、交付失败、实际评价、新有效程序、调用和输入／输出token；usage缺失保留为空。
- `summary.json.development`：紧凑的用途成本与已结算前沿收益。统计时不要把算子次数与资源用途份额混同。

段成本只包含实际候选，重复和失败照常收费。尾部一次授予剩余长度；提前放弃的未花预算留在全局池。waiting只是未分配机会，不是预算承诺。champion现在表示单元实际交付的最好有效结果，可以是缓存程序；新增前沿收益仅计新有效源码。

每次只有宿主基底可直接编辑。历史差异与反馈按容量选取，记录`omitted_materials`、`evidence_excerpts`及真正展示的ID。基底自身装不下才排除该基底，局部问题装不下只关闭本单元；零尝试的局部失败阻止相同来源／起点空转，不删除程序。

## 本地验证

```bash
uv run pytest -q tests/method/test_traceaad_v1024.py \
  tests/method/test_traceaad_v1024_edits.py \
  tests/method/test_traceaad_v1023.py tests/method/test_traceaad_v1021.py \
  tests/method/test_traceaad_v1017.py tests/method/test_traceaad_v1015.py \
  tests/experiments/test_traceaad_v1015_experiments.py
```

当前143项相关检查通过：V10.24机制与编辑52项，旧版本与公共入口91项。检查覆盖共同生命周期、跨段弱工作版、有效缓存回退、重复失败、同分前沿与donor边界、最小／局部上下文超长、材料压缩、来源优先、尾部计费、断点恢复、事件重建、六任务提示和实验入口；也运行旧版本及公共入口回归。离线检查不代表真实模型搜索效果。

edit应用先校验整批在原始基底上的唯一匹配与不重叠，不评价中间半成品；匹配失败仍占一次候选但不进入评价。完整结果原样保存，身份规范化用于去重。详细规则与证据见[成组编辑与连续开发](../../docs/04-研究认识与构想/2026-10-10-成组编辑与连续开发.md)。

真实效果以固定 V10.23 与完整 V10.24 的同协议搜索比较为主；需要解释差异时再做“提案来源×开发规则”和调度闭环检查。具体设计见最终设计第12节。1:1交替若每问题用满8次，每百候选约12.5个开题，Explore约4.17个；这比首版更少，须同时核对连续开发收益与开题机会成本。
