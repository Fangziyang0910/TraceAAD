# 围绕求解器作用的有限开发

V10.24 已实现，并于2026-10-10接入成组编辑。它围绕一个具体改进问题组织 anchor、proposal、champion、worktip 与完整试验记录；按四次候选一块分配预算，每个新问题至多两块。默认账期为前沿精炼4、前沿借用4、新问题4、后续开发4。第二块来自评分前承诺或分配时的质量前沿资格。成熟程序默认SEARCH/REPLACE，初始化与适合整体重组的情况保留完整Code；成组编辑只评价一次完整结果。

[最终设计与完整提示](../../docs/04-研究认识与构想/2026-10-09-从求解行为到开发决策的V10.24全局设计.md)解释推导、来源、公式与边界。[实验准则](../PROTOCOL.md)保持不变。此次只运行离线测试与入口检查，没有发出真实模型请求、启动训练或重启已有批次。

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

恢复使用相同运行名、配置、任务、评价协议和交付协议。当前交付条件为`v1024-atomic-edit-1`，不能无记录地续接此前full-only检查点。不要复用 V10.23 目录。任务结束后，按原协议另行评价冻结程序：

```bash
uv run python -m experiments.infra.evaluate \
  experiments_result/traceaad_v10_24/<任务>/<运行> --primary --condition traceaad
uv run python -m experiments.traceaad_v10_24.diagnose \
  --run-dir experiments_result/traceaad_v10_24/<任务>/<运行>
```

主协议仍为训练16、同规模测试50、三路种子0/1/2、每路1000候选、训练前五重评后冻结。评价器、模型服务、函数约束与测试读取规则沿用公共设施。修改这些条件不是本版默认配置。

## 怎样读取结果

- `events.jsonl`：每次候选保留 `unit_id`、`block_id`、真实父程序、声明 Base、回退原因、actual_diff、元字段、材料 ID、prompt_hash，并新增delivery_mode、edit_base_id、entered_evaluation与delivery_cost。失败的edit提交保留为未应用证据。`development` 增量保存受影响的单元、块与调度状态。
- `resume.json`：`state.v1024` 是可由事件增量重建的最新检查点，包含所有块、单元、槽内未用额度与预留。已有三个 RNG 一起恢复。
- `calls.jsonl[.gz]`：实际请求／回复与模型调用成本；源码复用现有程序档案，评价包含逐实例分数与 CPU 数据。
- `diagnostics.json` 的 `explorations`：在 V10.24 中是有限问题的诊断，`kind=finite_development_requests`。显示实际用途成本、分配原因、各单元事件链、每百候选开题率、块收益及 `gain_accounting_error`。其中delivery按edit/full汇总尝试、交付失败、实际评价、新有效程序、调用和输入／输出token；usage缺失保留为空。
- `summary.json.development`：紧凑的用途成本与已结算前沿收益。统计时不要把算子次数与资源用途份额混同。

块成本只包含实际候选。重复和失败收费；空上下文块为零成本。提前结束的未用额度在当前名义槽内回流到精炼；尾部预留块跨槽时用 `nominal_allocations` 分摊实际支出，承诺长度不截短；续投空额按账期交替回流。预算结束时未获第二块者仍标记 waiting，报告实际观察长度。有效冠军为空时，相对 anchor 的收益为空，不用 anchor 冒充产物。

## 本地验证

```bash
uv run pytest -q tests/method/test_traceaad_v1024.py \
  tests/method/test_traceaad_v1024_edits.py \
  tests/method/test_traceaad_v1023.py tests/method/test_traceaad_v1021.py \
  tests/method/test_traceaad_v1017.py tests/method/test_traceaad_v1015.py \
  tests/experiments/test_traceaad_v1015_experiments.py
```

当前134项相关检查通过：V10.24状态机制24项、成组编辑19项，以及旧版本和公共入口91项。

检查涵盖状态转移、预算／预留、非贪心工作版、Repair 证据、源码去重、前沿分组、资格复查、恢复一致性、事件重建、原冻结流程、六任务提示和18路批次计划。离线模型只输出预设代码；通过这些检查不代表真实模型的算法收益已得到验证。

edit应用先校验整批在原始基底上的唯一匹配与不重叠，不评价中间半成品；匹配失败仍占一次候选但不进入评价。完整结果原样保存，身份规范化用于去重。详细规则与证据见[成组编辑与连续开发](../../docs/04-研究认识与构想/2026-10-10-成组编辑与连续开发.md)。

真实效果以固定 V10.23 与完整 V10.24 的同协议搜索比较为主；需要解释差异时再做“提案来源×开发规则”和调度闭环检查。具体设计见最终设计第12节。
