# TraceAAD V10.14-3

本版修复 Idea 交付契约与分数单位对开发机会的影响，并检验独立结构探索。证据、反例与验证口径见 [机制诊断](../../docs/03-机制探索与验证/2026-09-28-V10.14-3-机制诊断与修订.md)。这是联合修订实验，启动成功不代表搜索性能已改善。

## 冻结协议

- **交付**：提示给出 `Idea:`＋`code:`＋完整 Python 代码块的明确模板，不使用服务端 JSON Schema／自定义语法。优先读取显式 Idea，缺少有效标签内容时保留块外叙述；兼容块后说明、Markdown 标题和缩进的代码围栏，不再按最终代码的邻接位置过滤说明。先保存说明，再检查代码交付；代码格式失败不会抹掉已提取的 Idea。缺少独立说明时标记缺失，有效代码照常评价，不补写或免费重试。完整模块、原响应和 Idea 全文保留；下轮上下文中的 Idea 视图限真实 tokenizer 的 500 token，监控详情展示全文。
- **选父**：所有有效语法组竞争，质量为经验中分位 `(严格较差数 + 同分数/2)/组数`，加 `1/sqrt(父代使用次数+1)`。次数在每次付费尝试前持久化，失败和重复同样计数。同分按次数少、ID 早。没有每个新候选都获得预算的保证。
- **语法组**：Python AST 哈希只忽略注释和格式，保留名称、数值、docstring 与语句顺序。共享机会次数，以最早记录代表组；不按有限探针否决。源码、评价缓存、分数和形成事实仍独立保存。donor／bootstrap／最终前五名也按语法组去重。
- **动作**：可用 Refine／Tune／Pivot／Transfer 等概率。Tune 需显式数值赋值；Transfer 需另一语法组，均匀选 donor。正常 Refine 仍可结构修订。主干 Pivot 默认使用独立上下文，只含任务／接口／执行和交付约束，不含父代、历史或参考源码；记录无父代的真实根，不产生伪 diff、不计调度候选的父代次数。独立提案付费且不附赠 bootstrap。
- **上下文**：有父代的请求补充同语法程序历次付费尝试按动作汇总的结果、训练最好和当前差距。原有最近尝试与形成证据保留 4,000 token／6 事件／深度 6 的上限；原始事实不截断。`evidence_policy=none` 同时关掉新增结果摘要。解释是未验证假设，局部提升不是全局贡献。
- **其余协议**：八提案 hybrid 初始化；有效唯一语法根一次 bootstrap；候选尝试预算包含失败和重复；默认关闭在线重验、固定三步票和行为资格门槛。诊断探针不改变已有评分。训练面板 `[730241]`、原评价超时和独立选择集不变；最多五个语法不同的训练候选冻结后独立选择，再通过单独入口做 held-out。

当前批次为 `20260928_v1014_3_template2`，五任务各四路、每路 1000 次尝试，复用 -2 的 backend／seed 映射（local 3、server3 8、server3b 9），T=1、top_p=.95、top_k=20、thinking=False、输出 8192、ACO 4 workers。新建结果目录，不读取旧批次的搜索状态；不自动重启或迁移后端。


当前源码已在交付提示中加入明确示例，完整模式的段落名改为 `Idea:` 和小写 `code:`，内容由模型替换：

````text
Idea:
<Explanation of the proposed algorithm or change>

code:
```python
<Complete executable Python code>
```
````

当前源码按此模板提取：优先读取显式 `Idea`，没有有效标签内容时保留块外叙述，兼容块后说明和旧标题；多代码块不再触发说明邻接过滤。文本采集不判断解释是否正确。缺少独立说明时仍接收有效代码并标记缺失，代码选取和合法性检查继续沿用原规则。解析补丁的离线核验见 `validation_20260928/parser_simplification/`，本次重启已加载这些修改。

`edit` 提示另给 JSON 示例，包含 `mode="edit"`、`idea` 和 `edits` 的 `search/replacement`。它是可选的输出方式；当前二十路配置均为 `full`，初始化和独立 Pivot 总是输出完整代码。当前批次已经从零重新启动，实际使用上述模板与简化解析器。模板加入前的 20 路保存在 `experiments_result/traceaad_v10_14_3_before_template/`；首次模板启动在监控中发现缩进围栏误拒收，修补后重新启动，其 402 次尝试保存在 `experiments_result/traceaad_v10_14_3_template_first/`。两份归档都可从监控同名入口查看，源码与原始日志均保留。当前验证位于 `validation_20260928/restart_template2/`，首次模板启动的证据位于 `restart_template/`。

## 运行

```bash
uv run python -m experiments.traceaad_v10_14_3.run \
  --task tsp_construct --run-name trial_1 --budget 1000 --dry-run

uv run python -m experiments.traceaad_v10_14_3.launch_batch \
  --from-batch experiments_result/traceaad_v10_14_3/batch_20260928_v1014_3_template2.json \
  --batch new_unique_trial --dry-run
```

去掉 `--dry-run` 正式运行。批量入口要求旧批次会话已经退出，且新清单／运行目录不存在；部分启动不能通过整批重试覆盖。外部结果未知时拒绝自动重放。正式版本 ID 为 `v1014_3`，结果根为 `experiments_result/traceaad_v10_14_3/`。

对照开关：`--parent-policy raw_count`、`--pivot-context anchored`。这两项仍共享新版交付和语法分组，用于拆分选父与上下文作用。`--output-mode full` 为正式默认；`edit` 仅保留旧版的显式编辑对照。

## 检查与证据

```bash
uv run pytest -q tests
uv run python -m experiments.traceaad_v10_14_3.preflight \
  --output /tmp/v10143_preflight_unique
```

preflight 会产生实际 LLM 调用，使用五任务的小规模数据，每路三候选，第三次强制独立 Pivot，并执行独立选择；不能当性能对照。普通文本最终验证位于 `validation_20260928/live_text/`，提取回放位于 `text_replay.json`。已放弃的结构化探索及其失败尝试分别保存在 `validation_20260928/` 的不同目录，未覆盖或隐藏。离线诊断入口 `diagnose.py` 冻结日志边界，`audit_snapshot.py` 审计交付与语法／上下文，不调用模型也不重评代码。

已知边界：分位秩丢掉差值量级；独立提案可能浪费开发预算；AST 不是行为等价判定；普通提示不能保证每次都写说明，说明非空也不保证机制正确。判断以完整预算的四路分布、资源成本和独立评价为准。


本次重启的输出核验固定为每路前五次完成尝试（含失败，共 100 次），使用真实正式响应，不额外采样。复核命令：

```bash
uv run python -m experiments.traceaad_v10_14_3.audit_output \
  --manifest experiments_result/traceaad_v10_14_3/batch_20260928_v1014_3_template2.json \
  --output /tmp/v10143_output_audit.json --limit 5
```

该入口记录实际模板、严格格式遵循率、兜底解析、Idea 缺失和原文与归档一致性。严格格式有偏差而成功兜底，与程序执行失败分别报告；启动样本不能证明长期遵循率或性能收益。

当前批次固定 100 次核验已完成：Idea 非空 100/100，代码解析成功 99/100（1 份真实源码语法失败，说明仍保留），评价成功 97/100。独立复核的单代码块模板遵循为 83/100，其余 17 份兜底；52 份说明超过 500 token。多代码块情况下，显式段落优先会排除部分无标签中间叙述，不能把非空率称为全文保真率。实际监控、原始审计与逐项复核见 [启动核验报告](../../experiments_result/traceaad_v10_14_3/validation_20260928/restart_template2/README.md)。
