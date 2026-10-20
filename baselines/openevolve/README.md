# OpenEvolve 基线

本实现以官方源码 [algorithmicsuperintelligence/openevolve@34427bae](https://github.com/algorithmicsuperintelligence/openevolve/tree/34427bae6239063d46be0d38e0fabe11279d05aa)（v0.4.0，2026-10-20）为准。OpenEvolve 是社区对 AlphaEvolve 的开源复现，不是 DeepMind 官方代码。本地参考副本位于 `reference_code/OpenEvolve/`。

## 代码来源

`upstream/` 保存官方模块，许可证为 Apache-2.0，随代码保留。除 `config.py` 外，各文件只把 `openevolve.` 导入改为 `baselines.openevolve.upstream.`。`config.py` 只保留 `PromptConfig` 与 `DatabaseConfig` 两个数据类，内容逐字相同；YAML 加载与 LLM 配置不使用，因此不引入 `dacite`。2026-10-20 已逐文件比对：去掉导入改写后，10 个模块、14 个提示文件和许可证与官方提交相同。

| 官方部件 | 本地位置 | 用法 |
| --- | --- | --- |
| `ProgramDatabase`：MAP-Elites 网格、岛、archive、父代与灵感程序采样、迁移 | `upstream/database.py` | 直接使用。子类只记录多样性缓存键对应的代码，用于检查点恢复。 |
| `PromptSampler` 与默认模板 | `upstream/prompt/`、`upstream/prompts/defaults/` | 直接使用。 |
| SEARCH/REPLACE 解析、应用与变更摘要 | `upstream/utils/code_utils.py` | 直接使用。 |
| 控制器与 worker 的单次迭代 | `openevolve.py` | 按 `process_parallel.py` 的顺序改写为串行循环。 |

## 保留的官方机制

默认参数取自 `DatabaseConfig`、`PromptConfig` 与 `configs/default_config.yaml`。

| 机制 | 设置 |
| --- | --- |
| 岛与容量 | 5 个岛，population 1000，archive 100。初始程序只放入岛 0。 |
| MAP-Elites | 特征为 `complexity`（代码长度）与 `diversity`（与参照集的快速代码差异），各 10 箱，min-max 缩放；参照集 20 个程序。 |
| 父代采样 | 在指定岛内：0.2 均匀随机，0.7 从 archive 中优先取本岛程序，0.1 按成绩加权。 |
| 提示内容 | 父程序完整代码与成绩；本岛前 3 名作为 Previous Attempts，前 5 名中前 3 个作为 Top Programs、后 2 个随机作为 Diverse Programs；灵感程序为 2 个，含岛最好程序、近邻格和随机程序，已展示者去重。 |
| 生成方式 | diff 模式。依次检查：分隔符非法、没有 diff、所有 SEARCH 都不匹配、代码未改变、代码超过 10000 字符。任一条件成立，本次迭代结束，不评价。部分 SEARCH 匹配时照常应用匹配部分。 |
| 失败程序 | 评价失败的程序也加入数据库，可成为父代。若它被选为父代，提示中显示其执行输出。 |
| 迁移 | 每个岛在新程序加入后增加 generation。最大 generation 比上次迁移多 50 时，各岛前 10%（至少 1 个）复制到相邻两岛；目标岛已有相同代码时跳过，迁移副本不再迁移。 |
| 模型组合 | 默认单一模型。指定 `--secondary-model` 时，每次生成按 0.8/0.2 选择主、副模型，同于官方默认配置。 |

## 本地适配

| 问题 | 本地规则与理由 |
| --- | --- |
| 成绩方向 | 官方最大化 `combined_score`。本地令 `combined_score = -fitness`，与 Shinka、DeltaEvolve 的口径一致。官方失败程序记为 `{"error": 0.0}`，得分 0，低于所有正分；本地分数为负，因此失败记为 `combined_score = -inf`，保持“失败排在所有有效程序之后”。系统消息附一句说明这一换算。 |
| 负分的副作用 | 官方加权采样使用 `max(score, 0.001)`，提示中的程序类型和特征标签使用 0.3、0.8、0.9 等绝对阈值，均假设分数在 [0, 1] 附近。负分下，加权采样（占父代抽样的 10%）退化为岛内均匀抽样，标签固定为 Exploratory 与 Alternative。这些是官方代码对负分的实际行为，本地不修改。 |
| 系统消息 | 官方示例由用户配置写入任务。本地保留默认系统消息，再接共同任务说明与成绩换算句。 |
| 执行输出 | 官方评价器异常时保存 `stderr`、`traceback` 和 `failure_stage`，超时另有 `timeout` 字段。本地保存评价器的错误信息作为 `stderr`，不提供本地评价框架的 traceback；超时保留官方字段。按协议，错误信息中的实例数量与实例编号被删去，实测耗时与上限保留。 |
| 并行与快照 | 官方默认 4 个进程并行，至多约 5 个迭代同时在途，各自读取提交时的数据库快照。本地串行执行，每次迭代读取最新数据库。岛按迭代顺序轮转，对应官方默认控制器在结果按序返回时的分配。快照只携带前 100 个程序的执行输出，本地同样只向这些程序展示执行输出。 |
| 评价 | 本地评价器只有一个阶段，对应官方在评价器缺少 `evaluate_stage1` 时的直接评价。官方只在评价器抛出异常时重试；本地评价器以结果返回失败，且实例种子固定，因此不重试。LLM 反馈、embedding 新颖性检查、元提示均为官方默认关闭，本地不启用。 |
| 采样参数 | 官方默认 temperature 0.7、top_p 0.95、输出上限 4096；本地使用共同模型配置，输出上限放宽到 16384，与 TraceAAD、Shinka 相同，避免截断成为失败来源。 |
| 预算 | 种子评价和每个返回的生成结果各占一次预算，包括未通过 diff 检查的结果；接口故障不占预算。官方 `max_iterations` 不计种子，因此预算 1000 对应种子加 999 次迭代。 |
| 不使用的部件 | 提示日志 `log_prompts`、evolution trace 和磁盘数据库不进入搜索；调用与事件由公共记录器保存。执行输出全部保存在内存与检查点中，不写入工作目录。 |

## 可复现性与恢复

官方数据库用字符串集合保存岛与 archive，采样时对集合取列表。集合顺序依赖 Python 字符串哈希，而哈希在不同进程间随机，因此同一种子在不同进程中的搜索轨迹不完全相同。官方实现也是如此，本地不修改。

检查点保存完整数据库、待评价程序与随机状态。多样性缓存按字符串哈希索引，检查点改用代码作为键，恢复时重新计算哈希。中断于评价中时，恢复后完成这次评价，不重新生成，也不再次扣预算。由于集合顺序问题，恢复后的轨迹与不中断的运行可能不同，但不会重复评价历史程序。其他边界沿用[DeltaEvolve 的公共恢复规则](../deltaevolve/README.md#实验框架适配)。

## 使用

仅检查配置，不调用模型：

```bash
uv run python -m experiments.openevolve.run --task tsp_construct --dry-run
```

单路运行：

```bash
uv run python -m experiments.openevolve.run \
  --task tsp_construct --budget 1000 --seed 0 --run-name BATCH_tsp_openevolve_rep1
```

六任务三路计划：

```bash
uv run python -m experiments.infra.local_queue \
  --method openevolve --suite co6 --batch BATCH --repeats 3 --slots 3 --dry-run
```

恢复使用 `experiments.infra.resume_baseline --run-dir RUN_DIR`，加 `--execute` 才继续。[离线检查](../../tests/method/test_openevolve.py)覆盖提示结构、失败程序的排序与执行输出、diff 拒绝与预算、岛轮转与迁移、检查点恢复，以及六任务的完整运行入口。它们只验证机制和框架行为，不证明搜索效果。
