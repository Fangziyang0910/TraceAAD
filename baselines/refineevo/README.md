# RefineEvo 基线

本实现以作者源码 [samwu-learn/RefineEvo@7de015d5](https://github.com/samwu-learn/RefineEvo/tree/7de015d5230e32b8b85fa83a3ecc55d117489138)（2026-05-27）为准，对应 ICML 2026 论文 *RefineEvo: Planning-Guided Heuristic Evolution with Bidirectional Experience*。本地参考副本位于 `reference_code/RefineEvo/`，论文源码位于 `papers/RefineEvo_Planning-Guided_Heuristic_Evolution_with_Bidirectional_Experience/`。上游仓库没有许可证文件；本地复制的提示仅用于复现实验。

RefineEvo 有三个角色。Planner 每代选择探索或利用算子，并在算子停滞时改写算子描述。Evolver 用选中的算子生成子代。Reflector 把每个子代的结果提炼为成功或失败经验，存入按 embedding 检索的经验池。

## 代码来源

| 官方部件 | 本地位置 | 用法 |
| --- | --- | --- |
| `prompts/common/` 中实际使用的 12 个提示 | `prompts/common/` | 逐字复制。消融用的 `user_operate_wo_exp.txt`、`user_sel_ops_k.txt` 不复制。 |
| 任务外部知识 | `prompts/{tsp_construct,cvrp_aco}/` | 复制 `tsp_constructive`、`cvrp_aco` 的版本。只进入初始化提示。其他任务没有外部知识，按上游写 “No additional domain knowledge provided.”。 |
| `planner_interface.py` | `planner.py` | 保留统计量、提示拼接、强制探索、算子检查与改写。 |
| `pool_interface.py` | `pool.py` | 保留成败判断、提炼、检索、计分与删除。JSON 后端改为内存列表，由检查点保存。 |
| `evo_interface.py`、`evo_agent.py`、`utils/utils.py` | `refineevo.py`、`utils.py` | 选择、种群管理、回复提取与主循环按原顺序改写为串行。 |

## 保留的官方机制

默认参数取自 `cfg/config.yaml` 与论文附录表 “Key hyperparameters”。

| 机制 | 设置 |
| --- | --- |
| 初始化 | 不评价种子。生成 int(30/10)×10 = 30 个初始个体，提示给出参考实现与外部知识。之后用 `pop_greedy` 保留 10 个。 |
| 种群管理 | `pop_greedy`：去掉无效个体；同一目标值只保留先出现的一个；取最好的 N 个。子代按 `add2pop` 去掉与种群同分者后并入。目标值按上游四舍五入到 5 位小数。 |
| 选择 | `prob_rank`：按种群排名 r 以 1/(r+1+N) 为权重有放回抽取。op1、op2 取 2 个父代，op3–op5 取 1 个。 |
| Planner 决策 | 每代一次。提示给出种群统计、最近 3 代趋势和 5 个算子描述，要求只选一类算子。连续 5 代利用后强制探索 op1、op2。回复缺少字段或没有合法算子时，使用上游启发式规则。 |
| 算子改写 | 对本代选中的每个算子，在其历史记录不少于 3 代时，先调用一次检查。检查认为需要改写时，再调用一次改写，新描述立即用于本代。 |
| 子代生成 | 每个选中算子生成 N 个子代。每个子代先检索经验，再生成。回复缺少 `{描述}` 或 python 代码块时重试，最多 5 次；代码与种群成员相同时重试，最多 3 次，第 4 个重复照常评价。 |
| 经验检索 | 查询文本只含父代的目标值与描述。取余弦相似度最高的 50 条，筛出 `operator_goal` 与当前算子描述完全相同的经验，取前 3 条，分为成功与失败两组放入提示。 |
| 经验写入 | 每个有查询的子代调用一次 Reflector。子代严格优于最好父代（容差 1e-9）时用成功提示，否则用失败提示。解析失败时存入上游的兜底条目。 |
| 经验计分 | 新经验分数为 0。被检索的经验在子代优于所有父代时 +1，否则 −1；低于 0 即删除。 |
| 终止 | 预算用完即停，不设代数上限。 |

## 本地适配

| 问题 | 本地规则与理由 |
| --- | --- |
| 预算 | 上游跑 30 代，不限评价次数；论文称各方法的评价预算为 1000。本地预算 1000 按返回的生成回复计数，含上游重试的解析失败与重复回复。Reflector、Planner 与 embedding 调用不占预算，单独记录。每次生成前检查预算，用完即停；最后一批的反思不再调用，因为搜索已结束。 |
| 并行与经验时序 | 上游在一个算子批次内用 N 个线程并行生成，各线程几乎同时检索。所以同批子代看不到彼此的经验。本地串行生成，但把同批的经验写入和计分推迟到批次末尾，使检索仍看到批次开始时的经验池。计分沿用上游的快照算法：两个子代检索到同一经验时，后写入的分数覆盖先写入的分数，不累加。 |
| 种子与任务说明 | 上游 `seed_func.txt` 作为参考实现进入初始化提示。本地使用任务模板模块。`problem_desc` 使用共同任务说明，`func_desc` 使用模板函数的 docstring。 |
| 评价 | 上游把代码写入 `gpt.py`，按函数名导入后评价。本地补 numpy 导入的规则相同，并要求模块中有同名入口函数；没有或语法错误时记为 `invalid_program`，占用预算，目标值为空，照常进入经验提炼。最终选择使用未取整的训练成绩。 |
| 模型 | 上游生成与 Planner 用 deepseek-v3，Reflector 用 deepseek-v3.1，temperature 1，不设输出上限。本地三个角色使用同一共同模型配置，输出上限 16384，与 HSEvo、ReEvo 相同。 |
| embedding | 与论文相同，使用 DashScope `text-embedding-v4`（1024 维），复用 `experiments.shinka_evo.embedding`。同一查询文本在检索时已计算向量，写入经验时复用，不再调用。上游吞掉 embedding 异常，检索返回空，经验不写入。本地改为停止搜索并保留检查点，避免凭据缺失时静默退化为无经验的搜索。 |
| 接口故障 | 上游直接调用接口，异常会结束运行。本地由公共客户端重试；非临时错误时跳过该子代位置，不占预算。Reflector 调用失败时不写入经验，与上游捕获异常后的结果相同。 |
| Planner 回复异常 | 上游在回复不是 JSON 或不是列表时抛出异常，运行中断。本地改用上游的启发式规则，并在 `strategy` 事件中记录 `fallback`。 |
| 初始化全部无效 | 上游在 Planner 统计时报错。本地结束搜索，`summary.json` 的 `termination` 写明原因。 |
| 记录 | 不写 `memory.json`、`planner_summary.json` 等文件。决策、算子改写、经验写入、计分与 embedding 调用写入 `events.jsonl`；经验池随检查点保存；`summary.json` 的 `refineevo` 字段汇总算子使用次数与经验池大小。 |
| 不使用的部件 | 消融开关（经验过滤、随机或全选算子、固定 k 个算子）、启发式停滞判断、30 代上限、最终验证脚本、token 统计器。 |

## 论文与代码不一致处

本地按代码实现。下列差异会影响对论文结果的解释：

- 论文附录说经验初始效用为 1；代码为 0。因此代码中一条新经验第一次没有帮助就会被删除。
- 论文附录描述了写入时的矛盾检查：与已有经验矛盾的新经验被丢弃，旧经验扣 1 分。代码没有这一步。
- 论文正文说 Planner 同时参考各算子的近期效用 Ψ_ops。代码的决策提示只含种群统计与算子描述；算子历史只用于改写检查。
- 检索按算子描述原文过滤。算子被改写后，旧描述下积累的经验不再能被该算子检索到。这是代码的实际行为，论文没有说明。

## 恢复

检查点保存在三类边界：开始前、每个初始子代评价后、每代结束后。检查点含种群、两份算子描述、Planner 历史、经验池（含向量）和随机状态。中断于一代中间时，恢复后从这一代开头重来，随机状态与检查点一致；已消耗的预算不退还，这一代已评价的程序仍留在记录中，参与最终选择。其他边界沿用[DeltaEvolve 的公共恢复规则](../deltaevolve/README.md#实验框架适配)。

## 使用

仅检查配置，不调用模型：

```bash
uv run python -m experiments.refineevo.run --task tsp_construct --dry-run
```

单路运行；embedding 凭据沿用项目环境配置：

```bash
uv run python -m experiments.refineevo.run \
  --task tsp_construct --budget 1000 --seed 0 --run-name BATCH_tsp_refineevo_rep1
```

六任务三路计划：

```bash
uv run python -m experiments.infra.local_queue \
  --method refineevo --suite co6 --batch BATCH --repeats 3 --slots 3 --dry-run
```

恢复使用 `experiments.infra.resume_baseline --run-dir RUN_DIR`，加 `--execute` 才继续。[离线检查](../../tests/method/test_refineevo.py)覆盖提示顺序与内容、种群管理、解析与重复重试的预算、Planner 兜底与强制探索、算子改写与经验过滤、同批经验时序与计分、embedding 故障、检查点恢复，以及六任务的完整运行入口。它们只验证机制和框架行为，不证明搜索效果。
