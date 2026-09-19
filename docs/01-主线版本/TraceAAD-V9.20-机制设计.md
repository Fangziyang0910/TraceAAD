# TraceAAD V9.20 完整机制设计（历史版本）

## 1. 任务对象

自动算法设计（AAD）在一个有明确 evaluator 的问题上，反复请求语言模型改写算法函数，并用有限的真实评价预算选择更好的函数。一次改写机会的完整对象是：

1. 一个已经被 evaluator 测量过的算法状态；
2. 一次是否值得继续购买改写机会的选择；
3. 一条与该选择匹配的上下文和操作指令；
4. 一个 `Idea + Code` 候选、一次真实评价和一次状态更新。

V9.20 将机制压缩为两个职责：

- **Opportunity Allocation**：决定下一次机会给哪个已测量状态；
- **Assisted Decision**：决定这次机会采用 Develop、Explore 还是 Crossover，并提供与 action 匹配的证据。

BehaveSim 是执行行为距离。它用于行为覆盖和参考检索，不被解释为语义算法簇，也不单独构成质量或因果证据。

## 2. 状态对象

每个有效节点 `a` 保存：

- `code`, `fitness`, `parent_id`, `idea`, `action`；
- `novelty`, `behavior_tag` 以及完整的形成路径；
- `opportunities(a)`：从该节点发出的主改写机会数；
- `improvements(a)` 与 `failures(a)`：这些机会的最终结果；
- `last_outcome(a)`：最近一次机会的最终结果。

形成路径记录算法如何来到当前状态；直接 outcome ledger 记录是否值得再购买一次机会。两者职责不同：形成路径用于解释当前方向，ledger 用于预测下一次直接改写的价值。

## 3. Opportunity Allocation

设当前有效节点集合为 `A_t`，质量分位为 `Q_t(a)`，直接延续价值为：

`C_t(a) = (1 + improvements(a)) / (2 + opportunities(a))`

新节点的 `C` 为 `0.5`，避免把没有证据误认为高价值。质量和延续价值形成主要机会分布：

`H_t(a) = 0.5 Q_t(a) + 0.5 C_t(a)`

使用 Boltzmann 分布：

`p_quality(a) = exp(beta H_t(a)) / sum_b exp(beta H_t(b))`

`beta` 由目标 ESS 求解。目标为：

`ESS_target = min(|A_t|, max(0.10 |A_t|, 2))`

行为覆盖分布只看节点及其 BehaveSim 邻域已经被购买过多少机会。令：

`R_t(a) = sum_{b in {a} union N_t(a)} opportunities(b)`

则欠覆盖原始值为 `1 / (1 + R_t(a))`，归一化后得到 `p_coverage(a)`。最终父节点分布是显式混合：

`p_parent(a) = 0.80 p_quality(a) + 0.20 p_coverage(a)`

因此 BehaveSim 只在一个明确的位置承担覆盖责任，不会同时被重复塞进质量、轨迹响应和 action 概率。

## 4. Assisted Decision

先为选中的父节点检索一个行为上有差异、质量上有潜力的参考节点。参考值在候选参考集合中按质量分位和距离分位计算：

`V_ref = 0.5 rank_quality(reference) + 0.5 rank_distance(reference)`

随后计算三个 action utility：

- `Develop`: `U_D = C_t(a)`；
- `Explore`: `U_E = 1 / sqrt(1 + opportunities(a))`；
- `Crossover`: `U_X = V_ref`，没有参考节点时移除该 action。

以温度 `0.35` 的 softmax 采样 action。这个采样只回答“怎样使用已经购买的机会”，不改变父节点分布。

### 4.1 Develop 上下文

提供：

- 当前算法完整代码和 fitness；
- 最近形成路径；
- 直接 outcome ledger；
- Develop 指令。

要求沿着已有证据支持的方向做一个连贯改进，保留有效结构。

### 4.2 Explore 上下文

提供：

- 当前算法完整代码和 fitness；
- 压缩后的直接 outcome ledger；
- Explore 指令。

不把完整成功形成路径放入 Explore 上下文，减少路径锚定；同时保留失败摘要，避免盲目重复已知死路。

### 4.3 Crossover 上下文

提供：

- 当前算法和形成路径；
- 参考算法完整代码、fitness、行为标签和距离；
- 参考算法形成路径；
- Crossover 指令。

要求迁移一个有用机制并解释接口兼容性，形成一个可执行混合，而不是粘贴参考代码。

## 5. 原子循环

```text
evaluate initial roots until n_roots valid states exist
while primary_evaluations < budget:
    profile = tracked evaluator(parent candidates)
    stats = allocation statistics(profile, quality, direct ledgers)
    parent = sample_parent(stats)
    reference = retrieve_behavior_different_reference(parent)
    action = sample_action(parent, reference)
    prompt = action_matched_prompt(parent, reference, path, ledger)
    candidate = LLM(prompt)
    evaluate candidate once
    if candidate fails and repair_count < 2:
        issue bounded repair using the failure report
        evaluate the repair for the same primary slot
    commit only the final result for this slot
```

一次主机会只增加一个 `budget_slots`。修复评价增加 `evaluator_call_count`，但不增加主预算、父节点机会数或节点数。重复代码仍然执行真实 evaluator，随后以 `duplicate` 结果结算。

## 6. 行为测量

每个 tracked evaluator 与正式 evaluator 使用相同训练实例、随机顺序和 fitness 聚合，只额外保留每个实例的有限轨迹。TSP 轨迹从已经包含 depot sentinel 的 route 直接取前缀，避免重复插入 depot。BehaveSim 协议、距离矩阵和 profile 会写入 checkpoint。

## 7. 工件与恢复

每个 run 写入：

- `evaluations.csv`：主槽位、修复尝试、分配值、action 概率和结果；
- `decisions.jsonl`：完整 prompt、响应、形成路径和决策快照；
- `mechanism_events.jsonl`：分配、action、参考检索和新节点事件；
- `checkpoints/latest.json`、`behave.npz`、`view.json`；
- `logs/summary.json` 与 `best_history.jsonl`。

checkpoint 在每次 LLM 响应之后、每次结算之后写入。若进程在评价前退出，恢复会结算 checkpoint 中记录的同一个 attempt；不会再购买新的主机会。

## 8. 固定参数

| 参数 | 值 |
| --- | --- |
| primary budget | 1000（runner 默认） |
| initial roots | 8 |
| coverage mix | 0.20 |
| ESS fraction | 0.10 |
| minimum ESS | 2 |
| action temperature | 0.35 |
| max repairs | 2 |
| history events | 6 |
