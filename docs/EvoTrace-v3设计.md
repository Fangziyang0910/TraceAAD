# EvoTrace V3：树状种群与反思简化

V3 以演化树组织算法派生历史与经验归属，以覆盖代表集组织空间竞争与预算分配。节点保存受评算法与**节点经验**（`node.experience`），有向边记录派生关系与**边经验**（`edge.experience`）。代表集中的可行算法按名次质量与自身直接开发投入竞争父代机会；**反思树简化（Reflective Tree Simplification）**通过路径收缩与子树剪枝维护树结构，将被简化区域的试错经历凝练到对应边和节点上。研究主线见[树状种群与反思简化](树状种群与反思简化.md)。

实现位于独立包 [`evotrace_v3/`](../evotrace_v3/README.md)，仅依赖 Python 标准库，不依赖历史版本或其他方法包。

## 1. 演化树表示与评价记账

每个受评尝试的最终算法版本作为节点挂入 `TreePopulation`：
- `Node` 保存源码 `code`、事实性计算描述 `design`、实测适应度 `fitness`（越小越优，`None` 表示不可行）或错误摘要 `failure`、节点经验 `experience`、原始比较记录 `observations`，以及以此个体为主父代直接消耗的评价次数 `direct_evaluations`。
- `Edge` 保存主父代编号 `parent_id`、生成算子 `operator`、交叉参考编号 `reference_id`、收缩标记 `contracted`、边经验 `experience` 与原始比较记录 `observations`。虚拟根（`root_id = 0`）单独保存初始阶段归并的 `root_experience` 与 `root_observations`。

`Init` 生成的个体挂在虚拟根下（`operator = None`）；`Refine`、`Explore` 与 `Crossover` 生成的子代挂在所选主父代下，`Crossover` 参考个体仅提供上下文而不建立结构边。

每次包含程序评价的尝试完成时（含运行失败及修复重评），按实际执行的评价次数推进全局 `evaluations_used`，并将该次数累加到主父代的 `direct_evaluations`。新节点的 `direct_evaluations` 从 `0` 起计；未进入程序评价的请求错误、格式修复与反思调用不推进评价时钟，其调用与词元用量单列记录。

## 2. 覆盖代表集与直接投入父代分配

首先，`distinct_individuals` 将标准化源码（`identity(code)`）与实测适应度均相同的可行节点折叠为最早编号（保留原节点的 `direct_evaluations`）。随后，`representatives` 按 `(fitness, id)` 从优到劣贪心扫描独立可行个体：若候选 $i$ 与所有已入选代表 $r \in R$ 均满足 $\operatorname{code\_distance}(i, r) \ge \delta$（`cover_distance`，默认 `0.15`），则将 $i$ 加入代表集 $R$，直至 $|R| = K$（`max_representatives`，默认 `24`）。全局最优个体必排在 $R$ 首位。

父代仅在代表集 $R$ 上抽样：

$$w_r = \frac{2^{-\operatorname{rank}_R(r) / H}}{1 + n_r}, \qquad p_r = \frac{w_r}{\sum_{u \in R} w_u},$$

其中 $\operatorname{rank}_R(r)$ 为代表 $r$ 在 $R$ 中相对最优并列组的平均名次差，$H = \text{rank\_half\_life}$（默认 `10.0`）为质量权重减半的代表名次跨度，$n_r$ 为 `tree.nodes[r].direct_evaluations`：
- **未突破近邻自动覆盖：** 与已选更优代表结构距离小于 $\delta$ 的候选不进入 $R$，不会挤占其他区域代表的名次，也不会以零计数反超父代；其消耗的评价次数已计入父代 $n_r$。
- **局部改进与跨区探索即时接棒：** 更优近邻（含常数调优，`code_distance == 0`）若未被更高名次代表覆盖而进入 $R$，即覆盖旧代表并以 $n = 0$ 跟进开发；结构距离均不小于 $\delta$ 且排入前 $K$ 的新子代（无论来自 `Explore`、`Crossover` 还是大幅 `Refine`）作为独立代表以 $n = 0$ 参选，不继承祖先的历史投入，同时保留树上的真实父代连接。
- **冷却不触发过期淘汰：** 代表不会因采样冷却而过期删除，仅在被更优代表覆盖或跌出前 $K$ 时退出 $R$；退出后若不属于分叉骨架则作为非骨架节点等待成批简化。

选定主父代后，按 `0.45 / 0.30 / 0.25` 抽取 `Refine`、`Explore` 或 `Crossover`。执行 `Crossover` 时，在 $R \setminus \{\text{parent}\}$ 中排除标准化源码相同的个体，保留秩质量不低于 `reference_quality_floor`（默认 `0.30`）的代表，优先挑选未配对过的 `(parent_id, reference_id)` 组合并取 AST 结构距离最大者；无合格参考时回退为 `Refine`。

## 3. 反思树简化（路径收缩与子树剪枝）

初始化集齐 `initial_size`（默认 `5`）个可行个体后，每次受评个体入树均执行树简化检查（`TopologyConfig`：`max_representatives = 24`，`cover_distance = 0.15`，`region_size = 4`）：

1. **确定保留骨架：** 先选出代表集 $R$，再对 $R$ 求两两最近公共祖先闭包得到分叉节点集 $F$，保留骨架为 $R \cup F$。由于 $F$ 在 $R$ 选定之后补齐，纯结构分叉节点 $F \setminus R$ 不会在贪心覆盖中阻挡更优候选，也不参与父代抽样。
2. **划分路径与子树区域：** 其余非骨架节点在树拓扑上唯一归入两类互不相交的区域：
   - **路径收缩（`PathContraction`）：** 连接两个保留骨架端点（或虚拟根与骨架端点）的中间链 `removed_ids`，连同挂载在该中间链上的未保留分支 `branch_ids`。
   - **子树剪枝（`SubtreePruning`）：** 其余未保留节点按最近保留骨架祖先（或虚拟根）归集为待剪除子树 `removed_ids`。
3. **成批简化与幂等性：** 仅当路径区域节点总数 `len(removed_ids) + len(branch_ids) >= region_size` 或子树区域节点数 `len(removed_ids) >= region_size` 时，执行拓扑简化并发起一次反思调用；未满阈值的区域暂留在树上，因其位于 $R$ 之外，既不改变 $R$ 也不影响父代抽样概率。移除非 $R$ 节点不会改变贪心覆盖结果，单轮简化严格幂等。

简化重连边标记 `contracted = True`（`operator` 与 `reference_id` 置空），所有存活节点的 `direct_evaluations` 原样保留。

## 4. 节点经验与边经验合成

每个达到阈值的简化区域调用一次结构化反思模型（严格 JSON 模式输出单一 `"experience"` 字段，约 120 英文词）：
- **路径收缩区域**（`build_edge_reflection_prompt`）读取起止代码及净差分、中间步骤代码与分数变化、沿途剪除分支差分、同起点存活参照（`[Related Results]`）、携带的既有经验以及当次区域的比较表（`[Comparison Catalog]`），合成**边经验**写入重连边的 `edge.experience`。
- **子树剪枝区域**（`build_node_reflection_prompt`）读取保留父代基准代码、各分支相对真实父代的代码差分与实测结果/报错、同起点存活参照、既有经验以及当次区域的比较表，合成**节点经验**写入保留父代的 `node.experience`（或 `tree.root_experience`）。

原始比较元数据保存在节点与边的 `observations` 列表中供审计与无模型回退使用；`[Comparison Catalog]` 仅向模型呈现当次简化区域的比较项。若反思调用异常，系统自动根据比较记录与既有笔记生成有长上限的确定性回退摘要。

## 5. 面向算子的树上下文读取

| 算子 | 树上下文内容 |
| --- | --- |
| `Refine` | 当前算法源码与分数、从根到当前节点的演化路径（未收缩边展示代码 diff，收缩边展示边经验 `edge.experience`，沿途节点展示节点经验 `node.experience`）、初始根经验及当前节点的直接子尝试。 |
| `Crossover` | 主父代与参考代表两份源码及分数、最近公共祖先及其节点经验、自公共祖先分叉后的两条演化路径与沿途经验、主父代的直接子尝试。 |
| `Explore` | 按适应度排序的树上可行个体概览（`design`、分数、来源及附着的边/节点经验，含初始根经验），后附所选主父代源码作为一份现有实现对照。 |

生成提示默认输入上限为 `30,000` tokens。超限时自动折半缩减可选展示窗口，始终保留核心源码与差分基准。将 `ContextConfig(include_experiences=False)` 设为 `False` 可在不直接修改当前树拓扑与抽样公式的前提下隐藏提示中的节点与边经验（严格控制展示条数的固定状态对照可同时固定可选窗口参数）。

## 6. 运行、恢复与边界

统一实验入口（`--method evotrace.v3` / `evotrace_v3`）与独立 CLI（`python -m evotrace_v3`）均支持 `--max-representatives`、`--cover-distance`、`--region-size` 与 `--rank-half-life`（将 `cover_distance` 设为 `0.0` 即退化为无结构覆盖的前 $K$ 个体对照）。`attempts.jsonl` 是唯一权威恢复日志，发生简化的步骤保存更新后的完整树快照，恢复时确定性重放而不重复调用反思；`run_config.json` 严格校验机制列表、选择规则（`representative_rank_direct_sampling`）、经验表示（`node_and_edge_experience`）及拓扑/分配配置，防止不同版本混跑。

已知边界：
- **AST 结构距离与覆盖非传递性：** `code_distance` 以语法结构二元组近似算法差异，结构相近但需连续退步改写才能突破的变体会被当前更优代表覆盖；覆盖关系不具传递性，且 `distinct_individuals` 仅在当前保留树内折叠精确重复（已被简化移除的程序若在后续搜索中重现，按新节点入树，不维护全局永久黑名单）。
- **反思归纳边界：** 单步自然语言反思是有损归纳，面对多处同时修改的代码仍可能出现归因偏差。当前对照批次与已验证边界见[实验与边界台账](记录/README.md)。
