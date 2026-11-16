# EvoTrace V3：树状种群与反思简化

V3 以演化树统一组织算法种群、经验压缩与生成上下文。节点保存受评算法与**节点经验**（`node.experience`），有向边记录派生关系与**边经验**（`edge.experience`）。树上可行个体按全局名次质量与沿演化路径共享的子树评价投入竞争开发机会；**反思树简化（Reflective Tree Simplification）**通过路径收缩与子树剪枝维护种群，将被简化区域的试错经历凝练到对应边和节点上。研究主线见[树状种群与反思简化](树状种群与反思简化.md)。

实现位于独立包 [`evotrace_v3/`](../evotrace_v3/README.md)，仅依赖 Python 标准库，不依赖历史版本或其他方法包。

## 1. 演化树表示与评价记账

每个受评尝试的最终算法版本作为节点挂入 `TreePopulation`：
- `Node` 保存源码 `code`、事实性计算描述 `design`、实测适应度 `fitness`（越小越优，`None` 表示不可行）或错误摘要 `failure`、节点经验 `experience`、原始比较记录 `observations`，以及评价统计字段 `subtree_evaluations`、`last_used`（另附不参与决策的诊断计数 `direct_evaluations`）。
- `Edge` 保存主父代编号 `parent_id`、生成算子 `operator`、交叉参考编号 `reference_id`、收缩标记 `contracted`、边经验 `experience` 与原始比较记录 `observations`。虚拟根（`root_id = 0`）单独保存初始阶段归并的 `root_experience` 与 `root_observations`。

`Init` 生成的个体挂在虚拟根下（`operator = None`）；`Refine`、`Explore` 与 `Crossover` 生成的子代挂在所选主父代下，`Crossover` 参考个体仅提供上下文而不建立结构边。

每次包含程序评价的尝试完成时（含运行失败及修复重评），按实际执行的评价次数推进全局 `evaluations_used`，更新主父代的 `last_used`（`Crossover` 同时刷新参考个体的 `last_used`），并将评价次数累加到主父代及其当前保留祖先链上所有节点的 `subtree_evaluations`。新节点的 `subtree_evaluations` 从 `0` 起计；未进入程序评价的请求错误、格式修复与反思调用不推进评价时钟，其调用与词元用量单列记录。

## 2. 全局名次与谱系投入父代分配

每次演化生成前，`distinct_individuals` 将标准化源码（`identity(code)`）与实测适应度均相同的可行节点折叠为最早编号；全体独立可行个体 $V^*$ 按适应度从优到劣计算平均并列名次（`quality_scores`）。同时，`lineage_peaks` 沿当前保留边找到每个可行个体 $i$ 从根到自身路径上适应度最优的可行祖先（同分取更早编号）：

$$a(i) = \arg\min_{u \in \operatorname{path}(\text{root} \to i),\, u \text{ feasible}} (f_u, u).$$

全体 $i \in V^*$ 按同一概率分布抽样主父代：

$$w_i = \frac{2^{-r_i / H}}{1 + s_{a(i)}}, \qquad p_i = \frac{w_i}{\sum_{j \in V^*} w_j},$$

其中 $r_i$ 为个体 $i$ 相对当前最优并列组的平均名次差，$H = \text{rank\_half\_life}$（默认 `10.0`）为质量权重减半的名次跨度，$s_{a(i)}$ 为 `tree.nodes[a(i)].subtree_evaluations`。

这一权重同时结合种群全局择优与演化树分支探索：
- **同路径未突破变体共享投入：** 在当前树状态下，未严格超越 $a(i)$ 的后代（含退步、局部回升未破原纪录及同分改写）与 $a(i)$ 共享同一个 $1 + s_{a(i)}$ 分母；组内严格按实测名次排序，未改进后代不会因自身计数为零而获得高于父代的权重，且同一路径峰值 $a$ 名下所有未突破个体的未归一化权重之和受峰值自身的基础质量权重 $2^{-r_a / H}$ 约束（因每保留一个后代至少向 $s_a$ 计入一次评价）。
- **路径新突破即时跟进与分支轮换：** 一旦后代严格刷新路径最优适应度，即有 $a(i) = i$ 并以 $s_i = 0$ 参选；持续消耗预算而未再刷新纪录的分支则随 $s_{a(i)}$ 增长逐步降低抽样权重，将预算让渡给投入较少的竞争分支。

选定主父代后，按 `0.45 / 0.30 / 0.25` 抽取 `Refine`、`Explore` 或 `Crossover`。执行 `Crossover` 时，排除主父代及标准化源码相同的节点，在秩质量不低于 `reference_quality_floor`（默认 `0.30`）的候选中优先挑选未配对过的 `(parent_id, reference_id)` 组合，并取 AST 结构距离（`code_distance`）最大者；无合格参考时回退为 `Refine`。

## 3. 反思树简化（路径收缩与子树剪枝）

初始化集齐 `initial_size`（默认 `5`）个可行个体后，每次受评个体入树均执行树简化检查（`TopologyConfig`：`active_evaluations = 40`，`region_size = 4`）：

1. **保留活跃个体与分叉骨架：** 距最近一次使用（`evaluations_used - node.last_used < active_evaluations`）不足 40 次评价的独立可行个体，连同全局最优个体（`champion`），组成活跃集合 $A$；对 $A$ 求两两最近公共祖先闭包得到分叉节点集 $F$，保留骨架为 $A \cup F$。
2. **划分路径与子树区域：** 其余节点在原树拓扑上唯一归入两类互不相交的区域：
   - **路径收缩（`PathContraction`）：** 连接两个保留骨架端点（或虚拟根与骨架端点）的中间链 `removed_ids`，连同挂载在该中间链上的未保留分支 `branch_ids`。
   - **子树剪枝（`SubtreePruning`）：** 其余未保留节点按最近保留祖先（或虚拟根）归集为待剪除子树 `removed_ids`。
3. **成批简化与经验凝练：** 仅当路径区域节点总数 `len(removed_ids) + len(branch_ids) >= region_size` 或子树区域节点数 `len(removed_ids) >= region_size` 时，执行拓扑简化并发起一次反思调用；未满阈值的区域原样留在树中，待后续尝试自然积累后再一并整理。

简化重连边标记 `contracted = True`（`operator` 与 `reference_id` 置空），所有存活节点的 `subtree_evaluations` 与 `last_used` 原样保留，被简化节点的历史评价不会重复计入或转嫁给保留后代。

## 4. 节点经验与边经验合成

每个达到阈值的简化区域调用一次结构化反思模型（严格 JSON 模式输出单一 `"experience"` 字段，约 120 英文词）：
- **路径收缩区域**（`build_edge_reflection_prompt`）读取起止代码及净差分、中间步骤代码与分数变化、沿途剪除分支差分、同起点存活参照（`[Related Results]`）、携带的既有经验以及当次区域的比较表（`[Comparison Catalog]`），合成**边经验**写入重连边的 `edge.experience`。
- **子树剪枝区域**（`build_node_reflection_prompt`）读取保留父代基准代码、各分支相对真实父代的代码差分与实测结果/报错、同起点存活参照、既有经验以及当次区域的比较表，合成**节点经验**写入保留父代的 `node.experience`（或 `tree.root_experience`）。

原始比较元数据保存在节点与边的 `observations` 列表中供审计与无模型回退使用；为防止多轮简化导致反思输入随历史线性膨胀，`[Comparison Catalog]` 仅向模型呈现当次简化区域的比较项。若反思调用异常，系统自动根据比较记录与既有笔记生成有长上限的确定性回退摘要。

## 5. 面向算子的树上下文读取

| 算子 | 树上下文内容 |
| --- | --- |
| `Refine` | 当前算法源码与分数、从根到当前节点的演化路径（未收缩边展示代码 diff，收缩边展示边经验 `edge.experience`，沿途节点展示节点经验 `node.experience`）、初始根经验及当前节点的直接子尝试。 |
| `Crossover` | 主父代与参考算法两份源码及分数、最近公共祖先及其节点经验、自公共祖先分叉后的两条演化路径与沿途经验、主父代的直接子尝试。 |
| `Explore` | 按适应度排序的树上可行个体概览（`design`、分数、来源及附着的边/节点经验，含初始根经验），后附所选主父代源码作为一份现有实现对照。 |

生成提示默认输入上限为 `30,000` tokens（接入调度器时按服务端分词器精确计数，独立运行时按 UTF-8 字节保守估算）。超限时自动折半缩减可选展示窗口，始终保留核心源码与差分基准。将 `ContextConfig(include_experiences=False)` 设为 `False` 可在不改变树拓扑和抽样轨迹的前提下纯粹隐藏节点与边经验，用于消融对照。

## 6. 运行、恢复与边界

统一实验入口（`--method evotrace.v3` / `evotrace_v3`）与独立 CLI（`python -m evotrace_v3`）均支持 `--active-evaluations`、`--region-size` 与 `--rank-half-life`。`attempts.jsonl` 是唯一权威恢复日志，发生简化的步骤保存更新后的完整树快照，恢复时确定性重放而不重复调用反思；`run_config.json` 严格校验机制列表、选择规则（`rank_decay_lineage_sampling`）、经验表示（`node_and_edge_experience`）及拓扑/分配配置，防止不同版本混跑。

已知边界：
- **个体代码去重与单路径峰值计费：** `distinct_individuals` 仅合并代码与分数均相同的节点，`lineage_peaks` 按单条保留路径判定最优祖先；在离散或平台型目标上，多条路径到达同一平台分时会各自计算峰值投入，窗口内的同分改写也会压低后排个体名次。
- **路径收缩下的计费祖先重算：** 路径收缩保留存活节点原有的 `subtree_evaluations`，但若非全局最优的路径峰值因超出活跃窗口且非分叉点而被收缩移除，其后代会在当前保留树上重新确定路径最优祖先。
- **探索挂接与反思归纳边界：** `Explore` 子代挂在所选主父代下，未超越路径最优祖先时共享其累计投入；单步自然语言反思是有损归纳，不等同于精确的因果归因。当前批次与已验证边界见[实验与边界台账](记录/README.md)。
