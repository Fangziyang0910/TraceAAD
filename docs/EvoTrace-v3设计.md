# EvoTrace V3：树状种群与反思简化

论文方法名为 **TREES（Tree-structured Reflective Evolutionary Search）**，标题为 *Reflective Elimination: Accumulating Design Experience in Tree-Structured Algorithm Populations*。本文保留EvoTrace V3作为实现代号；核心更新在论文中称为反思式淘汰（Reflective Elimination）。

V3通过树状种群，使算法的优胜劣汰同时成为积累和复用算法设计经验的过程。选择规则确定保留节点，演化关系据此划分待简化的路径区域与子树区域。反思结合区域内的算法变化、评价结果与已有经验，形成边经验或节点经验；路径收缩和子树剪枝移除算法节点，将经验保存在重连边或保留节点上。后续反思继续整合已有经验，生成算子结合相关算法及其演化轨迹读取经验。

本文档给出该过程的精确规则，依次说明表示、种群简化、经验合成、预算分配与生成上下文。研究论证见[研究概述](树状种群与反思简化.md)，实验安排见[实验方案](EvoTrace-v3实验方案.md)。实现位于独立包 [`evotrace_v3/`](../evotrace_v3/README.md)，仅依赖Python标准库。

## 1. 演化树表示与评价记账

每次生成经过程序评价后，将该次生成最终得到的算法作为节点加入 `TreePopulation`：

- `Node` 保存源码 `code`、模型生成的计算描述 `design`（约 60 英文词）、实测适应度 `fitness`（越小越优，`None` 表示未通过评价）或错误摘要 `failure`、节点经验 `experience`，以及以此个体为主父代直接消耗的评价次数 `direct_evaluations`。本文档中的可行程序指通过评价器检查并返回有限目标值的程序。
- `Edge` 保存主父代编号 `parent_id`、生成算子 `operator`、交叉参考编号 `reference_id`、收缩标记 `contracted` 与边经验 `experience`。虚拟根（`root_id = 0`）单独保存从零设计谱系剪枝归并的 `root_experience`。

`Init` 与 `Explore` 生成的个体均挂在虚拟根下（`parent_id = 0`，开启新谱系）；虚拟根没有算法实现或分数。`Refine` 与 `Crossover` 生成的子代挂在所选主父代下，`Crossover` 参考个体仅提供上下文而不建立结构边，`Explore` 直接从代表集选取要展示的已有做法。未收缩边记录生成关系，收缩边记录跨越已移除节点的祖先关系。

每次包含程序评价的尝试完成时（含运行失败及修复后的再次评价），实际执行的评价次数计入总评价预算；当子代挂在算法主父代下（`Refine` 或 `Crossover`）时，该次数同时累加到主父代的 `direct_evaluations`。新节点的 `direct_evaluations` 从 `0` 起计；`Init` 与挂接在虚拟根下的 `Explore` 不增加任何代表的 `direct_evaluations`。每次生成总共至多修复一次，格式修复与运行修复共用这个机会。最终被评价的程序作为一个节点入树，即使其评价失败；同次生成中的两次评价不会形成两个节点。未进入程序评价的请求错误、格式修复与反思调用不消耗评价预算，其调用与词元用量单列记录。

## 2. 算法选择与树简化

首先，`distinct_individuals` 对标准化源码（`identity(code)`）与实测适应度均相同的可行节点，仅保留最早编号节点作为代表候选，其 `direct_evaluations` 不变。随后，`representatives` 按 `(fitness, id)` 从优到劣贪心扫描去重后的可行个体：若候选 $i$ 与所有已入选代表 $r \in R$ 均满足 $\operatorname{code\_distance}(i, r) \ge \delta$（`cover_distance`，默认 `0.15`），则将 $i$ 加入代表集 $R$，直至 $|R| = K$（`max_representatives`，默认 `24`）。存在可行节点时，适应度最优且编号最早者排在 $R$ 首位。

`identity` 比较忽略排版、注释和文档字符串后的语法树。`code_distance` 使用AST父子标签对与深度优先遍历二元组的多重集Jaccard距离；标签保留节点类别、运算符类型与被调用的函数名或属性名，其余标识符名称和常数值不进入特征。因而距离为零不等于源码重复。

初始化集齐 `initial_size`（默认 `5`）个可行个体后，每次将经过评价的个体加入树中，均执行树简化检查（`TopologyConfig`：`max_representatives = 24`，`cover_distance = 0.15`，`region_size = 4`）：

1. **确定保留骨架：** 先选出代表集 $R$，再求代表两两之间的最近公共祖先。记额外的非代表分叉节点为 $F=\operatorname{LCA}_T(R)\setminus(R\cup\{v_0\})$，保留骨架为 $\mathcal H=\{v_0\}\cup R\cup F$。由于 $F$ 在 $R$ 选定之后补齐，其节点不会在贪心覆盖中阻挡候选，也不参与父代抽样。
2. **划分路径与子树区域：** 其余非骨架节点在树拓扑上唯一归入两类互不相交的区域：
   - **路径收缩（`PathContraction`）：** 连接两个保留骨架端点（或虚拟根与骨架端点）的中间链 `removed_ids`，连同挂载在该中间链上的未保留分支 `branch_ids`。
   - **子树剪枝（`SubtreePruning`）：** 其余未保留节点按最近保留骨架祖先（或虚拟根）归集为待剪除子树 `removed_ids`。
3. **成批简化与幂等性：** 仅当路径区域节点总数 `len(removed_ids) + len(branch_ids) >= region_size` 或子树区域节点数 `len(removed_ids) >= region_size` 时，执行拓扑简化并发起一次反思调用。未满阈值的区域暂留在树上，其节点位于 $R$ 之外，不参与父代抽样。简化保留 $R$ 和存活节点之间的祖先关系；在没有新增节点时，再次执行不会继续移除节点，证明见论文附录C。

简化重连边标记 `contracted = True`（`operator` 与 `reference_id` 置空），所有存活节点的 `direct_evaluations` 原样保留。

初始化完成后，每次更新保留的算法节点数至多为 $4Km-2K-m$，其中 $m=\text{region\_size}$；默认参数下为332。其依据是骨架至多含 $2K$ 个节点（含虚拟根），路径与子树区域共至多 $4K-1$ 个，且更新后每个剩余区域至多含 $m-1$ 个节点。该上界与评价预算无关，约束树中的算法节点数；归档不在这一计数内。证明见论文附录C。

## 3. 节点经验与边经验合成

每个达到阈值的简化区域调用一次事实反思（严格 JSON 输出单一 `"experience"` 字段，约 120 英文词）。路径收缩与子树剪枝共用一个模板（`build_reflection_prompt`），仅 `[Algorithms]` 开头的情境句不同。

提示中，区域内每个将被移除的算法给出一条记录：来源与结果 → `design` → 失败原因 → 已有经验 → 源码。每个算法的源码只出现一次，不给代码差分：生成是整段重写，差分比源码更长，也没有可读的局部改动。若被移除算法的直接父代位于区域外且仍在树中，则将该保留父代作为基准列出分数、`design` 与源码。

- **路径收缩区域**：说明终点相对起点（或从零设计起点）的路径与成绩变化；中间链与沿途分支逐条给出；终点 $v$ 给出记录及其入边经验，不给它的节点经验（该经验仍留在 $v$ 上）。生成的**边经验**写入重连边的 `edge.experience`。
- **子树剪枝区域**：依附于保留节点 $p$ 时给出 $p$ 的基准与已有节点经验，区域内节点逐条给出，生成的**节点经验**写入 `node.experience`；依附于虚拟根时给出已有 `root_experience`，从零设计及其衍生尝试按适应度从好到差排列（`listed from best to worst fitness (lower fitness is better)`），并在根下算法记录上标注其相对当前最优保留算法分数的差距，生成的经验写入 `tree.root_experience`。

提示中出现的已有经验都将被新经验取代，反思任务要求记录事实观察并保留仍然重要的早期观察：说明每个有信息量的尝试实际计算了什么（若为修改则相对起始计算改了什么）及其测得的适应度或失败，只写具体计算与结果，不写规则、建议或推测原因，也不引用算法编号（被移除的编号不会再出现）。同一次更新中的全部反思读取同一棵更新前的树。

反思调用失败或返回空经验时，新经验取提示中已有经验的去重拼接，淘汰照常进行：已有经验不丢，节点数上界不受影响，本轮区域的新证据不进入经验。自然语言经验是有损总结。

## 4. 预算分配与算子选择

父代仅在代表集 $R$ 上抽样：

$$w_r = \frac{2^{-\operatorname{rank}_R(r) / H}}{1 + n_r}, \qquad p_r = \frac{w_r}{\sum_{u \in R} w_u},$$

其中 $\operatorname{rank}_R(r)$ 为代表 $r$ 在 $R$ 中相对最优并列组的平均适应度名次差，$H = \text{rank\_half\_life}$（默认 `10.0`）为适应度排名权重减半的名次跨度。$n_r$ 即 `tree.nodes[r].direct_evaluations`，统计以 $r$ 为主父代执行 `Refine` 或 `Crossover` 时消耗的实际评价次数（包括执行失败和修复后的再次评价；`Explore` 挂接于虚拟根下开启新谱系，不计入 $n_r$）：

- **未入选子代：** 与排序更早的已选代表结构距离小于 $\delta$ 的候选不进入 $R$；排序同时考虑适应度与节点编号。生成该子代消耗的评价次数已计入其主父代的 $n_r$（`Refine`/`Crossover`）。
- **新入选子代：** 新节点以 $n=0$ 进入代表竞争，不继承祖先节点的评价计数。适应度更优且距离小于 $\delta$ 的子代可以覆盖旧代表；与已选代表距离均不小于 $\delta$ 的子代在名额允许时加入 $R$。`Refine` 与 `Crossover` 保留原有父子关系，`Explore` 作为虚拟根下的新谱系起点进入竞争。
- **代表资格：** $n_r$ 只影响抽样权重，不决定是否淘汰。代表仅在被排序更早的代表覆盖，或代表名额已满时退出 $R$；退出后若不属于分叉骨架，则作为非骨架节点等待成批简化。

每步先按 `0.45 / 0.30 / 0.25` 抽取 `Refine`、`Explore` 或 `Crossover`。执行 `Explore` 时，`parent_id = 0`（不抽取假父代）；在代表集 $R$ 上先取适应度最好者，再逐个加入与已选个体的最小 AST 距离最大者，至多 `explore_references`（默认 `6`）个。执行 `Refine` 或 `Crossover` 时，先按上式在 $R$ 上抽样主父代；若为 `Crossover`，再在 $R \setminus \{\text{parent}\}$ 上复用同一排名衰减并乘以结构距离 $2^{-\operatorname{rank}/H} \cdot \operatorname{code\_distance}(\text{parent}, r)$ 软抽样参考代表（单代表无候选时回退为 `Refine`；`cover_distance = 0` 下若距离全为零则退回纯排名衰减）。代表两两之间的结构距离不小于 $\delta$，精确重复已由 `distinct_individuals` 折叠，所以候选中不会出现主父代的克隆。

## 5. 面向算子的树上下文读取

上下文使用无内部编号的设计语言（`First version` / `Step k`、`Attempt k`、`Approach k`），由相关经验、被修改算法的形成轨迹与提供想法的算法卡片组成。源码只给被重写或被移植的算法；其余算法按算子角色呈现，不给代码差分。所有段落从全局到局部、从过去到现在排列，需要修改的主父代源码放在最后。

| 算子 | 树上下文内容 |
| --- | --- |
| `Refine` | 从根到主父代的演化路径（`[Development of the Current Algorithm]`），每步给出修改方式、成绩变化、`design`，收缩边给出边经验 `edge.experience`，沿途祖先给出节点经验 `node.experience`；从主父代出发的尝试（`[Attempts from the Current Algorithm]`），包括主父代的节点经验和当前树中各直接子代的结果（失败时给失败原因）、`design` 与经验；最后给主父代分数与源码。 |
| `Crossover` | 与 `Refine` 相同的演化路径与尝试（参考代表若为主父代的直接子代，不在尝试中重复）；参考代表卡片（`[Other Algorithm]`），给出分数、`design`、其从根到自身路径上前文未出现的经验，以及源码；最后给主父代分数与源码。 |
| `Explore` | 虚拟根经验（`[Experience from Earlier Approaches]`，仅在 `Explore` 中展示）；分配阶段从代表集选出的已有做法卡片（`[Existing Approaches]`），按适应度从好到差排序，每张卡片仅给出分数与 `design`（不混入各谱系内部的局部演化经验，也不给源码）。函数接口由任务模板给出。 |

上下文不设长度裁剪，实际输入词元数记录在调用日志的 `usage.prompt_tokens` 中。树的节点数有上界（第2节），但 `design`、经验与源码的长度只受软约束，因此该上界不直接给出词元上界；提示超过服务端上限时，服务端拒绝请求，记为调用错误。将 `EvoTrace(..., include_experiences=False)` 设为 `False` 可在固定树状态下隐藏提示中的节点与边经验，不直接修改该状态的父代抽样概率。

`Crossover` 提示允许在无法形成有用组合时，根据比较结果改进主父代。`Explore` 提示要求构思不同的核心计算过程；实现没有独立检验是否达到这一要求。

## 6. 运行、恢复与边界

统一实验入口（`--method evotrace.v3` / `evotrace_v3`）与独立 CLI（`python -m evotrace_v3`）均支持 `--max-representatives`、`--cover-distance`、`--region-size` 与 `--rank-half-life`（将 `cover_distance` 设为 `0.0` 即退化为无结构覆盖的前 $K$ 个体对照）。`attempts.jsonl` 是唯一权威恢复日志，发生简化的步骤保存更新后的完整树快照，恢复时确定性重放而不重复调用反思；`run_config.json` 严格校验机制列表、选择规则（`representative_rank_direct_sampling`）及拓扑、分配与 `include_experiences` 配置，防止不同版本混跑。

已知边界：

- **AST 结构距离与覆盖非传递性：** `code_distance` 以语法结构二元组近似算法差异，结构相近但需连续退步改写才能突破的变体会被当前更优代表覆盖；覆盖关系不具传递性，且 `distinct_individuals` 仅在当前保留树内折叠精确重复（已被简化移除的程序若在后续搜索中重现，按新节点入树，不维护全局永久黑名单）。
- **反思归纳边界：** 单步自然语言反思是有损归纳，面对多处同时修改的代码仍可能出现归因偏差。当前对照批次与已验证边界见[实验与边界台账](记录/README.md)。
