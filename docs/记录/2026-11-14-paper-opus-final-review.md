# EvoTrace论文第二轮审阅（成稿逐段）

时间：2026-11-14（北京时间）。审阅对象：`docs/paper.md`、`docs/paper-中文.md`、`paper/main.tex`、`paper/evotrace.pdf`（16页）、`paper/README.md`、`paper/evidence/snapshot.json`。源码核对：`evotrace_v2/topology.py`、`allocation.py`、`models.py`、`reflection.py`、`context.py`。

## 总体判断

成稿的主线已经成立：开发价值与经历价值的区分 → 反思是压缩 → 树为压缩提供范围和位置 → 一次保留决策同时决定显式候选、压缩区域和承接位置。两条贡献与这条主线对应。机制描述与源码一致，没有发现会改变主张的错误。

数字全部与快照吻合：951次评价（12路之和）、894次尝试、207次简化更新、286个移除节点、47个路径区域、175个旁支区域、222次反思、0次回退、186个单节点区域、最大区域5、输入505,171与输出29,283词元、保留42–62、表1的20个分数与均值和样本标准差。中英文的数字、公式和表格一致。

剩余问题集中在三处：S/V的命名会被误读；第4.3节缺少种群规模的分母，读者无法理解“压缩”在这个阶段实际覆盖多少；第3.5节和第4.4节仍有实现流水和项目说明式语句。以下按“必须改／值得改”列出8项。

已核对、无需改动的边界：

- 唯一归属的表述正确。正文写“each removed node belongs to exactly one region in that update”，附录B末尾写明保留端点代码可出现在多个反思输入中。这是本轮移除节点的归属，不是全部上下文不重复。源码`plan_consolidation`与此一致。
- `LCA(R∪U)`与源码一致。源码取已选节点祖先闭包中有至少两个已选子树的节点，等价于两两最近共同祖先集合。
- 冠军保留、已超越判断、覆盖判断“代表适应度不差于候选”、同分门槛0.20、近期窗口只在同路线且非根时取父代时间，均与源码一致。
- “raw scores remain in the archive and the program attaches endpoint scores to path experience”正确。源码在路径经验前附加`[父代分数 -> 子代分数 (outcome)]`。
- “Generated design rationales are excluded from the evidence”正确。`design`字段只在回退摘要中出现，本快照回退次数为0。

---

## 必须改

### 1. 第3.2节：S/V的名称没有结果方向，也不参与父代选择

**问题。** “Selection experience”容易被读成“供父代选择使用的经验”或“被淘汰（失败）尝试的经验”；“Variation experience”容易被读成“成功改进的经验”。两者都不对。源码中，S只进入生成上下文和后续反思，不进入式（4）–（5）。旁支区域可以包含比承接节点更好的尝试：当承接节点因近期或分叉而保留、而更好的子代既非代表也非近期时，这个子代就会进入S。路径区域可以是净退步，程序会在V前附加`regressed`。正文目前没有错误陈述，但缺少一句定义来防止误读。

**位置。** 英文§3.2第二段，“**Variation experience** $V_{uv}$ ... between retained endpoints.”之后。中文同段同步。

**替换（插入一句）：**

> The names refer to the operation that produced the summary: $S_v$ condenses alternatives that the update removed around $v$, and $V_{uv}$ condenses the variation steps along a contracted path. Neither carries an outcome direction. A side branch may contain a trial better than its receiver, and a contracted path may record a net regression. Selection experience informs generation and later reflection; it does not enter parent sampling.

中文：

> 两个名称指产生摘要的操作：$S_v$凝练本次更新在$v$周围移除的替代尝试，$V_{uv}$凝练收缩路径上的变异步骤。二者都不表示结果方向：旁支可以包含优于承接节点的尝试，收缩路径也可以记录净退步。选择经验进入生成上下文和后续反思，不进入父代采样。

### 2. 第4.3节：补充种群规模的分母，说明此阶段压缩作用于哪部分历史

**问题。** “the retained populations contain 42–62 algorithms”没有分母。快照中12路共接入889个节点（603个保留＋286个移除），约三分之二仍是显式节点，各路保留比例为0.58–0.80。原因是机制本身：近期窗口为40次评价，而前缀只有62–97次评价，窗口覆盖了大半前缀。这同时解释了另外两个数：旁支区域占多数（175/222），单节点区域占多数（186/222）。读者需要这条因果链才能正确理解“压缩”在当前快照里的规模。这是解释现象，不是防御；省略它反而会让读者从图3下排自己推出“种群几乎不收缩”。

**位置。** 英文§4.3第一段末句“At the observed endpoints, the retained populations contain 42–62 algorithms.”。中文同段。

**替换：**

> At the observed endpoints, the retained populations contain 42–62 algorithms; across runs, 603 of the 889 attached nodes remain explicit. With a 40-evaluation recency window and prefixes of 62–97 evaluations, recent development still covers most of each run. Compression at this stage therefore acts mainly on side attempts that have left the window, which matches the predominance of side regions (175 of 222) and of single-node regions.

中文：

> 快照时各路保留42–62个算法；12路共接入889个节点，其中603个仍为显式节点。近期窗口为40次评价，而各路前缀只有62–97次评价，近期开发仍覆盖大半前缀。因此，此阶段的压缩主要作用于已离开窗口的旁支尝试，这与旁支区域占多数（222个中175个）、单节点区域占多数一致。

（第二段“Of these regions, 186 contain one newly removed implementation”保留，承接上面的“single-node regions”。）

### 3. 第3.5节：精确化“路线最大值”的作用，删去实现流水

**问题。** （a）“A route competes through its maximum node weight, removing direct accumulation of weights from historical versions”方向正确，但读者会把它理解为“路线概率不随版本数变化”。实际情况是：新增一个权重低于路线最大值的版本，不会直接提高该路线概率；但新版本可以成为新的最大值，父代的$c(v)$加一会降低它自己的权重，代表集合变化也会改变$m$和$s$，从而改变所有路线的概率。（b）本段混入了多句计数规则（失败且后被移除计一次、修复/Explore/只读参考/重连/继承历史不计），以及与式后定义重复的“This uses the same quality and difference selection...”。这些是实现规格，应放在附录A。

**位置。** 英文§3.5第一段，从“where $m$ is median fitness ...”到“Internal and leaf algorithms share this rule.”。中文同段。

**替换：**

> where $\sigma$ is the sigmoid function, $c(v)$ counts evaluated Refine and Crossover attempts that used $v$ as primary parent, and $m$ and $s$ are the median and the median-absolute-deviation scale of fitness over the quality representatives, recomputed on the current tree before each sampling step. The probability of $v$ is $P(D(v))\,P(v\mid D(v))$; internal and leaf algorithms share this rule. Because a route competes through its largest weight, adding a version that does not exceed that weight does not by itself raise the route's probability. Route probabilities still change when a new version becomes the route maximum, when development discounts its parent, or when the representative set shifts $m$ and $s$.

附录A“Parent sampling”一行改为：

> Sigmoid coefficient 10; discount $1/(1+c(v))$, where $c(v)$ counts evaluated Refine/Crossover attempts with $v$ as primary parent, including failed attempts and attempts later removed, and excluding repairs, Explore, read-only reference use, and reconnection; route maximum and within-route normalization

### 4. 第3.4节：把提示要求与机制保证分开

**问题。** “Earlier experience is retained insofar as it remains consistent with the measured comparisons.”写成了机制性质，但它是提示对模型的要求。程序保证的是：比较标签由程序计算，路径经验的端点分数由程序附加，同一更新各区域读同一快照。模型是否遵守保留规则由反思质量决定，这正是附录E“Summary fidelity”要检验的内容。把两者分开，读者会更清楚哪些性质由结构保证、哪些由模型完成；这比后文另加限定句更直接。

**位置。** 英文§3.4第二段第三句。中文“旧经验只有与测量比较仍一致时才予以保留。”

**替换：**

> The prompt asks the model to keep earlier experience only where it remains consistent with the measured comparisons; the comparison labels themselves are computed by the program.

中文：

> 提示要求模型只保留与当前测量比较一致的旧经验；比较标签本身由程序计算。

---

## 值得改

### 5. 引言第3段：把缺口写成可检查的结构性事实

**问题。** 本段最后一句“They also motivate a joint maintenance operation: when the candidate structure changes, the representation of its accumulated experience should change with it.”是愿望式陈述，没有说清现有做法缺什么。缺口应留在本文叙事轴上：候选维护与经验维护由不同事件驱动，因此候选离开时，没有任何结构决定其历史中哪部分被凝练、凝练结果放在哪里。

**位置。** 英文§1第3段末句。中文对应句“也引出了一个联合维护问题……”。

**替换：**

> In these designs, candidate maintenance and experience maintenance are typically driven by different events: a population update decides which programs survive, while reflection or memory is triggered by generations, comparisons, or branch termination. When a program leaves the population, the update itself does not determine which part of its history should be condensed or where the condensed account should remain.

中文：

> 在这些设计中，候选维护与经验维护通常由不同事件驱动：种群更新决定哪些程序留下，反思或记忆则由代数、比较或分支结束触发。程序离开种群时，更新本身并不决定其历史中哪部分需要凝练，也不决定凝练结果留在何处。

### 6. 引言第5–6段：合并重复，让第6段只承担“为什么是一次更新”

**问题。** 第5段末句“Subsequent generation draws both its starting program and its relevant history from the maintained tree.”与第6段末句“These supporting operations make the compressed history available at the next design decision.”说的是同一件事。第6段首句“makes candidate selection, structural simplification, and experience accumulation parts of one update”是本文最重要的设计后果，却被夹在采样和算子视图之间。

**位置。** 英文§1第6段全文。中文同段。

**替换（第6段）：**

> Because one retention decision fixes the surviving candidates, the regions to compress, and their receivers, candidate selection and experience accumulation cannot drift apart: every program that leaves the population hands its history to a definite node or edge. Parent sampling then allocates evaluations over the surviving candidates, and operator-specific views give refinement its local development, crossover two development paths, and exploration route-level ideas.

中文：

> 一次保留决策同时确定留下的候选、需要压缩的区域及其承接位置，因此候选选择与经验积累不会分离：每个离开种群的程序都把其经历交给确定的节点或边。随后，父代采样在留下的候选之间分配评价，算子视图为改进提供局部发展、为交叉提供两条发展路径、为探索提供路线思想。

### 7. 删减项目说明式与重复语句（不隐藏必要信息）

以下语句的信息已在别处给出，或属于项目内部说明。删除后，正文仍保留：早期快照性质（§4首段）、未完成测试与对照（§4首段、附录E）、旧档案条件不同（README）。

| 位置 | 现文 | 处理 |
| --- | --- | --- |
| §4.4 全节 | 第一段“Existing experiment archives include earlier EvoTrace variants ... stated explicitly.”；第二段受控对照设计 | 第一段是项目档案说明，不属于论文。第二段与附录E重复。整节替换为下面两句，并入§4.3末尾，删去§4.4标题。 |
| §4.3 第三段后两句 | “The recorded updates establish the structural transfer ... The utility of that experience for future generation is a separate experimental outcome.” | 由下面的替换句承担。 |
| 表1说明末句 | “These early values describe the deployed method at a common checkpoint, before completion of its intended budget.” | 删除。§4首段已说明。 |
| §3.3 第二段末句 | “These checks define the implemented coverage heuristic.” | 删除。空句。 |
| §3.4 首段 | “**Compression as a population operation.** ...” | 删除。第三段末句已说明树存摘要、档案存完整证据。中文稿本来就没有这一段，删除后中英文一致。 |
| §5 Scope 末句 | “Population size follows the retention rules and branching structure, while the prompt budget controls how much of the maintained state is read at a time.” | 删除。§3.3已说明无固定容量，§3.5已说明输入预算。 |

§4.3末尾替换句（取代§4.4和§4.3第三段后两句）：

> These records establish the structural handoff from explicit implementations to node and edge experience. Whether that experience improves subsequent generation is tested by the fixed-state comparison in Appendix E, which holds the tree snapshot, parent, operator, and reference fixed and changes only how the removed regions are represented.

中文：

> 这些记录确认了显式实现向节点和边经验的结构交接。这些经验能否改善后续生成，由附录E的固定状态对照检验：固定树快照、父代、算子和参考，只改变移除区域的表示方式。

### 8. 排版与中英差异

1. **引用括号重复。** PDF第6页显示“ShinkaEvolve ([Lange et al., 2026])”。`docs/paper.md`第112行`([@shinka])`改为`[@shinka]`，中文同步。
2. **中文§3.6末段不准确。** “该过程保留当前冠军、主父代连通关系与移除区域的唯一归属”中的“主父代连通关系”不对：重连对象是最近保留祖先，不是主父代。改为与英文一致：

   > 每次更新保留当前最佳算法，并将每个保留节点重连到其最近保留祖先。两两分叉闭包使各收缩路径互不重叠，其余移除节点各有唯一承接区域。附录B给出这些结构性质的论证。

3. **英文§3.6末段可同时收紧。** “Pairwise fork closure ensures that contracted intermediate paths are disjoint; the remaining removed nodes have a unique receiving region.”保留；不需要改。
4. PDF排版无溢出、无未定义引用；16页。附录A表格在第10–11页跨页，可接受。

---

## 可采用的摘要

保持机制主张，不含工程计数（约180词）：

> Automated algorithm design with large language models evaluates many candidate programs, and each evaluation serves two purposes: the program can seed further search, and its comparison with its parent can inform later revisions. These purposes diverge, because a program can stop being worth developing while its history stays informative. We present EvoTrace, which treats each population update as a decision about how search history is represented. EvoTrace maintains a tree-structured population whose nodes are executable programs and whose nodes and edges carry reflected experience. One retention decision keeps the selected programs and the forks that connect them explicit. The same tree then assigns each removed program to one region: a contracted path, summarized on the reconnected edge, or a side branch, summarized at its retained ancestor. When a later update removes a carrier, its experience enters the new summary, so the account of development persists as the explicit population changes. Refinement, crossover, and exploration read operator-specific views of this tree. Selection thus decides which programs remain available, and reflection decides how the history of removed programs reaches subsequent generation.

与现稿相比：删去“A program can lose the first value while its history retains the second”与首句的重复；把“jointly determines … receive their experience”和“Contracted paths … retained ancestors”两句合成“一次保留决策 → 每个移除程序归入一个区域”的因果顺序；末句把“evolving representation of both available programs and their design history”换成选择与反思各自承担的职责。

若完整实验结果到位，在末句前增加一句结果陈述；在此之前不加。

---

本次只写本文件。没有修改共享主稿、源码或实验，没有启动实验，没有委派。论文写作已成全文，但实验未完成，当前稿不是可直接投稿的终稿。
