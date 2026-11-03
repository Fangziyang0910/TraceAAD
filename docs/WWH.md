# EvoTrace：研究问题与思路

研究目标是在有限预算下找到并持续精炼更好的算法，让设计过程形成的经验帮助后续算法发现。

**个体适应度评价与演化过程学习是两个不同层次。** Fitness反映算法当前已经实现的性能；演化轨迹还提供理解算法如何改进、哪些方向值得继续、哪些策略可以迁移的信息。算法演化既产生候选算法，也产生关于如何继续搜索的知识。

当前科学问题是：**如何将算法演化过程转化为后续发现的学习信号，使轨迹引导的算法改进、跨代经验学习与节点选择相互促进？**

EvoTrace围绕同一演化过程组织三个相互关联的机制：

1. **轨迹引导的算法改进。** 以形成路径和同起点尝试构成局部演化上下文，帮助模型理解当前策略及观察到的限制，再结合全局经验提出改进。
2. **轨迹经验学习与累积。** 联合反思最新结果、跨代演化与相关替代，从中归纳可迁移的算法改进策略，形成所有候选共享的正负经验。
3. **轨迹节点价值评估与预算分配。** 结合当前适应度与节点周围的演化响应，评价它作为后续演化起点的价值。形成结果、同父代比较和后续子代产出提供补充信息。

三者形成的研究闭环是：节点周围的轨迹支持价值判断与预算选择，所选算法结合轨迹上下文和累积经验产生候选，评价形成新轨迹，再更新经验与节点价值。经验通过生成结果改变轨迹，进而影响后续选择；三个机制通过这一演化过程相互联系。

当前[V19.4设计](机制设计/V19.4.md)将第三环具体化为候选规则\(V_t(p)=q_t(p)(1+e_t(p))\)：当前质量权重结合近期局部演化响应。节点直接评价形成自身反馈，一跳父子邻域以固定强度提供参考。轨迹未知时回到排名选择，响应修正限于两倍以内。该规则已有独立实现，其算法收益待在线检验。

轨迹在这里既提供理解算法演化的信息，也通过模型上下文影响后续生成。局部轨迹保留当前算法的发展背景，全局经验帮助不同实现借鉴策略；新生成与评价再更新节点价值。这一信息联系不要求每个环节使用完全相同的表示。相关学科启示与概念区分见[深入分析](记录/2026-11-01-轨迹的信息价值与三环机制.md)。

[独立评审与协调者核验](../experiments_result/analysis/v19_4_review_20261101/README.md)针对此前KNN多步片段估计器，其冻结稿与证据继续留存。当前节点响应方案是重新形成的设计，不以旧原型或旧评审作为有效性证据。

[V19.3](机制设计/V19.3.md)已实现轨迹上下文、独立轨迹反思与全局正负经验池，当前批次按原有排名和经验效用规则解释。V19.4的完整三环已有[独立实现](../traceaad_v19_4/README.md)和[实验入口](../experiments/traceaad_v19_4/README.md)，正式主实验已在server4启动六任务×三个种子，每路1000次实际评价，见[启动记录](../experiments_result/traceaad_v19_4/preparation_20261102f/README.md)。

相关方法已建立反思与经验复用机制：[ReEvo](https://arxiv.org/html/2402.01145)通过个体适应度比较形成短期反思，并累积长期反思；[RefineEvo](https://arxiv.org/html/2607.11358v1)从单次演化反馈提取带适用条件的正负经验，并维护使用效用。V19.3关注以演化轨迹为反思单元，学习并迁移算法改进策略。

经验学习机制的核心假设是，以演化轨迹为反思单元，比逐次累积局部反思更有利于学习改进策略。对应检验是在相同演化经历下，这些经验能否更有效地改进未见算法。预算机制比较当前质量、质量加直接响应、质量加邻域响应，区分额外反馈与局部共享是否有用。

当前V19.4设计见上文；下文保留早期问题与方案供追溯。

## 现有 LLM 自动算法设计的不足（Limitations）

**中文草稿**

**搜索局限。** 现有基于 LLM 的自动算法设计方法容易围绕少数当前表现较好的算法展开搜索，反复生成相同或高度相似的候选算法。新的算法思路难以得到充分探索，限制了发现更优算法的能力。

**成因分析。** 与按预设规则扰动个体的传统变异算子相比，LLM 的变异过程还受到预训练知识与提示上下文的影响，倾向于复用熟悉的算法设计。适应度驱动的选择优先保留当前表现较好的算法，使表现落后的新算法难以继续演化。高适应度算法既获得更多演化机会，又被反复用作上下文示例，进一步强化 LLM 对这些算法的偏好。生成偏好与选择压力由此相互强化，使搜索更趋保守并集中于少数算法。

**研究启示。** 上述分析提示，有效探索需要支持不同算法思路持续演化，使其潜力在这一过程中逐步显现。新思路即使获得更多演化机会，后续生成仍可能回到熟悉或当前表现较好的算法。因此，选择应为尚未充分探索的思路保留演化机会，生成则应利用探索每种思路时积累的经验引导其后续演化，从而支持不同方向的持续探索，减少向少数高适应度算法的集中。

**English draft**

**Search limitations.** Existing methods for LLM-based automatic algorithm design tend to focus search on a few algorithms that currently perform well, repeatedly generating identical or highly similar candidates. New algorithmic ideas often remain underexplored, limiting the discovery of better algorithms.

**Underlying causes.** Conventional mutation operators typically perturb individuals according to predefined rules. LLM-based mutation is also shaped by knowledge acquired during pretraining and by prompt context, and tends to reuse familiar algorithmic designs. Fitness-based selection favors algorithms that currently perform well, leaving new algorithms with lower fitness fewer opportunities to evolve. High-fitness algorithms both receive more opportunities to evolve and repeatedly appear as examples in the prompt, further reinforcing the LLM's preference for them. Generation bias and selection pressure thus reinforce each other, making search more conservative and concentrated on a few algorithms.

**Research implications.** This analysis suggests that effective exploration requires sustained evolution of different algorithmic ideas, allowing their potential to emerge through this process. Even when new ideas receive more opportunities to evolve, subsequent generation may still return to familiar algorithms or those that currently perform well. Selection should therefore preserve opportunities for underexplored ideas to evolve, while generation should draw on the experience accumulated when exploring each idea to guide its further evolution. Together, these principles support sustained exploration in different directions and reduce concentration on a few high-fitness algorithms.

## 对 LLM-AAD 搜索过程的理解（Key Insight）

**中文草稿**

**过程理解。** 我们将 LLM-AAD 理解为在有限预算下，通过算法个体的生成、选择与精炼，持续探索和发展不同算法思想的过程。算法思想指算法的主要决策规则；只在参数、权重或附加项上不同的算法属于同一思想。选择决定哪些思想及其中哪些个体获得进一步精炼的机会，生成则决定这些机会产生怎样的后代。我们将一个初始算法及其演化后代归为一个谱系（lineage），用于追踪算法之间的演化关系。谱系不等同于思想：不同谱系可能实现同一思想，一个谱系也可能转向新的思想。

**理解依据。** 当前适应度同时受到算法思想质量与个体精炼程度的影响，直接据此选择，容易让已经充分精炼的个体淘汰思想更好但尚未充分精炼的个体。即使这些个体获得精炼机会，LLM 仍可能沿用上下文中的高适应度算法，使原本不同的父代产生相同或高度相似的后代。LLM 单次生成也倾向于给出它最熟悉的设计，不同的初始算法常常只是同一思想的参数变体。因此，按谱系分配机会不能保证不同思想获得机会。

**设计要求。** 选择应以算法思想为单位分配后续机会，结合各思想的精炼程度与演化进展，避免仅凭当前适应度过早淘汰尚未充分精炼的思想。生成应利用父代及其演化历史中积累的经验推进后续精炼，在借鉴其他算法的同时继续发展不同的算法思想，减少后代向少数高适应度算法趋同。

**English draft**

**Process perspective.** We view LLM-AAD as a process of exploring and developing different algorithmic ideas by generating, selecting, and refining candidate algorithms under a limited budget. An algorithmic idea is the main decision rule of an algorithm; algorithms that differ only in parameter values, weights, or added terms share one idea. Selection determines which ideas, and which individuals within them, receive further refinement, while generation determines the resulting offspring. We group an initial algorithm and its evolutionary descendants into a lineage to track evolutionary relationships among algorithms. A lineage is not an idea: different lineages can implement the same idea, and one lineage can move to a new idea.

**Rationale.** Current fitness reflects both the quality of an individual's algorithmic idea and the extent of its refinement. Selection based directly on current fitness can therefore discard a less refined individual with a better idea in favor of a more thoroughly refined one. Even when less refined individuals receive opportunities to evolve, the LLM may reuse high-fitness algorithms from the prompt context, causing distinct parents to produce identical or highly similar offspring. In a single generation, the LLM also tends to produce the design most familiar to it, so different initial algorithms are often parameter variants of one idea. Allocating opportunities by lineage therefore does not ensure that different ideas receive them.

**Design requirements.** Selection should allocate further refinement opportunities by algorithmic idea, based on each idea's refinement level and evolutionary progress, avoiding premature elimination of less refined ideas solely on the basis of current fitness. Generation should draw on the parent algorithm and the experience accumulated throughout its evolutionary history to guide further refinement. It should continue to develop different algorithmic ideas while drawing on other algorithms, reducing convergence of offspring toward a few high-fitness algorithms.

## V18 的操作性认识修正（2026-10-29）

上述按思想分配是研究目标；当前V18以具体源码起点的开发搜索为统计单位，不识别潜在思想。重复调用生成器未必形成新的探索，返回父代的价值也取决于能否产生不同候选。已观察到的失败是：复制产生了新的零龄任务，同龄持平使其持续存活，而近似相同的生成条件又继续产生复制。

当前修订先保证已有有效源码不能再次申请独立起点保护；同一生成操作与父代源码连续三次没有有效新候选时，关闭该操作并转向其他接续或不提供父代的初始化。计数跨状态与恢复保留，失败调用仍计费；取得首个有效起点之前保留原16次初始化额度。有效但暂时较差的新候选仍可继续开发，同龄持平规则不变。

这些规则用于阻断重复建任务与持续无效重试。它们不保证生成器始终保留原算法思想，也不把源码不同视为算法改进；同龄竞赛与LDS是否提高发现质量，仍由完整预算实验判断。具体规则见[V18机制说明](../traceaad/v18/README.md)，诊断依据见[本周记录](记录/2026-W44.md)。

## V18.1：先确定出生，再比较精炼成熟度（2026-10-29）

本轮实验将问题收回到新思想尚未充分精炼、却直接面对成熟思想的当前成绩。年龄不再是固定程序被选作父代的次数，也不是多个起点的年龄组合；先以核心决策机制首次有效实现定义出生，再沿同一机制的精炼版本继承唯一年龄。

[V18.1](../traceaad/v18_1/README.md)保留基础 TraceAAD 的初始化、提示和常用算子。独立代码审核为思想登记出生标识，历史同龄成绩补充原父代选择；每个思想共享有限保护额度，微小修改和旧机制重现不重新获得额度。此处维护最小出生身份与额度，不使用 V18 的任务树或LDS。出生识别是可审查的操作性判断，精炼深度不等于累计预算；审核的额外模型成本必须单列。首次18路用于观察完整方法效果，不能单凭它隔离出生审核、保护额度和同龄排名各自的贡献。

## 2026-11-01：以可核查的开发过程组织预算

本轮重新检查 V14 以来和12种基线后，我把问题收回到：有限预算如何使一个算法改动成为可执行、可继续开发的计算。当前适应度、生成是否正确落实原理、有限接续后的最好结果，是不同的量。已观察到实现失真、修改/修复删除计算、接续不足，以及小幅训练改善未迁移；这些现象不能统一归因于过早淘汰新思想。

当前质量对下一步产出有预测价值，不能以“保护探索”为由削弱到几乎均匀选择。另一方面，每个新程序都增加新的父代—算子组合，使多数局部统计只有零到一次直接观测，增加总预算也未必形成可靠估计。我们不再给这种稀疏统计追加潜力预测或信用层。

本轮 V19.2 候选用有成本上限的连续开发、每段一个最好代表、以及实际源码与编辑结果组织搜索。段是操作边界，不是思想身份；精确重复不能重新领取份额。工作版本可以在有限额度内暂时退步，最好结果始终保留。源码反馈用于帮助落实和保留计算，不构成自动语义正确性保证。

可检验的核心主张是：接续和源码反馈是否分别改善了发现质量，而不只是增加 Refine 数量或恢复对高质量程序的集中。有限精炼在 V14 已有，增量历史、经验引导和谱系分配也有公开先例；新的贡献必须建立在这个区别的受控证据上。当前尚无新版在线优势结论。完整证据、基线特性、先例与最小对照见[V19.2重构与科研叙事](记录/2026-11-01-V19.2重构与科研叙事.md)，操作定义见[V19.2机制](机制设计/V19.2.md)。
