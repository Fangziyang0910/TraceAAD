# TraceAAD：研究问题与思路

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
