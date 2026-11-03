# EvoTrace: Learning to Design Algorithms from Evolutionary Trajectories

## Introduction

**中文**

自动算法设计（Automatic Algorithm Design, AAD）旨在为一类问题自动发现高性能算法，减少对人工设计与专家经验的依赖 [11–13]。人类专家在反复修改与评价中完善算法，同时积累哪些改法有效、在何种条件下有效的认识。大语言模型（Large Language Models, LLMs）能够直接生成和修改程序，为程序演化提供了新的设计手段 [14, 15]。将这一能力与演化搜索结合，已有研究在组合优化 [1, 2, 17–19] 与数学发现 [3, 4] 等领域取得了进展，系统评测也表明搜索策略会影响算法发现的效果 [16]。LLM 驱动的搜索同样产生了丰富的改进经历，已有工作开始分析其中的代码结构变化 [30]。如何从这些经历中学习设计知识，是进一步提高算法发现能力的重要问题。

现有研究从搜索控制和知识复用两方面利用改进经历。候选表现、访问次数及后代评价被用于分配探索机会 [1, 5, 6, 20–22]，计算预算也可在多条演化运行之间动态分配 [23]。性能比较用于归纳修改建议和可复用经验 [2, 9]，评价反馈还可用于演化生成提示 [26, 27] 或训练生成模型 [31]。祖先程序与语义修改记录为生成提供历史上下文 [6, 7]，设计原则记忆和分支内外的经验共享进一步支持知识复用 [25, 28]；代码结构与性能的关联也被转化为修改指导 [29]。规划方法将反思建议或推导关系纳入搜索决策 [8, 24]。PhyloEvolve 联合利用祖先轨迹、邻近分支比较与精英轨迹提出修改，并从轨迹中总结可复用知识 [10]。这些进展体现了设计尝试的两种价值：产生候选算法，以及积累指导后续设计的知识。

现有方法针对不同决策，保留了改进经历的不同部分。基于统计的选择将候选质量、探索程度、后代表现或近期改进概括为数值依据 [5, 6, 10, 20–22]，具体改法与评价反馈之间的联系并未直接体现在这些量中。局部比较式反思则以一次父子或个体差异为主要证据 [2, 9]。这些数值与局部比较能够描述搜索状态和局部结果，却未充分呈现当前候选怎样形成、从它出发已经尝试过哪些改法，以及相应反馈之间的联系。

以候选为中心的演化轨迹（evolutionary trajectory）为利用这些联系提供了具体方式。祖先路径关联形成当前设计的连续修改及其反馈，已有子代尝试则记录此前从该候选出发试过的改法及其结果。前者提供设计背景，后者提供在同一起点上比较不同改法的参照。两类信息共同支持继续投入的判断、当前算法的改进，以及带适用条件的经验归纳。

本文提出 EvoTrace，一种从演化轨迹中学习设计知识的自动算法设计框架。我们将有限预算下的算法发现建模为依赖演化历史的序贯决策问题 [8]，以最大化所获最佳算法的期望性能为目标。框架包含三个机制。轨迹引导的预算分配结合当前性能，依据候选经历的具体修改及其反馈评估继续搜索的价值。轨迹引导的算法设计以祖先路径和已有子代尝试为上下文，结合检索经验指导新的修改。轨迹引导的经验凝练在连续修改与同一起点的尝试对照中分析局部结果，提炼带适用条件的正负经验 [9]。搜索不断产生新的改进经历，轨迹中积累的知识又指导后续搜索，构成算法发现与经验学习的闭环。

本文的主要贡献如下：

- 我们提出 EvoTrace，一种从演化轨迹中学习设计知识的自动算法设计框架。该框架将预算约束下的算法发现建模为依赖演化历史的序贯决策问题，使改进经历中积累的知识持续指导搜索决策与算法改进。
- 我们设计轨迹引导的预算分配、算法设计与经验凝练机制，将候选的祖先路径和已有子代尝试作为共同依据，分别用于评估进一步搜索的价值、指导当前算法的修改，以及提炼带适用条件的正负经验。
- 【结果待确认：我们在覆盖{问题类别与规模}的基准上系统评估 EvoTrace。在相同{预算口径}下，EvoTrace 相较{代表性方法}在{核心指标}上达到当前最优水平，并在{优势场景}中取得{具体收益}。】

**English**

Automatic algorithm design (AAD) aims to discover high-performing algorithms for a class of problems while reducing reliance on manual design and expert knowledge [11–13]. Human experts refine algorithms through repeated modification and evaluation, accumulating knowledge of which changes work and under what conditions. Large language models (LLMs) can generate and modify programs directly, offering new operators for program evolution [14, 15]. Combining this capability with evolutionary search has led to advances in combinatorial optimization [1, 2, 17–19] and mathematical discovery [3, 4]. Systematic evaluations further show that search strategies affect algorithm discovery [16]. LLM-driven search similarly produces a rich history of design attempts, and recent work has begun to analyze the resulting changes in code structure [30]. An important challenge is to turn this history into design knowledge that can guide subsequent search.

Existing research uses modification histories for search control and knowledge reuse. Candidate performance, visit counts, and descendant evaluations guide the allocation of search effort [1, 5, 6, 20–22], while computational budgets can also be allocated adaptively across parallel evolutionary runs [23]. Performance comparisons yield modification advice and reusable experience [2, 9], and evaluation feedback can improve generation prompts [26, 27] or train the generating model [31]. Ancestral programs and semantic modification records provide historical context for generation [6, 7]. Memories of design principles and experience sharing within and across branches further support knowledge reuse [25, 28], while associations between code structure and performance provide modification guidance [29]. Planning methods incorporate reflective suggestions or derivation relationships into search decisions [8, 24]. PhyloEvolve combines ancestral trajectories, comparisons with neighboring branches, and elite trajectories to propose modifications, and summarizes trajectories into reusable knowledge [10]. These advances illustrate two sources of value in a design attempt: the candidate it produces and the knowledge it contributes to subsequent design.

Existing approaches retain different aspects of modification histories for different decisions. Selection based on statistics summarizes candidate quality, exploration, descendant performance, or recent improvement numerically [5, 6, 10, 20–22]. These quantities do not directly represent the relationships between concrete modifications and their evaluation feedback. Reflection based on local comparisons primarily draws on a parent–offspring pair or differences between two candidates [2, 9]. These numerical summaries and local comparisons describe search states and local outcomes, but provide limited information about how the current candidate was formed, which modifications have already been attempted from it, and how the resulting feedback relates across attempts.

Organizing evolutionary trajectories around each candidate provides a way to use these relationships. The ancestral path links successive modifications and their feedback leading to the current design, while existing offspring attempts record modifications previously made from that candidate and their outcomes. The former provides design context; the latter supports comparison of different modifications from the same starting algorithm. Together, they inform further investment, improvements to the current algorithm, and the distillation of experience with applicability conditions.

We introduce EvoTrace, a framework that learns from evolutionary trajectories to guide automatic algorithm design. We formulate algorithm discovery under a finite budget as a sequential decision problem conditioned on evolutionary history [8], with the objective of maximizing the expected quality of the best algorithm found. The framework comprises three mechanisms. Trajectory-guided budget allocation uses concrete modifications and their feedback, together with current performance, to assess the value of further search. Trajectory-guided algorithm design uses the ancestral path and existing offspring attempts as context, together with retrieved experience, to guide new modifications. Trajectory-guided experience distillation analyzes local outcomes through comparisons across successive modifications and attempts from the same starting algorithm, deriving positive and negative experiences with applicability conditions [9]. Search continually produces new design attempts, while knowledge accumulated from their trajectories guides subsequent search, forming a feedback loop between algorithm discovery and learning from experience.

Our main contributions are:

- We introduce EvoTrace, a framework for automatic algorithm design that learns from evolutionary trajectories. It formulates budget-constrained algorithm discovery as a sequential decision problem conditioned on evolutionary history, using accumulated design knowledge to guide search decisions and algorithm refinement.
- We develop trajectory-guided budget allocation, algorithm design, and experience distillation. These mechanisms use each candidate's ancestral path and existing offspring attempts as a common basis for assessing the value of further search, guiding modifications to the current algorithm, and deriving positive and negative experiences with applicability conditions, respectively.
- [RESULTS TO BE VERIFIED: We systematically evaluate EvoTrace on benchmarks covering {problem classes and scales}. Under matched {budget measure}, EvoTrace achieves state-of-the-art performance on {primary metric} against {representative methods}, with {specific gains} in {advantageous settings}.]

## References

[1] Liu, F., et al. [Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model](https://proceedings.mlr.press/v235/liu24bs.html). In *Proceedings of the 41st International Conference on Machine Learning (ICML)*, PMLR 235:32201–32223, 2024.

[2] Ye, H., et al. [ReEvo: Large Language Models as Hyper-Heuristics with Reflective Evolution](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4ced59d480e07d290b6f29fc8798f195-Abstract-Conference.html). In *Advances in Neural Information Processing Systems (NeurIPS)*, 37:43571–43608, 2024.

[3] Romera-Paredes, B., et al. [Mathematical discoveries from program search with large language models](https://doi.org/10.1038/s41586-023-06924-6). *Nature*, 625:468–475, 2024.

[4] Novikov, A., et al. [AlphaEvolve: A coding agent for scientific and algorithmic discovery](https://arxiv.org/abs/2506.13131). *arXiv preprint*, arXiv:2506.13131, 2025.

[5] Chen, B., et al. [A₂DEPT: Large Language Model–Driven Automated Algorithm Design via Evolutionary Program Trees](https://proceedings.mlr.press/v306/chen26hb.html). In *Proceedings of the 43rd International Conference on Machine Learning (ICML)*, PMLR 306:18654–18693, 2026.

[6] Zheng, Z., et al. [Monte Carlo Tree Search for Comprehensive Exploration in LLM-Based Automatic Heuristic Design](https://proceedings.mlr.press/v267/zheng25o.html). In *Proceedings of the 42nd International Conference on Machine Learning (ICML)*, PMLR 267:78338–78373, 2025.

[7] Jiang, J., et al. [DeltaEvolve: Accelerating Scientific Discovery through Momentum-Driven Evolution](https://proceedings.mlr.press/v306/jiang26g.html). In *Proceedings of the 43rd International Conference on Machine Learning (ICML)*, PMLR 306:52188–52212, 2026.

[8] Gungordu, O., et al. [PathWise: Planning through World Model for Automated Heuristic Design via Self-Evolving LLMs](https://proceedings.mlr.press/v306/gungordu26a.html). In *Proceedings of the 43rd International Conference on Machine Learning (ICML)*, PMLR 306:37779–37816, 2026.

[9] Wu, Y., et al. [RefineEvo: Planning-Guided Heuristic Evolution with Bidirectional Experience](https://arxiv.org/abs/2607.11358). *arXiv preprint*, arXiv:2607.11358, 2026.

[10] Zhao, L., et al. [Large Language Model-Powered Evolutionary Code Optimization on a Phylogenetic Tree](https://arxiv.org/abs/2601.14523). *arXiv preprint*, arXiv:2601.14523, 2026.

[11] Zhao, Q., et al. [Automated Design of Metaheuristic Algorithms: A Survey](https://openreview.net/forum?id=qhtHsvF5zj). *Transactions on Machine Learning Research*, 2024.

[12] Liu, F., et al. [A Systematic Survey on Large Language Models for Algorithm Design](https://doi.org/10.1145/3787585). *ACM Computing Surveys*, 2026.

[13] Ma, Z., et al. [Toward Automated Algorithm Design: A Survey and Practical Guide to Meta-Black-Box-Optimization](https://doi.org/10.1109/TEVC.2025.3568053). *IEEE Transactions on Evolutionary Computation*, 2025.

[14] Lehman, J., et al. [Evolution through Large Models](https://doi.org/10.1007/978-981-99-3814-8_11). In *Handbook of Evolutionary Machine Learning*, pp. 331–366, 2024.

[15] Hemberg, E., Moskal, S., and O'Reilly, U.-M. [Evolving Code with a Large Language Model](https://doi.org/10.1007/s10710-024-09494-2). *Genetic Programming and Evolvable Machines*, 25, Article 21, 2024.

[16] Zhang, R., et al. [Understanding the Importance of Evolutionary Search in Automated Heuristic Design with Large Language Models](https://doi.org/10.1007/978-3-031-70068-2_12). In *Parallel Problem Solving from Nature — PPSN XVIII*, pp. 185–202, 2024.

[17] Pham Vu Tuan Dat, et al. [HSEvo: Elevating Automatic Heuristic Design with Diversity-Driven Harmony Search and Genetic Algorithm Using LLMs](https://doi.org/10.1609/aaai.v39i25.34898). In *Proceedings of the AAAI Conference on Artificial Intelligence*, 39(25):26931–26938, 2025.

[18] Yao, S., et al. [Multi-objective Evolution of Heuristic Using Large Language Model](https://doi.org/10.1609/aaai.v39i25.34922). In *Proceedings of the AAAI Conference on Artificial Intelligence*, 39(25):27144–27152, 2025.

[19] Gong, H., et al. [Large language models discover complementary heuristics for combinatorial optimization](https://doi.org/10.1038/s42256-026-01307-8). *Nature Machine Intelligence*, 2026.

[20] Lange, R. T., Imajuku, Y., and Cetin, E. [ShinkaEvolve: Towards Open-Ended and Sample-Efficient Program Evolution](https://arxiv.org/abs/2509.19349). *arXiv preprint*, arXiv:2509.19349, 2025.

[21] Chen, Z., et al. [QUBE: Enhancing Automatic Heuristic Design via Quality-Uncertainty Balanced Evolution](https://arxiv.org/abs/2412.20694). *arXiv preprint*, arXiv:2412.20694, 2024.

[22] Lai, K., Lai, Y., and Liu, H.-L. [Beyond the Node: Clade-level Selection for Efficient MCTS in Automatic Heuristic Design](https://arxiv.org/abs/2602.00549). *arXiv preprint*, arXiv:2602.00549, 2026.

[23] Xing, S., et al. [Compute Allocation in Evolutionary Search: From Depth–Breadth to Multi-Armed Bandits](https://arxiv.org/abs/2605.29268). *arXiv preprint*, arXiv:2605.29268, 2026.

[24] Wang, H., Zhang, X., and Mu, C. [Planning of Heuristics: Strategic Planning on Large Language Models with Monte Carlo Tree Search for Automating Heuristic Optimization](https://arxiv.org/abs/2502.11422). *arXiv preprint*, arXiv:2502.11422, 2025.

[25] Chen, C., et al. [HiFo-Prompt: Prompting with Hindsight and Foresight for LLM-based Automatic Heuristic Design](https://arxiv.org/abs/2508.13333). *arXiv preprint*, arXiv:2508.13333, 2025.

[26] Liu, Y., et al. [Experience-Guided Reflective Co-Evolution of Prompts and Heuristics for Automatic Algorithm Design](https://arxiv.org/abs/2509.24509). *arXiv preprint*, arXiv:2509.24509, 2025.

[27] Qiu, Z., et al. [MeLA: A Metacognitive LLM-Driven Architecture for Automatic Heuristic Design](https://doi.org/10.1016/j.eswa.2026.133022). *Expert Systems with Applications*, 330:133022, 2026.

[28] Haji, F., Delarosa Quiros, J., and Najafirad, P. [Memory-Guided Tree Search with Cross-Branch Knowledge Transfer for LLM Solver Synthesis](https://arxiv.org/abs/2605.17539). *arXiv preprint*, arXiv:2605.17539, 2026.

[29] van Stein, N., et al. [LLaMEA-SAGE: Guiding Automated Algorithm Design with Structural Feedback from Explainable AI](https://arxiv.org/abs/2601.21511). *arXiv preprint*, arXiv:2601.21511, 2026.

[30] van Stein, N., et al. [Code Evolution Graphs: Understanding Large Language Model Driven Design of Algorithms](https://arxiv.org/abs/2503.16668). *arXiv preprint*, arXiv:2503.16668, 2025.

[31] Huang, Z., et al. [CALM: Co-evolution of Algorithms and Language Model for Automatic Heuristic Design](https://openreview.net/forum?id=x6bG2Hoqdf). In *The Fourteenth International Conference on Learning Representations (ICLR)*, 2026.
