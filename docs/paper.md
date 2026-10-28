# Introduction

**中文**

近年来，大语言模型（Large Language Models, LLMs）与进化搜索的结合推动了自动算法设计（Automatic Algorithm Design, AAD）的发展 [1, 2]。现有 LLM 驱动的自动算法设计（LLM-AAD）方法通过候选程序生成、执行评价与反馈驱动的迭代改进，在组合优化等任务中展现出自动发现高性能算法的能力 [3–5]。与求解单个问题实例不同，LLM-AAD 旨在发现适用于一类问题实例的求解算法，因而需要在庞大且复杂的程序空间中进行搜索 [3, 5]。如何在有限预算下有效组织这一搜索过程，仍是充分发挥 LLM-AAD 算法发现能力的关键挑战 [6, 7]。

现有 LLM-AAD 研究主要通过改进搜索策略与利用历史经验来提升算法发现能力。在搜索策略方面，研究通过种群管理与树结构搜索改进候选算法的选择和扩展，以更有效地探索算法空间 [1, 5, 6]。与此同时，反思反馈、语义修改记录与轨迹记忆被用于指导后代生成，使搜索能够利用先前尝试中积累的经验 [2, 7, 8]。综合来看，这些研究凸显了两个相互关联的关键决策：哪些算法获得进一步演化的机会，以及如何利用搜索经验推动其后续改进。

尽管取得了上述进展，现有 LLM-AAD 方法仍容易围绕少数高适应度算法展开搜索，使其他算法思想难以得到充分探索 [9, 10]。我们从选择与生成两个环节理解这一局限：基于当前适应度的选择可能过早淘汰尚未充分精炼的算法思想，而 LLM 的代码生成本身就表现出算法多样性收缩的倾向 [11]，可能使后代在原有思想尚未充分开发前转向熟悉或已有的高性能设计。前者限制了新思想的演化机会，后者则使获得的机会未必用于发展原有思想；当高适应度设计反复成为选择对象与上下文示例时，两者可能相互强化，推动搜索向少数方向集中。

借鉴 Go-Explore 记忆并重访有希望状态的探索原则 [12]，我们的关键认识是：有效探索既需要为不同算法思想保留演化机会，也需要支持这些思想在已有进展上持续发展。当前适应度反映的是算法现有实现的表现，未必能够代表进一步精炼后的表现；这种当前表现与持续开发价值的区别，也体现在递增收益老虎机（rising bandits）的研究中 [13]。因此，选择应结合精炼投入与演化进展分配后续机会，生成则应利用相应思想的演化经验引导改进，使演化机会能够转化为对不同思想的持续探索。

**English**

Recent advances in automatic algorithm design (AAD) have been driven by the integration of large language models (LLMs) with evolutionary search [1, 2]. Existing LLM-driven AAD (LLM-AAD) methods generate candidate programs, evaluate them through execution, and iteratively refine them using feedback, demonstrating the ability to automatically discover high-performance algorithms for tasks such as combinatorial optimization [3–5]. Rather than solving a single problem instance, LLM-AAD aims to discover algorithms that solve a class of instances, requiring search over a vast and complex program space [3, 5]. Effectively organizing this search under a limited budget remains a central challenge in realizing the full potential of LLM-AAD for algorithm discovery [6, 7].

Existing LLM-AAD research primarily advances algorithm discovery by improving search strategies and drawing on past experience. Population management and tree-structured search improve candidate selection and expansion to explore the algorithm space more effectively [1, 5, 6]. Meanwhile, reflective feedback, records of semantic modifications, and trajectory memory guide offspring generation, allowing the search to draw on experience accumulated from previous attempts [2, 7, 8]. Together, these studies highlight two interrelated decisions: which algorithms receive opportunities for further evolution and how search experience guides their subsequent improvement.

Despite these advances, LLM-AAD methods remain prone to concentrating search on a few high-fitness algorithms, leaving alternative algorithmic ideas underexplored [9, 10]. We interpret this limitation through selection and generation: selection based on current fitness may prematurely discard ideas that have not been sufficiently refined. LLM-generated code also tends to exhibit reduced algorithmic diversity [11]; in evolutionary search, this tendency may redirect offspring toward familiar or existing high-performing designs before the original idea has been sufficiently developed. The former limits the evolutionary opportunities available to new ideas; the latter means that allocated opportunities need not develop the original idea. When high-fitness designs repeatedly serve as selection targets and contextual examples, these two processes may reinforce each other, concentrating search on a narrow set of directions.

Inspired by Go-Explore's principle of remembering and revisiting promising states [12], our key insight is that effective exploration requires both opportunities for different algorithmic ideas to evolve and support for their continued development from prior progress. Current fitness measures the performance of an existing implementation and need not reflect its performance after further refinement; this distinction between current performance and the value of continued development also arises in rising bandits [13]. Selection should therefore account for refinement effort and evolutionary progress when allocating further opportunities, while generation should draw on each idea's evolutionary experience to guide subsequent improvements, enabling sustained exploration of different ideas.

## References

[1] Liu, F., et al. [Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model](https://proceedings.mlr.press/v235/liu24bs.html). In *Proceedings of the 41st International Conference on Machine Learning (ICML)*, PMLR 235:32201–32223, 2024.

[2] Ye, H., et al. [ReEvo: Large Language Models as Hyper-Heuristics with Reflective Evolution](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4ced59d480e07d290b6f29fc8798f195-Abstract-Conference.html). In *Advances in Neural Information Processing Systems (NeurIPS)*, 37:43571–43608, 2024.

[3] Romera-Paredes, B., et al. [Mathematical discoveries from program search with large language models](https://doi.org/10.1038/s41586-023-06924-6). *Nature*, 625:468–475, 2024.

[4] Novikov, A., et al. [AlphaEvolve: A coding agent for scientific and algorithmic discovery](https://arxiv.org/abs/2506.13131). *arXiv preprint*, arXiv:2506.13131, 2025.

[5] Chen, B., et al. [A₂DEPT: Large Language Model–Driven Automated Algorithm Design via Evolutionary Program Trees](https://proceedings.mlr.press/v306/chen26hb.html). In *Proceedings of the 43rd International Conference on Machine Learning (ICML)*, PMLR 306:18654–18693, 2026.

[6] Zheng, Z., et al. [Monte Carlo Tree Search for Comprehensive Exploration in LLM-Based Automatic Heuristic Design](https://proceedings.mlr.press/v267/zheng25o.html). In *Proceedings of the 42nd International Conference on Machine Learning (ICML)*, PMLR 267:78338–78373, 2025.

[7] Jiang, J., et al. [DeltaEvolve: Accelerating Scientific Discovery through Momentum-Driven Evolution](https://proceedings.mlr.press/v306/jiang26g.html). In *Proceedings of the 43rd International Conference on Machine Learning (ICML)*, PMLR 306:52188–52212, 2026.

[8] Gungordu, O., et al. [PathWise: Planning through World Model for Automated Heuristic Design via Self-Evolving LLMs](https://proceedings.mlr.press/v306/gungordu26a.html). In *Proceedings of the 43rd International Conference on Machine Learning (ICML)*, PMLR 306:37779–37816, 2026.

[9] Dat, P. V. T., et al. [HSEvo: Elevating Automatic Heuristic Design with Diversity-Driven Harmony Search and Genetic Algorithm Using LLMs](https://doi.org/10.1609/aaai.v39i25.34898). In *Proceedings of the AAAI Conference on Artificial Intelligence*, 39(25):26931–26938, 2025.

[10] Chen, S., and Liu, Y. [Illuminating Heuristic Design: Language-Model-Driven Quality-Diversity Evolution for Combinatorial Optimization](https://doi.org/10.1109/TEVC.2026.3721299). *IEEE Transactions on Evolutionary Computation*, Early Access, 2026.

[11] Wu, F., et al. [Generative Monoculture in Large Language Models](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5178b2f2d7c44aa390c0777dc77b3f0c-Abstract-Conference.html). In *International Conference on Learning Representations (ICLR)*, 2025.

[12] Ecoffet, A., et al. [First return, then explore](https://doi.org/10.1038/s41586-020-03157-9). *Nature*, 590:580–586, 2021.

[13] Mussi, M., et al. [Best Arm Identification for Stochastic Rising Bandits](https://proceedings.mlr.press/v235/mussi24b.html). In *Proceedings of the 41st International Conference on Machine Learning (ICML)*, PMLR 235:36953–36989, 2024.
