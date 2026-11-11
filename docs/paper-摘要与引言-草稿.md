# EvoTrace：摘要与引言草稿

2026年11月11日第二稿。全文的主线是让种群在保留算法的同时保留它们的演化轨迹，并使这些轨迹成为预算分配与算法生成的依据。

论证顺序如下。

1. 有限预算下，搜索效果取决于评价投向何处、修改是否有针对性。
2. 这两项决定所需的依据蕴含在演化轨迹中，包括算法的形成过程、从它出发的已有尝试，以及各设计方向的进展。
3. 种群有界而不保留轨迹，搜索树保留轨迹而无界且未经整理。
4. 种群的更新与轨迹的整理面对同一批对象，可以统一为对一棵树的更新。
5. EvoTrace由此包含四个部分，即树状种群、反思简化、轨迹引导的预算分配和轨迹引导的生成算子。

---

## Abstract

**中文**

大语言模型（LLM）驱动的自动算法设计在有限的评价预算内反复选择、修改并评价候选算法。搜索在这一过程中不断积累演化轨迹，即每个算法由哪些修改逐步形成、从它出发做过哪些尝试、不同设计方向各自取得了怎样的进展。这些信息对选择哪个算法继续改进、对它做何种修改都有直接的指导意义，现有方法却难以利用。基于种群的方法只保留少数高分算法及其成绩，被淘汰算法所对应的尝试不再留存。保留完整搜索树的方法使历史随评价次数持续增长，其中多数是失败的或仅有细微差别的单步修改，难以直接用于后续设计。本文提出EvoTrace，将种群组织为一棵规模有界且持续生长的树，使种群在保留算法的同时保留它们的演化轨迹。树的节点是算法，连接表示算法之间的发展关系。每扩展并评价一批候选后，EvoTrace通过剪枝和反思简化整理这棵树。反思简化将一段发展过程收缩为一条连接，并把这段过程中的修改及其结果凝练为附着在该连接上的经验，从而在淘汰算法的同时保留它们所提供的认识。在这棵树上，EvoTrace依据算法的成绩及其发展历程分配评价预算，并让改进、融合与探索三种生成算子分别读取当前算法的形成过程、两个算法各自的发展分支以及全部设计方向的概貌。【结果待确认，在{问题类别}上，相同评价预算下EvoTrace相较{代表性方法}取得{具体收益}。】

**English**

Automatic algorithm design with large language models (LLMs) repeatedly selects, modifies, and evaluates candidate algorithms under a limited evaluation budget. As the search proceeds, it accumulates evolutionary trajectories that record how each algorithm was formed through successive modifications, which modifications have been attempted from it, and how far different design directions have progressed. This information bears directly on which algorithm deserves further effort and how it should be modified, yet existing methods can hardly exploit it. Population-based methods retain only a few top-scoring algorithms with their scores, and the attempts behind discarded algorithms are lost. Methods that keep the complete search tree let the history grow with every evaluation, and most of it consists of failed or nearly identical single-step modifications that are difficult to use in subsequent design. We present EvoTrace, which organizes the population as a bounded, continuously growing tree so that the population retains algorithms together with their evolutionary trajectories. Nodes of the tree are algorithms, and links represent how one algorithm developed into another. After each batch of candidates is generated and evaluated, EvoTrace reorganizes the tree through pruning and reflective simplification. Reflective simplification contracts a stretch of development into a single link and distills the modifications and outcomes within it into an experience record attached to that link, so that the knowledge gained from discarded algorithms is preserved when they are removed. On this tree, EvoTrace allocates the evaluation budget according to both the score of an algorithm and the course of its development, and its three generation operators, refinement, crossover, and exploration, read the formation of the current algorithm, the development branches of two algorithms, and an overview of all design directions, respectively. [RESULTS TO BE VERIFIED. On {problem classes}, EvoTrace achieves {specific gains} over {representative methods} under the same evaluation budget.]

---

## Introduction

**中文**

高性能算法的设计长期依赖专家经验与反复试验。自动算法设计（Automated Algorithm Design，AAD）旨在自动完成这一过程 [11–13]。近年来，大语言模型（LLM）驱动的演化搜索成为AAD的主要方法 [1–4]。这类方法维护一组候选算法，迭代地从中选择算法，提示LLM对其进行修改，并在一组问题实例上评价所得的新算法，已在组合优化 [1, 2, 17–19] 和数学发现 [3, 4] 中发现了优于人工设计的算法。评价一个候选需要实际运行该算法，代价较高，一次搜索通常只能进行数百至上千次评价。在这样的预算下，搜索的效果取决于每一次评价投向哪个算法，以及每一次修改是否具有针对性。

回答这两个问题所需的依据，很大程度上蕴含在搜索自身的历史之中。随着搜索进行，每个候选算法都处在一段演化轨迹（evolutionary trajectory）之中。这段轨迹记录了该算法经由哪些修改从早期版本逐步形成，从它出发已经尝试过哪些修改及其结果，以及其他设计方向在同一时期取得了怎样的进展。一个在连续几个阶段中稳定改进的算法，与一个经过多次修改仍未见效的算法，即使当前成绩相近，继续投入评价的价值也不相同。同样，了解一个算法中的各项计算分别在何时引入、是否带来了改进，以及哪些修改已经失败，可以使下一次修改避开重复的尝试，并针对尚未解决的环节。演化轨迹因而为预算分配和算法修改提供了成绩与代码本身所不具备的依据。

现有方法对演化轨迹的利用受限于它们的搜索结构。基于种群的方法 [1–4, 17] 以固定规模的种群保存成绩最好的若干算法，并依据成绩进行选择与淘汰。种群中只有算法及其成绩，算法之间的派生关系和被淘汰算法所对应的尝试均不保留，选择与生成只能依据当前的代码和成绩。基于树的方法 [5, 6, 22] 将全部已评价的算法按派生关系组织为搜索树，完整地保留了演化轨迹。然而，搜索树随评价次数不断增长，其中大部分节点是失败的或仅有细微差别的单步修改。这些方法主要利用节点的访问次数与成绩统计来引导选择，轨迹中与设计有关的内容并未得到整理。另一些工作通过反思从搜索历史中总结文本经验 [2, 8, 10]，这些经验保存在独立于种群的记忆之中，其更新与种群的更新各自进行。总体而言，种群有界但不保留演化轨迹，搜索树保留演化轨迹但无界且未经整理。如何在有限的规模内保留对后续设计有用的演化轨迹，仍是一个未解决的问题。

我们注意到，种群的更新与演化轨迹的整理面对的是同一批对象。种群更新需要决定哪些算法继续参与搜索，轨迹整理需要决定哪些发展过程值得保留、以何种粒度保留。当一段发展过程中的中间算法不再具有继续改进的价值时，它们可以从种群中移除。它们共同说明的内容，包括这一阶段尝试了哪些修改、哪些修改最终保留在更好的算法之中，则可以凝练为一条经验，保留在相应的发展关系上。基于这一观察，我们将种群的更新与轨迹的整理统一为对同一结构的更新，使种群在保持有界的同时携带经过整理的演化轨迹。

本文据此提出EvoTrace，一种基于树状种群的自动算法设计方法。EvoTrace将种群组织为一棵持续生长的树，节点是当前保留的算法，连接表示一个算法由另一个算法发展而来，并附有这段发展的经验记录。树的深度、分支数量和节点总数受到约束，使种群规模保持有界。搜索以批为单位进行，每一批从树上选择若干算法生成新的候选并加以评价，随后通过剪枝和**反思简化**（reflective simplification）整理这棵树。剪枝移除不再参与搜索的算法或分支，并借助代码结构与语义相似度保留计算方式不同的算法。反思简化作用于被移除算法所在的发展过程。沿一条发展路径，它将若干中间步骤收缩为一条连接，并把这些步骤中的修改及其结果凝练为这条连接上的阶段经验。在一个算法的多个后续方向之间，它比较各个方向的进展，并把比较的结论记录在这个共同的起点上。较早的发展阶段会随搜索被再次凝练，树上的经验因而在靠近当前算法处较为详细，在远离当前算法处较为概括。

整理后的树为搜索中的各项决定提供依据。在预算分配中，EvoTrace不仅考虑一个算法的当前成绩，还考虑它的发展历程以及从它出发的已有尝试的结果，据此估计继续改进它的价值，并决定下一批评价在各个算法与各种生成算子之间的分配。在算法生成中，三种算子从树上读取不同范围的演化轨迹。改进算子读取当前算法的形成路径以及从它出发的已有尝试，以确定下一步修改哪一项计算。融合算子读取两个算法各自的发展分支，以确定从参考算法中迁移哪一项计算、怎样与当前算法结合。探索算子读取各主要设计方向及其进展的概貌，以提出尚未覆盖的计算方式。生成与评价的结果作为新的节点和连接进入这棵树，并在下一次整理中更新其上的经验。

本文的主要贡献如下。

- 我们提出树状种群，将算法及其演化轨迹组织在同一个有界且持续生长的结构中，使种群的更新与演化轨迹的整理成为同一过程。
- 我们提出反思简化，在收缩发展路径和合并分支的同时，把被移除部分的修改及其结果凝练为保留在树上的阶段经验与比较经验。
- 我们基于树状种群设计了轨迹引导的预算分配和三种轨迹引导的生成算子，使评价的投向与算法的修改都以演化轨迹为依据。
- 【结果待确认，我们在{问题类别与规模}上评估EvoTrace。在相同评价预算下，EvoTrace相较{代表性方法}在{核心指标}上取得{具体收益}，消融实验验证了{各组成部分的作用}。】

**English**

Designing high-performing algorithms has long relied on expert knowledge and repeated trial and error. Automated algorithm design (AAD) aims to automate this process [11–13]. In recent years, evolutionary search driven by large language models (LLMs) has become the dominant approach to AAD [1–4]. Such methods maintain a set of candidate algorithms, iteratively select algorithms from it, prompt an LLM to modify them, and evaluate the resulting algorithms on a set of problem instances. They have discovered algorithms that outperform hand-crafted designs in combinatorial optimization [1, 2, 17–19] and mathematical discovery [3, 4]. Evaluating a candidate requires running the algorithm and is therefore expensive, so a search can typically afford only a few hundred to a thousand evaluations. Under such a budget, the effectiveness of the search depends on which algorithm each evaluation is spent on and on whether each modification is well targeted.

Much of the information needed to answer these two questions lies in the history of the search itself. As the search proceeds, every candidate algorithm becomes part of an evolutionary trajectory. The trajectory records through which modifications the algorithm was formed from earlier versions, which modifications have already been attempted from it and with what outcomes, and how far other design directions have progressed in the meantime. An algorithm that has improved steadily over several stages and an algorithm that has resisted many modifications deserve different amounts of further evaluation even when their current scores are close. Likewise, knowing when each computation in an algorithm was introduced, whether it brought an improvement, and which modifications have already failed allows the next modification to avoid repeated attempts and to address what remains unsolved. Evolutionary trajectories therefore provide a basis for budget allocation and algorithm modification that scores and code alone do not.

The extent to which existing methods exploit evolutionary trajectories is limited by their search structures. Population-based methods [1–4, 17] keep a fixed number of top-scoring algorithms and perform selection and elimination by score. The population contains only algorithms and their scores. Derivation relations among algorithms and the attempts behind discarded algorithms are not retained, so selection and generation can rely only on current code and scores. Tree-based methods [5, 6, 22] organize all evaluated algorithms into a search tree according to their derivation relations and thus keep the complete trajectories. However, the search tree grows with every evaluation, and most of its nodes are failed or nearly identical single-step modifications. These methods guide selection mainly through visit counts and score statistics of the nodes, while the design-related content of the trajectories is left unorganized. Other studies summarize textual lessons from the search history through reflection [2, 8, 10]. The lessons are stored in a memory separate from the population and are updated independently of it. In short, a population is bounded but does not retain evolutionary trajectories, whereas a search tree retains them but is unbounded and unorganized. How to retain, within a limited size, the evolutionary trajectories that are useful for subsequent design remains an open problem.

We observe that updating the population and organizing evolutionary trajectories operate on the same objects. Updating the population decides which algorithms continue to take part in the search. Organizing trajectories decides which courses of development are worth keeping and at what granularity. When the intermediate algorithms along a course of development are no longer worth improving, they can be removed from the population. What they jointly show, including which modifications were attempted during this stage and which of them survive in a better algorithm, can be distilled into an experience record that stays on the corresponding development relation. Based on this observation, we unify population update and trajectory organization as the update of a single structure, so that the population remains bounded while carrying organized evolutionary trajectories.

We accordingly present EvoTrace, an automated algorithm design method built on a tree-structured population. EvoTrace organizes the population as a continuously growing tree. Nodes are the algorithms currently retained, and a link indicates that one algorithm was developed from another and carries an experience record of that development. The depth, the branching, and the total number of nodes of the tree are constrained, which keeps the population bounded. The search proceeds in batches. In each batch, several algorithms are selected from the tree to generate new candidates, the candidates are evaluated, and the tree is then reorganized through pruning and *reflective simplification*. Pruning removes algorithms or branches that no longer take part in the search, and it uses code structure and semantic similarity to retain algorithms that compute in different ways. Reflective simplification acts on the courses of development that contain the removed algorithms. Along a development path, it contracts several intermediate steps into a single link and distills the modifications in these steps and their outcomes into stage experience on that link. Among the directions that follow from one algorithm, it compares their progress and records the conclusion at their common starting point. Earlier stages of development are distilled again as the search continues, so that experience on the tree is detailed near the current algorithms and condensed farther away from them.

The reorganized tree provides the basis for the decisions made in the search. For budget allocation, EvoTrace considers not only the current score of an algorithm but also the course of its development and the outcomes of the attempts already made from it. It thereby estimates the value of improving the algorithm further and decides how the next batch of evaluations is distributed over algorithms and generation operators. For algorithm generation, three operators read evolutionary trajectories of different scopes from the tree. The refinement operator reads the formation path of the current algorithm and the attempts already made from it, in order to determine which computation to modify next. The crossover operator reads the development branches of two algorithms, in order to determine which computation to transfer from the reference algorithm and how to combine it with the current one. The exploration operator reads an overview of the major design directions and their progress, in order to propose a way of computing that has not yet been covered. The generated and evaluated candidates enter the tree as new nodes and links, and the experience on the tree is updated in the next reorganization.

Our main contributions are as follows.

- We propose the tree-structured population, which organizes algorithms and their evolutionary trajectories in a single bounded and continuously growing structure, so that updating the population and organizing evolutionary trajectories become one process.
- We propose reflective simplification, which contracts development paths and merges branches while distilling the modifications and outcomes of the removed parts into stage experience and comparative experience retained on the tree.
- On the tree-structured population, we design trajectory-guided budget allocation and three trajectory-guided generation operators, so that both the spending of evaluations and the modification of algorithms are grounded in evolutionary trajectories.
- [RESULTS TO BE VERIFIED. We evaluate EvoTrace on {problem classes and scales}. Under the same evaluation budget, EvoTrace achieves {specific gains} over {representative methods} on {primary metric}, and ablation studies verify {the effect of each component}.]
