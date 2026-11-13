# EvoTrace: Evolving a Tree-Structured Population with Reflective Simplification

## Abstract

Large language models enable automated algorithm design through iterative program generation and evaluation. Each search produces both candidate algorithms and evidence about how those algorithms develop. Using this evidence requires a search structure that supports candidate selection while retaining useful developmental context. We present EvoTrace, which maintains a tree-structured population of algorithms, derivation relationships, and accumulated experience. Its central mechanism, dynamic reflective simplification, updates candidate survival and the representation of their history together. After each evaluated addition, EvoTrace selects recently used algorithms and distinct quality representatives, then retains the branching ancestors needed to connect them. This single retained set determines both branch pruning and path contraction. Reflection distills removed side branches into node-level selection experience and contracted paths into edge-level variation experience. Recorded evaluation and usage counts follow the same partition, keeping historical effort associated with its source region. The resulting tree supports further search through quality- and usage-aware parent sampling and operator-specific context for refinement, crossover, and exploration. EvoTrace thus organizes candidate survival and experience accumulation within one evolving population: retained programs provide starting points, while their surrounding history informs subsequent algorithm design.

## 1. Introduction

Large language models (LLMs) enable automated algorithm design (AAD) by generating executable programs that can be evaluated and revised. Evolutionary search turns this capability into an iterative process: generate candidates, measure their quality, and use existing candidates to produce the next ones. Learned variation operators support both program mutation and recombination ([Lehman et al., 2022](https://arxiv.org/abs/2206.08896); [Meyerson et al., 2024](https://doi.org/10.1145/3694791)). FunSearch searches over programs with an evolutionary population ([Romera-Paredes et al., 2024](https://www.nature.com/articles/s41586-023-06924-6)). EoH evolves executable heuristics together with descriptions of their computational ideas ([Liu et al., 2024](https://proceedings.mlr.press/v235/liu24bs.html)). These approaches make the organization of search consequential. Under a limited evaluation budget, the system must decide which algorithms to develop and what previous attempts should contribute to the next design.

Existing methods address these decisions through several forms of search state. Population-based methods select candidates for continued evolution, while reflective methods also use evaluated comparisons to guide generation, as in ReEvo ([Ye et al., 2024](https://papers.nips.cc/paper/2024/hash/4ced59d480e07d290b6f29fc8798f195-Abstract-Conference.html)). Tree-based methods preserve derivation relationships and use them for selection or generation. MCTS-AHD, for example, conditions a generation action on algorithms along a tree path ([Zheng et al., 2025](https://proceedings.mlr.press/v267/zheng25o.html)). More recent approaches combine derivation structure with population updates or pruning, including PathWise and PhyloEvolve ([Gungordu et al., 2026](https://proceedings.mlr.press/v306/gungordu26a.html); [Zhao et al., 2026b](https://arxiv.org/abs/2601.14523)). Together, these methods show that both candidate quality and developmental context can participate in algorithm search.

Their connection raises a further question: **how should the search update its candidates and the history attached to them together?** A candidate in a lineage is both a possible starting point and a position in a development process. Removing an intermediate version changes how a descendant's formation is represented. Removing a side branch changes what remains available about alternatives tried around an ancestor. These two changes carry different information. A path records successive revisions leading to a retained implementation; a side branch records attempts that diverged from a location in the search. Candidate selection must therefore determine how the surrounding evidence is represented after an algorithm leaves the explicit population.

Our central insight is that the survival of a program and the retention of its design experience can be decided separately. An implementation can leave the selectable population while its evaluated development remains available to subsequent design. This suggests a tree-structured population whose updates operate on algorithms, relationships, and experience together. Retained nodes provide implementations for further development. Retained connections preserve the branching relationships among those implementations. Node and edge experience carries a condensed account of the regions that have been removed.

We instantiate this idea in **EvoTrace** through **dynamic reflective simplification**. After each evaluated addition, EvoTrace determines one retained set from recent actual use and current quality differences, then restores the branching ancestors required by that set. The original topology determines which removed versions form intermediate paths and which form side branches. Path contraction writes variation experience onto the reconnected edge; branch pruning writes selection experience onto the nearest retained ancestor. Evaluation and usage counts are transferred along the same regions. The updated tree then supports parent sampling and three views of algorithm development: a formation path for refinement, diverging paths for crossover, and a view of design directions for exploration. Growth, selection, and experience accumulation continue on this maintained structure.

We make two contributions:

- **A tree-structured population maintained through dynamic reflective simplification.** One retention decision coordinates candidate selection, preservation of necessary forks, branch pruning, and path contraction. Region-specific reflection places the removed history on the retained node or edge that gives it context.
- **Algorithm generation grounded in the maintained population.** EvoTrace uses retained algorithms for budget allocation and provides refinement, crossover, and exploration with the relevant paths, evaluated alternatives, and accumulated experience. These views connect the consequences of population updates to subsequent design decisions.

## 2. Related Work

### 2.1. Automated Algorithm Design

Algorithm design has been automated through several search spaces. Genetic programming evolves executable programs ([Koza, 1994](https://doi.org/10.1007/bf00175355)). Hyper-heuristics select or construct heuristics from available components ([Burke et al., 2013](https://doi.org/10.1057/jors.2013.71)), while algorithm configuration optimizes the settings of an existing solver, as in irace ([López-Ibáñez et al., 2016](https://doi.org/10.1016/j.orp.2016.09.002)). Surveys of automated metaheuristic design organize this work through design spaces, search strategies, and evaluation procedures ([Zhao et al., 2024a](https://arxiv.org/abs/2303.06532)). Recent surveys also distinguish the roles of LLMs in optimization, prediction, knowledge extraction, and algorithm design ([Liu et al., 2026](https://doi.org/10.1145/3787585)). EvoTrace concerns the search state used to generate and select executable algorithm implementations under an evaluation budget.

### 2.2. LLM-Driven Program Evolution

Evolution through Large Models uses a code model for program mutation ([Lehman et al., 2022](https://arxiv.org/abs/2206.08896)), and language-model crossover generates offspring from multiple text-based genotypes ([Meyerson et al., 2024](https://doi.org/10.1145/3694791)). FunSearch combines program generation and evaluation with an island population ([Romera-Paredes et al., 2024](https://www.nature.com/articles/s41586-023-06924-6)). EoH jointly evolves heuristic descriptions and code ([Liu et al., 2024](https://proceedings.mlr.press/v235/liu24bs.html)), while LLaMEA generates and refines metaheuristics using execution feedback ([van Stein and Bäck, 2025](https://arxiv.org/abs/2405.20132)). AlphaEvolve extends evaluator-guided code evolution to scientific and computational tasks ([Novikov et al., 2025](https://arxiv.org/abs/2506.13131)). These methods provide the program-generation and evaluation setting in which EvoTrace operates.

Population maintenance determines which candidates remain available. HSEvo combines diversity measures with harmony search and genetic operations ([Dat et al., 2025](https://doi.org/10.1609/aaai.v39i25.34898)). MEoH treats heuristic evolution as a multi-objective problem ([Yao et al., 2025](https://arxiv.org/abs/2409.16867)). CDEoH maintains algorithm-category diversity ([Wang et al., 2026](https://arxiv.org/abs/2603.19284)), and PartEvo constructs niches for collaborative evolutionary search ([Hu and Zhang, 2025](https://papers.nips.cc/paper_files/paper/2025/hash/e389b15166cf98966ba058965a8c17e3-Abstract-Conference.html)). ShinkaEvolve studies parent sampling and novelty assessment, including a weighted sampler that combines fitness and offspring counts ([Lange et al., 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/7886b9bafe76c52fd568db10ff9772df-Abstract-Conference.html)). EvoTrace adapts this sampler and studies how selection also updates developmental relationships and the experience attached to removed candidates.

Other approaches change the generation process or the unit being evolved. CALM co-evolves heuristics and the generating language model through reinforcement learning ([Huang et al., 2025](https://arxiv.org/abs/2505.12285)). EvoStage decomposes design into stages with intermediate feedback ([Lu et al., 2026](https://arxiv.org/abs/2603.07970)). BEAM evolves high-level algorithm structures in an outer loop and realizes their components through memory-assisted inner MCTS ([Xiang et al., 2026](https://arxiv.org/abs/2604.12898)). These approaches address generator adaptation and algorithmic composition; EvoTrace focuses on maintaining the population and its developmental context.

### 2.3. Tree, Graph, and Lineage Structure

Bandit-based MCTS allocates search through node values and visit statistics ([Kocsis and Szepesvári, 2006](https://doi.org/10.1007/11871842_29)). MCTS-AHD applies tree search to heuristic evolution and introduces a tree-path reasoning action ([Zheng et al., 2025](https://proceedings.mlr.press/v267/zheng25o.html)). CogMCTS incorporates multi-round cognitive feedback and negative outcomes ([Wang et al., 2025a](https://arxiv.org/abs/2512.08609)). Clade-AHD aggregates descendant evaluations into Bayesian beliefs for clade-level selection and freezes sufficiently evaluated, low-performing clades ([Lai et al., 2026](https://arxiv.org/abs/2602.00549)). Related work on self-improving coding agents maintains an archive of agent implementations in the Darwin Gödel Machine ([Zhang et al., 2026](https://arxiv.org/abs/2505.22954)) and uses descendant performance to estimate self-improvement potential in the Huxley-Gödel Machine ([Wang et al., 2025b](https://arxiv.org/abs/2510.21614)).

The meaning of a structural edge also matters. A2DEPT uses a tree-structured evolutionary search over complete algorithms ([Chen et al., 2026](https://arxiv.org/abs/2604.24043)). DGA2D represents functional operators and their compositions as a directed graph ([Zhao et al., 2026a](https://arxiv.org/abs/2608.00700)), while GAE uses typed computation graphs for structure-aware evolutionary control ([Chen and Cheng, 2026](https://arxiv.org/abs/2607.10127)). Code Evolution Graphs analyzes how generated algorithms change during evolution ([van Stein et al., 2025](https://arxiv.org/abs/2503.16668)). These structures represent candidate derivation, program composition, or retrospective analysis. EvoTrace's primary-parent tree represents the development of selectable algorithms.

PathWise and PhyloEvolve are particularly relevant to the connection between populations and derivations. PathWise combines an outer population with an inner entailment graph, recording derivation metadata for coordinated planning, generation, and reflection ([Gungordu et al., 2026](https://proceedings.mlr.press/v306/gungordu26a.html)). PhyloEvolve maintains a phylogenetic forest with an elite pool, trajectory-conditioned generation, and pruning at several structural scales ([Zhao et al., 2026b](https://arxiv.org/abs/2601.14523)). EvoTrace studies an update on one continuously maintained tree: a single retained set induces both path contraction and branch pruning, and the removed regions contribute experience at their corresponding retained locations.

### 2.4. Reflection and Experience Reuse

Self-Refine iteratively revises outputs using self-feedback ([Madaan et al., 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/91edff07232fb1b55a505a9e9f6c0ff3-Abstract-Conference.html)). Reflexion retains verbal feedback across agent episodes ([Shinn et al., 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html)), and ExpeL extracts reusable insights from collected task experiences ([Zhao et al., 2024b](https://ojs.aaai.org/index.php/AAAI/article/view/29936)). In algorithm design, ReEvo uses short-term comparisons and accumulated long-term reflection ([Ye et al., 2024](https://papers.nips.cc/paper/2024/hash/4ced59d480e07d290b6f29fc8798f195-Abstract-Conference.html)). EvoPH co-evolves prompts and heuristics using performance-guided reflection ([Liu et al., 2025](https://arxiv.org/abs/2509.24509)). RefineEvo combines planning with positive and negative experience ([Wu et al., 2026](https://arxiv.org/abs/2607.11358)), while MeEvo alternates population evolution with reflection over shared design history ([Qiu et al., 2026](https://arxiv.org/abs/2606.14202)).

Experience can also be organized around changes and branches. DeltaEvolve represents successive modifications through structured semantic deltas and exposes them through a hierarchical context system ([Jiang et al., 2026](https://arxiv.org/abs/2602.02919)). MEMOIR combines branch-local execution memory with global summaries distilled at branch termination for cross-branch transfer ([Haji et al., 2026](https://arxiv.org/abs/2605.17539)). EvoTrace connects experience condensation to explicit population simplification. Each summary corresponds to a removed path or side-branch region and remains on the retained edge or ancestor that supplies its context. Subsequent operators retrieve this experience through the same tree that organizes the surviving programs.

## 3. Method

### 3.1. Problem and Overview

We consider AAD for a specified task interface. A candidate program implements a designated function used by an enclosing solver. The evaluator executes the program on a fixed search dataset and returns a fitness value or failure feedback. We use the convention that lower fitness is better. The search has a budget of $B$ program evaluations, including unsuccessful executions and repair evaluations. Model calls for generation, repair, and reflection are accounted for separately.

EvoTrace maintains an explicit tree-structured population alongside a complete attempt archive. The tree supplies selectable algorithms and context for generation; the archive records generated programs, evaluations, and population updates. Each iteration selects an algorithm, generates and evaluates a candidate, attaches the final evaluated version, and simplifies the updated population. The returned program is the feasible archived program with the best search fitness, with earlier creation order breaking ties. Any held-out evaluation is performed after this choice.

![EvoTrace overview: an evolving tree of algorithms, structurally located experience, and subsequent generation](figures/evotrace/01-main-framework.png)

**Figure 1. An evolving population of algorithms and experience.** Retained algorithms provide starting points for further search. Simplified paths contribute variation experience $V$ to reconnected edges; removed side branches contribute selection experience $S$ to retained ancestors. Refinement, crossover, and exploration read different views of the maintained tree. Each evaluated candidate returns as a child of its primary parent, continuing growth on the same structure.

### 3.2. Tree-Structured Population

Let $T=(\mathcal N\cup\{r\},\mathcal E)$ be the current tree, where $r$ is a virtual root without an implementation. Each algorithm node stores code, a description of its computation, fitness $f(v)$ or failure feedback, and usage records. Every node has one primary parent. Crossover also records a reference algorithm as an auxiliary input; this reference does not add a second edge to the maintained tree. All retained feasible nodes, including internal nodes, can be selected for further development.

The tree carries two forms of experience. **Selection experience** $S_v$, attached to a node, condenses side branches removed around that location. **Variation experience** $V_{uv}$, attached to an edge, condenses the intermediate development contracted between its retained endpoints. The virtual root can carry selection experience from removed initial directions. An ordinary edge represents one generation attempt. A contracted edge represents a developmental stage and is marked accordingly.

Initialization and exploration create design directions. Refinement and crossover inherit the direction of the primary parent. Direction identifiers remain associated with descendants when an origin is removed. They organize the exploration view of the tree.

### 3.3. Dynamic Reflective Simplification

Once initialization is complete, EvoTrace checks the population after every evaluated addition. The update has three steps: select algorithms, retain necessary forks, and partition the removed regions. All three steps use the same tree snapshot before simplification.

**Selecting quality representatives.** Let $\mathcal P\subseteq\mathcal N$ contain the feasible nodes, ranked by fitness and then creation order. Define rank quality as

$$
q(v)=1-\frac{\operatorname{rank}(v)}{\max(1,|\mathcal P|-1)},
\qquad v\in\mathcal P,
\tag{1}
$$

where the best node has rank zero. EvoTrace retains the best node and considers the remaining candidates with $q(v)\geq0.50$ in ranked order. A candidate is covered if an already selected representative has identical normalized source or an abstract syntax tree (AST) distance below $0.08$. When the fitness values differ by at most $10^{-9}$, the distance threshold is $0.20$. Otherwise, the candidate joins the representative set $\mathcal R$. Every representative used for coverage has fitness no worse than the candidate being considered.

The distance is a multiset Jaccard distance over AST parent–child type features and traversal bigrams. It retains operator types, called function names, and control structure while omitting ordinary variable identities and numerical values as feature values. Normalized-source identity is a separate check that preserves variables and constants while ignoring formatting, comments, and docstrings. These checks define the implemented coverage heuristic.

**Retaining recently used algorithms.** The evaluation clock $t$ counts actual program evaluations. Each feasible node has a last-use time $u(v)$, refreshed at birth or when an evaluated attempt uses it as primary parent or crossover reference. The recent set is

$$
\mathcal U=\{v\in\mathcal P:t-u(v)<W\},
\qquad W=40.
\tag{2}
$$

Failed executions and repair evaluations advance the clock. Displaying a node in context or retaining it as a structural connector does not refresh its time. Recent retention preserves a candidate's eligibility for continued development during the window; the parent sampler determines its actual opportunities.

**Preserving necessary forks.** The retained set is

$$
\mathcal K=\{r\}\cup\mathcal R\cup\mathcal U
\cup\operatorname{LCA}(\mathcal R\cup\mathcal U),
\tag{3}
$$

Here, $\operatorname{LCA}(A)$ denotes the set of lowest common ancestors of pairs of nodes in $A$, retaining the forks needed to connect the selected routes. Unary intermediate ancestors are eligible for contraction. Quality representatives are determined before recent retention and structural completion are added. Thus, a node retained solely for recency or connectivity does not exclude a better candidate from the representative set. The update computes ranks and representatives once. The number of retained nodes follows recent use, quality differences, and branching structure; the rule imposes no fixed population cap.

**Partitioning paths and side branches.** For each retained algorithm, EvoTrace finds its nearest retained ancestor in the original tree. Removed intermediate nodes on this connection form a path-contraction region. The remaining removed nodes are grouped by their nearest retained ancestor into side-branch regions. The root receives a region when no algorithm ancestor is retained. Each removed node belongs to exactly one region. Multiple side branches assigned to the same receiver are processed together.

This partition can shorten a path and remove its side branches in the same update. Consider $A\rightarrow B\rightarrow C\rightarrow D$, with an additional branch $B\rightarrow X$, when only $A$ and $D$ remain. Nodes $B,C$ belong to the contracted path, while $X$ belongs to the side-branch region received by $A$. Earlier experience already attached to $B$ or $C$ travels with those nodes into the path region.

![One retained set determines path contraction, branch pruning, and the destinations of experience](figures/evotrace/04-reflective-simplification.png)

**Figure 2. Joint structural simplification and experience transfer.** Intermediate versions $B,C$ and their inherited experience contribute to $V_{AD}$. The newly removed branch $X$ contributes to $S_A$. Its evidence is distinct from earlier side-branch experience already carried by $B$. Both destinations are determined on the original topology before any reconnection.

### 3.4. Reflection and Historical Effort

**Region-specific reflection.** EvoTrace makes one reflection call for each final contraction region and each grouped side-branch region. A path reflection receives retained endpoint implementations, their net difference, intermediate modifications and outcomes, and inherited node and edge experience. A side-branch reflection receives the receiver's implementation when available, the removed attempts and their actual parent-relative differences, evaluation or error feedback, and earlier relevant experience. The virtual root supplies a task-level origin when there is no retained algorithm receiver. Contracted inputs are explicitly presented as stages of development.

The model produces a concise experience paragraph relating implemented computations to observed outcomes and useful earlier experience. Reflection uses code, measured results, errors, and existing summaries as evidence. The generating model's rationale is not treated as evidence about an edit's effect. Endpoint fitness changes are computed by the system. All regions in one update read the same original snapshot, so the update does not summarize newly produced summaries again within that pass.

Path reflection updates $V_{uv}$ on the reconnected edge. Side-branch reflection updates $S_u$ at the receiver, incorporating its existing selection experience. When a later update removes a carrier, its accumulated experience enters the appropriate new region. This recursion retains a condensed account of development while allowing intermediate implementations to leave the explicit population. Textual reflection is lossy; the complete attempt archive remains available for inspection.

**Transferring historical effort.** Each node records its own version count, actual evaluations, failed evaluations, primary-parent uses, and crossover-reference uses. Selection and variation experience carry separate inherited counts of the same five quantities. During simplification, these counts follow the same disjoint regions as the evidence. Path transfer includes the removed nodes, their inherited histories, and the history on the descendant's old incoming edge. Side-branch transfer includes its removed nodes and histories, together with the receiver's previous selection history. Retained endpoints keep their own counts.

For each quantity, total recorded effort equals the sum of retained nodes' own counts, node selection histories, edge variation histories, and root selection history. The aggregated counts accompany reflection as information about the development represented by a region. They include earlier absorbed history. The current algorithm's own development count remains separate and continues to govern its parent-sampling discount.

### 3.5. Search from the Maintained Tree

**Parent and reference selection.** EvoTrace adapts the weighted parent sampler of ShinkaEvolve ([Lange et al., 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/7886b9bafe76c52fd568db10ff9772df-Abstract-Conference.html)). For each retained feasible node, its weight is

$$
w(v)=\sigma\!\left(10\frac{m-f(v)}{s}\right)\frac{1}{1+c(v)},
\qquad
p(v)=\frac{w(v)}{\sum_{z\in\mathcal P}w(z)},
\tag{4}
$$

where $m$ is median fitness, $s$ is a robust scale based on the median absolute deviation, $\sigma$ is the sigmoid function, and $c(v)$ is the cumulative number of evaluated child attempts generated directly from $v$. This count includes failed and subsequently removed children. Reconnection and inherited historical uses do not increment it. The sampler applies to internal and leaf algorithms under the same rule.

After selecting a parent, EvoTrace samples Refine, Crossover, or Explore with probabilities $0.45$, $0.25$, and $0.30$, respectively. Crossover first excludes the parent and normalized-source duplicates, then applies the rank-quality rule to the available references. Candidates with quality at least $0.30$ are eligible. EvoTrace prefers an unused ordered parent–reference pair and chooses the largest AST distance within that pool, breaking ties by fitness and creation order. It permits pair reuse once all eligible pairs have been used and uses Refine when no eligible reference exists.

**Context at the scale of the design decision.** The three operators read different views of the same population:

| Operator | Structural context | Current design task |
| --- | --- | --- |
| Refine | Current code and fitness; retained formation path; node selection experience; visible outgoing trials or developmental stages | Revise the current computation using its development and evaluated alternatives |
| Crossover | Both implementations; their common ancestor and diverging retained paths; relevant node and edge experience; outgoing trials from the primary parent | Decide which computations to transfer and how to integrate them |
| Explore | Distinct direction-entry ideas; the most recently generated retained version in each selected direction; node and edge experience; root experience; one existing implementation | Construct an alternative with a different main computation |

An ordinary transition supplies a difference relative to its actual parent and its outcome. A contracted transition supplies endpoint outcomes and accumulated variation experience. Difference sequences include the baseline code needed to interpret them. Crossover presents the common ancestor once. Exploration organizes ideas and experience before showing an existing implementation, allowing the model to reconsider its representation or procedure.

The context builder first assembles the relevant retained material. If a generation prompt exceeds its input budget, it reduces complete entries from path, trial, or direction views according to the actual tokens released. Current code, reference code, and required difference baselines remain available. Context fitting changes a prompt view; it does not change the tree, retention decisions, or experience ownership. Each operator returns a complete implementation and a description of its computation.

### 3.6. Search Procedure

Algorithm 1 summarizes the search. Initialization creates five feasible programs under the same evaluation budget. Each generation attempt has at most one repair opportunity and contributes at most one final evaluated node. Repair re-evaluation consumes budget separately. Attempts that produce no program evaluation do not enter the tree or advance its clock.

```text
Algorithm 1: EvoTrace
Input: task interface, evaluator, LLM, evaluation budget B
Create a virtual root, an empty population, and an attempt archive
while evaluation budget remains and the stopping rule is not met:
    if fewer than five feasible initial programs have been obtained:
        build an initialization prompt
    else:
        sample a parent and operator; choose a reference when needed
        build and fit the corresponding tree-context view
    generate and evaluate a program, allowing at most one repair
    record all requests and evaluation outcomes
    if the attempt contains an evaluated program:
        attach its final evaluated version; update actual-use records
        if initialization is complete:
            select representatives and recent nodes on one snapshot
            add necessary branching ancestors to obtain the retained set
            partition removed paths and side branches on that snapshot
            reflect on each region and transfer its historical counts
            reconnect retained nodes and attach the resulting experience
    commit the attempt and its population update
return the feasible archived program with the best search fitness
```

The procedure preserves the best current algorithm, primary-parent connectivity, unique ownership of removed regions, and the distinction between retained nodes' own use and inherited effort. These properties allow the updated tree to continue serving both candidate development and experience retrieval.

## Appendix A. Default Settings

| Component | Setting |
| --- | --- |
| Initialization; evaluation budget | Five feasible initial programs; 1,000 actual program evaluations |
| Recent retention | Fewer than 40 evaluations since birth or actual parent/reference use |
| Quality representatives | Rank quality at least 0.50; best-first coverage test |
| Coverage distance | Below 0.08; below 0.20 when fitness differs by at most $10^{-9}$ |
| Parent sampling | Sigmoid coefficient 10; own cumulative evaluated-child discount |
| Operator prior | Refine 0.45; Crossover 0.25; Explore 0.30 |
| Reference eligibility | Rank quality at least 0.30 after excluding the parent and identical normalized source |
| Direction-view filtering | Distinct normalized entry source and AST distance at least 0.12; latest retained version order |
| Evolution-prompt input | 30,000 tokens; whole-entry reduction when needed |
| Generation | Temperature 1.0; maximum output 16,384 tokens |
| Reflection | Temperature 0.3; maximum output 1,536 tokens; a paragraph of about 120 words, shorter for sparse evidence |

The scale $s$ in Equation (4) is the median absolute deviation with a floor of $10^{-6}$. If the deviation is at most $10^{-9}$, EvoTrace uses the median of absolute deviations larger than $10^{-9}$, with the same floor. If all sampling weights vanish, it samples uniformly. A reference pool with one candidate assigns it rank quality one.

Input tokens are counted by the serving tokenizer when supplied. A standalone client without a tokenizer uses UTF-8 byte length as a conservative proxy and records this choice. Context fitting applies to the three evolutionary generation prompts. Required material that still exceeds the budget produces an explicit error.

Reflection-call or parsing failures use a deterministic summary with a 960-character budget. This fallback preserves the structural update and numerical history transfer. Full region inputs and model responses remain archived. The search stops early after ten consecutive attempts without a program evaluation. Population updates are committed with their generating attempts; restoration reads committed updates without repeating their reflection calls.

## References

1. John R. Koza. 1994. [Genetic programming as a means for programming computers by natural selection](https://doi.org/10.1007/bf00175355). *Statistics and Computing* 4.
2. Edmund K Burke, Michel Gendreau, Matthew Hyde, et al. 2013. [Hyper-heuristics: a survey of the state of the art](https://doi.org/10.1057/jors.2013.71). *Journal of the Operational Research Society* 64, 1695–1724.
3. Manuel López-Ibáñez, Jérémie Dubois-Lacoste, Leslie Pérez Cáceres, et al. 2016. [The irace package: Iterated racing for automatic algorithm configuration](https://doi.org/10.1016/j.orp.2016.09.002). *Operations Research Perspectives* 3, 43–58.
4. Qi Zhao, Qiqi Duan, Bai Yan, et al. 2024a. [Automated Design of Metaheuristic Algorithms: A Survey](https://arxiv.org/abs/2303.06532). *arXiv:2303.06532*.
5. Fei Liu, Yiming Yao, Ping Guo, et al. 2026. [A Systematic Survey on Large Language Models for Algorithm Design](https://doi.org/10.1145/3787585). *ACM Computing Surveys* 58, 1–32.
6. Joel Lehman, Jonathan Gordon, Shawn Jain, et al. 2022. [Evolution through Large Models](https://arxiv.org/abs/2206.08896). *arXiv:2206.08896*.
7. Elliot Meyerson, Mark J. Nelson, Herbie Bradley, et al. 2024. [Language Model Crossover: Variation through Few-Shot Prompting](https://doi.org/10.1145/3694791). *ACM Transactions on Evolutionary Learning and Optimization* 4, 1–40.
8. Bernardino Romera-Paredes, Mohammadamin Barekatain, Alexander Novikov, et al. 2024. [Mathematical discoveries from program search with large language models](https://www.nature.com/articles/s41586-023-06924-6). *Nature* 625, 468–475.
9. Fei Liu, Xialiang Tong, Mingxuan Yuan, et al. 2024. [Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model](https://proceedings.mlr.press/v235/liu24bs.html). *Proceedings of the 41st International Conference on Machine Learning* 235, 32201–32223.
10. Niki van Stein, Thomas Bäck. 2025. [LLaMEA: A Large Language Model Evolutionary Algorithm for Automatically Generating Metaheuristics](https://arxiv.org/abs/2405.20132). *arXiv:2405.20132*.
11. Alexander Novikov, Ngân Vũ, Marvin Eisenberger, et al. 2025. [AlphaEvolve: A coding agent for scientific and algorithmic discovery](https://arxiv.org/abs/2506.13131). *arXiv:2506.13131*.
12. Pham Vu Tuan Dat, Long Doan, Huynh Thi Thanh Binh. 2025. [HSEvo: Elevating Automatic Heuristic Design with Diversity-Driven Harmony Search and Genetic Algorithm Using LLMs](https://doi.org/10.1609/aaai.v39i25.34898). *Proceedings of the AAAI Conference on Artificial Intelligence* 39, 26931–26938.
13. Shunyu Yao, Fei Liu, Xi Lin, et al. 2025. [Multi-objective Evolution of Heuristic Using Large Language Model](https://arxiv.org/abs/2409.16867). *arXiv:2409.16867*.
14. Yu-Nian Wang, Shen-Huan Lyu, Ning Chen, et al. 2026. [CDEoH: Category-Driven Automatic Algorithm Design With Large Language Models](https://arxiv.org/abs/2603.19284). *arXiv:2603.19284*.
15. Qinglong Hu, Qingfu Zhang. 2025. [Partition to Evolve: Niching-enhanced Evolution with LLMs for Automated Algorithm Discovery](https://papers.nips.cc/paper_files/paper/2025/hash/e389b15166cf98966ba058965a8c17e3-Abstract-Conference.html). *Advances in Neural Information Processing Systems* 38, Main Conference, 154771–154817.
16. Robert Lange, Yuki Imajuku, Edoardo Cetin. 2026. [ShinkaEvolve: Towards Open-Ended and Sample-Efficient Program Evolution](https://proceedings.iclr.cc/paper_files/paper/2026/hash/7886b9bafe76c52fd568db10ff9772df-Abstract-Conference.html). *International Conference on Learning Representations* 2026, 74026–74078.
17. Ziyao Huang, Weiwei Wu, Kui Wu, et al. 2025. [CALM: Co-evolution of Algorithms and Language Model for Automatic Heuristic Design](https://arxiv.org/abs/2505.12285). *arXiv:2505.12285*.
18. Chen Lu, Ke Xue, Chengrui Gao, et al. 2026. [Advancing Automated Algorithm Design via Evolutionary Stagewise Design with LLMs](https://arxiv.org/abs/2603.07970). *arXiv:2603.07970*.
19. Chuyang Xiang, Yichen Wei, Jiale Ma, et al. 2026. [BEAM: Bi-level Memory-adaptive Algorithmic Evolution for LLM-Powered Heuristic Design](https://arxiv.org/abs/2604.12898). *arXiv:2604.12898*.
20. Levente Kocsis, Csaba Szepesvári. 2006. [Bandit Based Monte-Carlo Planning](https://doi.org/10.1007/11871842_29). *Machine Learning: ECML 2006*, 282–293.
21. Zhi Zheng, Zhuoliang Xie, Zhenkun Wang, et al. 2025. [Monte Carlo Tree Search for Comprehensive Exploration in LLM-Based Automatic Heuristic Design](https://proceedings.mlr.press/v267/zheng25o.html). *Proceedings of the 42nd International Conference on Machine Learning* 267, 78338–78373.
22. Hui Wang, Yang Liu, Xiaoyu Zhang, et al. 2025a. [CogMCTS: A Novel Cognitive-Guided Monte Carlo Tree Search Framework for Iterative Heuristic Evolution with Large Language Models](https://arxiv.org/abs/2512.08609). *arXiv:2512.08609*.
23. Kezhao Lai, Yutao Lai, Hai-Lin Liu. 2026. [Beyond the Node: Clade-level Selection for Efficient MCTS in Automatic Heuristic Design](https://arxiv.org/abs/2602.00549). *arXiv:2602.00549*.
24. Bin Chen, Shouliang Zhu, Beidan Liu, et al. 2026. [A2DEPT: Large Language Model-Driven Automated Algorithm Design via Evolutionary Program Trees](https://arxiv.org/abs/2604.24043). *arXiv:2604.24043*.
25. Niki van Stein, Anna V. Kononova, Lars Kotthoff, et al. 2025. [Code Evolution Graphs: Understanding Large Language Model Driven Design of Algorithms](https://arxiv.org/abs/2503.16668). *arXiv:2503.16668*.
26. Jiale Zhao, Zimu Chen, Sirui Mao, et al. 2026a. [DGA$_2$D: Directed Graph-Guided Automated Algorithm Design with Large Language Models](https://arxiv.org/abs/2608.00700). *arXiv:2608.00700*.
27. Xuanzhou Chen, Taoli Cheng. 2026. [GAE: Graph-Augmented Evolution for Scientific Discovery via Reinforcement Optimization](https://arxiv.org/abs/2607.10127). *arXiv:2607.10127*.
28. Jenny Zhang, Shengran Hu, Cong Lu, et al. 2026. [Darwin Godel Machine: Open-Ended Evolution of Self-Improving Agents](https://arxiv.org/abs/2505.22954). *arXiv:2505.22954*.
29. Wenyi Wang, Piotr Piękos, Li Nanbo, et al. 2025b. [Huxley-Gödel Machine: Human-Level Coding Agent Development by an Approximation of the Optimal Self-Improving Machine](https://arxiv.org/abs/2510.21614). *arXiv:2510.21614*.
30. Oguzhan Gungordu, Siheng Xiong, Faramarz Fekri. 2026. [PathWise: Planning through World Model for Automated Heuristic Design via Self-Evolving LLMs](https://proceedings.mlr.press/v306/gungordu26a.html). *Proceedings of the 43rd International Conference on Machine Learning* 306, 37779–37816.
31. Leyi Zhao, Weijie Huang, Yitong Guo, et al. 2026b. [Large Language Model-Powered Evolutionary Code Optimization on a Phylogenetic Tree](https://arxiv.org/abs/2601.14523). *arXiv:2601.14523*.
32. Aman Madaan, Niket Tandon, Prakhar Gupta, et al. 2023. [Self-Refine: Iterative Refinement with Self-Feedback](https://proceedings.neurips.cc/paper_files/paper/2023/hash/91edff07232fb1b55a505a9e9f6c0ff3-Abstract-Conference.html). *Advances in Neural Information Processing Systems* 36, 46534–46594.
33. Noah Shinn, Federico Cassano, Ashwin Gopinath, et al. 2023. [Reflexion: language agents with verbal reinforcement learning](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html). *Advances in Neural Information Processing Systems* 36, 8634–8652.
34. Andrew Zhao, Daniel Huang, Quentin Xu, et al. 2024b. [ExpeL: LLM Agents Are Experiential Learners](https://ojs.aaai.org/index.php/AAAI/article/view/29936). *Proceedings of the AAAI Conference on Artificial Intelligence* 38, 19632–19642.
35. Haoran Ye, Jiarui Wang, Zhiguang Cao, et al. 2024. [ReEvo: Large Language Models as Hyper-Heuristics with Reflective Evolution](https://papers.nips.cc/paper/2024/hash/4ced59d480e07d290b6f29fc8798f195-Abstract-Conference.html). *Advances in Neural Information Processing Systems* 37, 43571–43608.
36. Yihong Liu, Junyi Li, Wayne Xin Zhao, et al. 2025. [Experience-Guided Reflective Co-Evolution of Prompts and Heuristics for Automatic Algorithm Design](https://arxiv.org/abs/2509.24509). *arXiv:2509.24509*.
37. Yang Wu, Junran Pan, Yifan Zhang, et al. 2026. [RefineEvo: Planning-Guided Heuristic Evolution with Bidirectional Experience](https://arxiv.org/abs/2607.11358). *arXiv:2607.11358*.
38. Zishang Qiu, Xinan Chen, Rong Qu, et al. 2026. [MeEvo: Metacognitive Evolution Combined with Natural Evolution for Automatic Heuristic Design](https://arxiv.org/abs/2606.14202). *arXiv:2606.14202*.
39. Jiachen Jiang, Tianyu Ding, Zhihui Zhu. 2026. [DeltaEvolve: Accelerating Scientific Discovery through Momentum-Driven Evolution](https://arxiv.org/abs/2602.02919). *arXiv:2602.02919*.
40. Fatemeh Haji, Javier Delarosa Quiros, Peyman Najafirad. 2026. [Memory-Guided Tree Search with Cross-Branch Knowledge Transfer for LLM Solver Synthesis](https://arxiv.org/abs/2605.17539). *arXiv:2605.17539*.
