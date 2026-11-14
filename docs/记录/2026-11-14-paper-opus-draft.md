# EvoTrace论文第一轮英文候选稿（Opus 5.5）

北京时间2026-11-14。本文是研究子线程交给父线程的第一轮候选稿，不是共享paper。依据：AGENTS.md、[反思压缩视角与论文叙事](2026-11-14-反思压缩视角与论文叙事.md)、[英文主稿](../paper.md)、[V2正式设计](../EvoTrace-v2设计.md)、[研究问题](../树状种群与反思简化.md)、`evotrace_v2/reflection.py`实际提示模板、[V2完整机制中期审查](2026-11-14-V2完整机制中期审查.md)、[V2最小修复与回放](2026-11-14-V2最小修复与回放.md)、修复后批次启动记录，以及`experiments_result/primary_runs.json`中基线的运行配置。

结构：第一部分说明主论证；第二部分是完整英文候选稿；第三部分是实质改进意见；第四部分是证据空缺清单。稿中`[[EDITOR: ...]]`标出需要父线程用实际数据填充或核对的位置。我没有填入任何未核实的数值，也没有把旧批次或定向回放当作最新完整搜索的收益。引用使用`[@key]`，key全部来自`docs/paper-references.bib`，本稿没有新增文献。

## 第一部分：主论证

一句话：**一个实现作为开发起点的价值，与它的经历对后续设计的参考价值不同；因此种群更新也应决定历史以什么形式继续存在。** EvoTrace把反思当作压缩搜索经历的过程，用树状种群把候选竞争和发展历史放在同一结构里，让同一次保留决策同时确定哪些程序留下、哪些历史区域被压缩、压缩后的经验由哪个保留位置承接。

论证链：

1. 问题：每次评价同时产生候选实现和设计经历，后续搜索两者都要用。
2. 矛盾：候选是否值得继续开发，与其经历是否值得记住，是两种价值判断；逐次历史的体量随搜索增长。
3. 洞察：种群更新本身是对搜索历史的选择性表示。退出种群的部分不必直接丢弃，可以经反思压缩成经验；保留结构决定压缩范围与承接位置。
4. 方法：带经验的树状种群（贡献1）＋动态反思简化（贡献2）。采样与三类算子视图是让更新后的结构继续参与搜索的支撑，不单列为贡献。
5. 检验：完整搜索看整体效果；固定树快照下的“无历史／原始历史／反思经验”三条件生成对照检验压缩是否带入有用材料；“同摘要异位置”对照检验承接位置的作用；过程统计说明压缩在搜索中实际是什么样子。

全文刻意不使用的表述：树或反思首创；因果知识或因果识别；无损压缩；每段历史只反思一次；树大小有硬上界；S等于失败、V等于进步；“相同保留规则”等于“相同树”。

## 第二部分：英文候选稿

---

# EvoTrace: Evolving Tree-Structured Populations through Reflective Compression

## Abstract

Large language models enable automated algorithm design through repeated program generation, evaluation, and revision. Each evaluation yields a candidate implementation and experience about a computational choice and its outcome. Population-based search decides which implementations remain available for development, but the value of an implementation as a starting point differs from the value of its history for later design. A superseded intermediate can explain how its descendant was formed. A regressed attempt can inform later revisions of its parent. We therefore treat each population update as a selective representation of search history, and reflection as the compression of the experience that leaves the population. We present EvoTrace, which maintains an experience-bearing tree-structured population through dynamic reflective simplification. After each evaluated addition, one retention decision selects quality representatives and recently developed algorithms and keeps the forks that connect them. The same decision fixes which history regions are compressed and where their experience is placed. Contracted development, together with side attempts made along it, is condensed onto the reconnected edge. Other removed branches are condensed into the nearest retained ancestor. When a carrier later leaves the population, its experience is compressed again with its new region. Parent sampling and operator-specific views then read the maintained tree for refinement, crossover, and exploration. [[EDITOR: one result sentence after the repaired full batch and the fixed-snapshot reading test are complete. Candidate form: "On four combinatorial optimization tasks, EvoTrace ... ; with the tree state held fixed, reflective experience ... relative to the raw history of the same regions at ... of its input length." Use only the claims that the data support.]]

## 1. Introduction

Large language models (LLMs) have made automated algorithm design (AAD) an iterative program search. A model writes an executable implementation, an evaluator measures it, and the search uses existing implementations to produce the next ones [@funsearch; @eoh; @elm]. Under a limited evaluation budget, every evaluation is costly. Each one produces two things. It produces a candidate that the search can develop further. It also produces experience: a computational choice was made relative to a known implementation, and the evaluator reported its effect. Both outputs matter for the next design. This paper studies how a search can keep and use the second output as the first one evolves.

Existing methods handle the two outputs through different parts of their search state. A population keeps the implementations that remain available for variation, whether as islands [@funsearch], a ranked pool with thought descriptions [@eoh], a diversity-aware pool [@hsevo; @partevo], or an archive with weighted parent sampling [@shinka]. Reflection and memory keep text derived from evaluated attempts, from short- and long-term reflection in ReEvo [@reevo] to positive and negative experience [@refineevo], shared design history [@meevo], and structured modification records [@deltaevolve]. Tree and lineage methods keep derivation relationships and use them for selection, generation, or pruning [@mctsahd; @cladeahd; @phyloevolve; @pathwise]. Together, these methods show that both candidate quality and developmental context can guide algorithm search.

These designs leave one decision implicit: what happens to the history of an implementation when the population stops keeping that implementation. If population selection is the only filter, the history leaves with the program. If history is kept in a separate memory, its scope and its relation to the current candidates are set by a different process. The generator must then reconnect the memory to the program it is revising. Keeping every intermediate implementation explicit avoids both losses, but the material that selection and generation must handle then grows with the search.

Our starting point is that these are two different values. An implementation is valuable as a starting point when further development from it is likely to pay off. Its experience is valuable when it helps explain or inform a later design. The two values often separate. An intermediate version can be superseded by its descendant, while the revisions and alternatives tried along the way still explain how the descendant was formed. An attempt can regress and leave the population, while its outcome remains relevant to the next revision of its parent. A population update therefore makes two decisions at once: which implementations stay executable, and in what form the rest of the history continues to exist. In this view, the current population is already a selective representation of search history, and reflection is a way to compress the part that leaves it. A tree-structured population makes this compression well defined. The retained nodes compete as candidates, and the edges between them record development. The retained structure then determines both the scope of each compression and the place where its result is kept.

We instantiate this idea in **EvoTrace**. EvoTrace keeps a tree of executable algorithms, primary-parent derivation edges, and condensed experience. Its population update, **dynamic reflective simplification**, runs after each evaluated addition. One retention decision keeps quality representatives, recently developed algorithms, and the forks that connect them. This decision induces both path contraction and side-branch removal on the same tree snapshot. Removed intermediate versions, together with side attempts made from them, are condensed onto the reconnected edge as *variation experience*. Other removed branches are condensed at the nearest retained ancestor as *selection experience*. When a later update removes a carrier, its experience joins the carrier's new region and is compressed again. Subsequent search samples a parent from the maintained tree, and refinement, crossover, and exploration each read the part of the tree that matches their design task.

Our contributions are as follows.

1. **An experience-bearing tree-structured population.** We organize executable candidates, derivation relationships, and condensed design experience in one evolving structure. The current population thus supplies both starting points for development and the history connected to them.
2. **Dynamic reflective simplification.** We couple each population update with compression of search experience. The retained structure jointly determines which history regions are summarized and which edge or node receives the result, so that removed attempts continue to inform subsequent generation.
3. **Empirical analysis.** [[EDITOR: state only what the experiments support. Planned components: same-budget comparison with [N] LLM-based AAD methods on four tasks with independent test sets; a fixed-snapshot test that compares no history, raw history, and reflective experience for the same regions; a placement test; and an analysis of how compression proceeds during search.]]

## 2. Related Work

### 2.1. LLM-Driven Algorithm Design and Population Maintenance

Algorithm design has been automated through genetic programming [@koza], hyper-heuristics [@hh], and algorithm configuration [@irace]; surveys organize this work by design space, search strategy, and evaluation [@survey_meta; @survey_aad]. LLMs added a new variation operator. Evolution through Large Models uses a code model for mutation [@elm], and language-model crossover recombines text genotypes [@lmcrossover]. FunSearch evolves programs with an island population [@funsearch]. EoH evolves heuristic descriptions together with code [@eoh]. LLaMEA generates and refines metaheuristics from execution feedback [@llamea], and AlphaEvolve extends evaluator-guided code evolution to scientific and computational problems [@alphaevolve]. Other work adapts the generator itself [@calm], decomposes design into stages [@evostage], or evolves algorithm structures in a bi-level loop [@beam].

Population maintenance decides which candidates remain available. HSEvo uses diversity measures with harmony search [@hsevo]; MEoH treats heuristic evolution as multi-objective [@meoh]; CDEoH keeps algorithm-category diversity [@cdeoh]; PartEvo builds niches [@partevo]. ShinkaEvolve samples parents by fitness and offspring count and assesses novelty before evaluation [@shinka]; EvoTrace adapts its weighted sampler. In these methods, the population answers the question of which implementations to keep. EvoTrace asks the population update to also answer how the history of the removed implementations is represented.

### 2.2. Trees, Lineages, and Pruning

Monte Carlo tree search allocates effort through node values and visit counts [@uct]. MCTS-AHD applies it to heuristic evolution and adds an action that reasons over the algorithms on a tree path [@mctsahd]. CogMCTS adds multi-round cognitive feedback [@cogmcts]. Clade-AHD aggregates descendant evaluations into clade-level beliefs and freezes low-performing clades [@cladeahd]. A2DEPT searches over evolutionary program trees [@a2dept]. The Darwin Gödel Machine keeps an archive of agent implementations [@dgm], and the Huxley-Gödel Machine estimates self-improvement potential from descendants [@hgm]. Graphs also represent program composition [@dga2d; @gae] or support retrospective analysis of code evolution [@codegraphs].

Two methods are closest in how they connect populations with derivations. PathWise combines an outer population with an inner entailment graph whose derivation metadata supports planning, generation, and reflection [@pathwise]. PhyloEvolve maintains a phylogenetic forest with an elite pool, trajectory-conditioned generation, and pruning at several structural scales [@phyloevolve]. EvoTrace shares the use of derivation structure. Its focus is the population update itself: one retained set induces both path contraction and branch removal, and each removed region is compressed into experience that stays on the edge or node inheriting its developmental relationship.

### 2.3. Reflection, Memory, and Compression of Experience

Self-Refine revises outputs with self-feedback [@selfrefine]. Reflexion keeps verbal feedback across episodes [@reflexion], and ExpeL extracts insights from collected experience [@expel]. In algorithm design, ReEvo combines short-term comparisons with an accumulated long-term reflection [@reevo]. EvoPH co-evolves prompts and heuristics with performance-guided reflection [@egr]. RefineEvo uses bidirectional experience with planning [@refineevo], and MeEvo alternates population evolution with reflection over shared history [@meevo]. DeltaEvolve represents successive changes as semantic deltas in a hierarchical context [@deltaevolve]. MEMOIR keeps branch-local execution memory and distills global summaries when a branch terminates [@memorytree].

These methods show that condensed experience can guide generation. They differ in what triggers condensation and where its result lives: at fixed intervals, at the end of a branch, or in a global pool. In EvoTrace, condensation is triggered by population simplification, its scope is a region removed by that simplification, and its result stays on the retained edge or node that inherits the region's developmental relationship. Generation then retrieves experience through the same tree that organizes the surviving programs.

## 3. Method

### 3.1. Setting

We consider AAD for a fixed task interface. A candidate program implements a designated function inside an enclosing solver. The evaluator runs the solver on a fixed search set and returns a fitness value or failure feedback; lower fitness is better. The search has a budget of $B$ program evaluations, which includes failed executions and repair evaluations. Model calls for generation, repair, and reflection are counted separately. When the budget is spent, the method returns the feasible archived program with the best search fitness, with earlier creation breaking ties; any held-out test happens after this choice.

EvoTrace keeps two records. The *attempt archive* stores every request, program, evaluation, and population update. The *tree* stores the state that the next search step selects from and reads. The archive is complete; the tree is a condensed and organized view of it. The rest of this section defines the tree and how it is updated.

### 3.2. An Experience-Bearing Tree-Structured Population

Let $T=(\mathcal N\cup\{r\},\mathcal E)$ be the current tree, where $r$ is a virtual root without code. An algorithm node $v\in\mathcal N$ stores its code, a factual description of its computation, its fitness $f(v)$ or failure feedback, and two usage records defined in Section 3.3. Each node has one primary parent. Crossover also records a reference algorithm, but the reference does not add a second structural edge. All retained feasible nodes, internal ones included, can be selected for further development.

The tree carries two kinds of experience (Table 1).

**Table 1. Where experience lives in the tree.**

| Location | Content | What it supports |
| --- | --- | --- |
| Selection experience $S_v$ on node $v$ (or on $r$) | Removed side attempts that branched from this location, with their outcomes | What has been tried from this implementation and how it turned out |
| Variation experience $V_{uv}$ on edge $(u,v)$ | Removed intermediate versions between retained endpoints $u$ and $v$, and side attempts made from those intermediates | How the current implementation was formed and which alternatives were tried along the way |

Both kinds hold whatever the removed history contains. Selection experience records improvements that were not retained, regressions, and failures alike. Variation experience describes a development stage; its net outcome can be an improvement, a tie, or a regression. An ordinary edge stands for one generation attempt; a contracted edge stands for a stage and is marked as such.

Initialization and Explore start a *route*; Refine and Crossover inherit the route of the primary parent. A route groups implementations by origin. Its identifier stays with descendants when the original entry is removed. Routes organize parent allocation and the scope of local context.

### 3.3. Dynamic Reflective Simplification

After initialization, every evaluated addition triggers one update. The update selects a retained set, partitions the removed nodes into regions, compresses each region by reflection, and reconnects the tree. All steps read the same pre-update snapshot.

**Quality representatives.** Let $\mathcal P\subseteq\mathcal N$ be the feasible nodes, ranked by fitness and then creation order. The rank quality is

$$
q(v)=1-\frac{\operatorname{rank}(v)}{\max(1,|\mathcal P|-1)},\qquad v\in\mathcal P, \tag{1}
$$

with rank zero for the best node. EvoTrace keeps the best node and then scans the candidates with $q(v)\geq 0.5$ from best to worst. A candidate is *superseded* if a strictly better descendant exists in the same route. It is *covered* if an already selected representative has identical normalized source or an abstract syntax tree (AST) distance below $0.08$; the threshold is $0.20$ when the two fitness values differ by at most $10^{-9}$. A candidate that is neither superseded nor covered joins the representative set $\mathcal R$. The AST distance is a multiset Jaccard distance over parent–child node-type pairs and traversal bigrams; it keeps operator types, called function names, and control structure, and ignores identifier names and literal values. Supersession uses derivation, not fitness alone. Only a same-route descendant can supersede an ancestor, so an Explore descendant never retires the implementation it started from.

**Recent development.** An evaluation clock $t$ counts actual program evaluations. Each feasible node has a last-use time $u(v)$, set at birth and refreshed when an evaluated Refine or Crossover attempt uses $v$ as primary parent. Explore and read-only reference use do not refresh it. With $p(v)$ the tree parent of $v$, the recent set is

$$
\mathcal U=\{v\in\mathcal P : t-\max\big(u(v),\,u(p(v))\big)<W\},\qquad W=40, \tag{2}
$$

where only $u(v)$ counts if $p(v)$ is the root or lies in another route. A parent under active development thus keeps its recent children eligible, and a cooled-down stage loses recency together.

**Necessary forks.** The retained set is

$$
\mathcal K=\{r\}\cup\mathcal R\cup\mathcal U\cup\operatorname{LCA}(\mathcal R\cup\mathcal U), \tag{3}
$$

where $\operatorname{LCA}(A)$ is the set of pairwise lowest common ancestors of nodes in $A$. Representatives are chosen before recency and fork closure are added, so a node kept only for recency or connectivity cannot block a better candidate from $\mathcal R$. Ranks and representatives are computed once per update. No fixed capacity appears in Eqs. (1)–(3). The tree size follows from quality differences, recent development, and branching structure, and it changes as the search proceeds.

**Partition into regions.** For each retained node, EvoTrace finds its nearest retained ancestor in the snapshot. The removed nodes on that connection form a *path region*. Every other removed node walks upward to its first ancestor that is not a removed side node. If that ancestor is a removed intermediate, the side node joins the intermediate's path region. Otherwise, the node joins the *side region* of its retained ancestor or of the root. Within one update, each removed node belongs to exactly one region, and all side branches with the same receiver form one region.

The partition lets one update shorten a path and remove its side branches together. Consider $A\to B\to C\to D$ with branches $B\to X$ and $A\to Y$, and suppose only $A$ and $D$ are retained (Figure 2). Nodes $B$ and $C$ form the path region of the new edge $(A,D)$, and $X$ joins it because it branched from $B$. Branch $Y$ forms the side region at $A$. Experience already stored on $B$, $C$, or the edges between them travels with its carrier into the path region. The existing $S_A$ stays at $A$ and merges with $Y$.

[[FIGURE 2: `docs/figures/evotrace/08-reflective-simplification.png`. Caption: "One retention decision fixes both the regions and their receivers. With A and D retained, B, C, and the side attempt X form one path region, compressed into $V_{AD}$. The separate branch Y is compressed into $S_A$, together with the earlier $S_A$. The earlier $S_B$ travels with B into $V_{AD}$."]]

**Reconnection.** Each retained node is reconnected to its nearest retained ancestor. A contracted edge whose endpoints lie in different routes is labeled Explore; otherwise it keeps the Crossover or Refine label of its history. The best current algorithm, primary-parent connectivity to the root, and unique ownership of removed nodes hold after every update.

### 3.4. Reflection as Compression

Each path region and each side region receives one reflection call. All calls in an update read the same snapshot, and new summaries are not reused within that update.

**Inputs.** A path reflection receives the two retained endpoint implementations and their net code difference; the removed intermediate implementations with their step differences and outcomes; the side attempts made from those intermediates; and the experience already stored on the removed edges and intermediate nodes. A side reflection receives the receiver's implementation when one exists; each removed attempt with its difference from its *actual* parent and its evaluation or error; and the experience already stored at the receiver and in the removed region. When the receiver is the root, initial implementations supply a task-level score with no parent comparison. The program computes every comparison label (improved, regressed, tied, or failed) from the archive; the model sees these labels instead of raw scores, and the program attaches endpoint scores to path experience. Generated design rationales are not supplied as evidence.

**Output.** Reflection returns one short paragraph of experience. The paragraph first states the observed outcome of the relevant comparisons, keeping net start-to-end outcomes apart from intermediate or side comparisons. It may then propose an explanation, introduced as one possible explanation and tied to the implementations tested. The prompt asks the model to treat jointly modified components as one combined change, to treat an Explore draft as a first implementation of an approach rather than local tuning, to let current measured comparisons override earlier summaries, and to avoid design instructions. Appendix C reproduces both templates.

**Recursion.** Path experience updates $V_{uv}$ on the reconnected edge; side experience updates $S_u$ at the receiver. When a later update removes a carrier, its stored experience enters the carrier's new region and is compressed again with the newly removed attempts. Experience therefore accumulates through repeated recompression as the tree changes. The tree holds the condensed account; the archive keeps the full code and evaluations behind it.

Compression here has a specific meaning. A structured, placed summary replaces a region of explicit history in the online population. EvoTrace does not define an information-theoretic fidelity target. How much useful material the summaries carry, and at what input cost, is an empirical question that Section 4.3 tests directly.

### 3.5. Searching from the Maintained Tree

The maintained tree serves the next step in two ways. Sampling decides where the next attempt starts, and an operator view decides which experience the generator reads.

**Parent sampling.** EvoTrace adapts the weighted sampler of ShinkaEvolve [@shinka]. Let $D(v)$ be the route of $v$. Every retained feasible node has weight

$$
w(v)=\sigma\!\left(10\,\frac{m-f(v)}{s}\right)\frac{1}{1+c(v)},\qquad
P(D)\propto\max_{u\in D}w(u),\qquad
P(v\mid D)=\frac{w(v)}{\sum_{u\in D}w(u)}, \tag{4}
$$

and the parent probability is $P(D(v))\,P(v\mid D(v))$. Here $m$ is the median fitness of the quality representatives, recomputed on the current tree before each sampling step with the rule of Section 3.3, and $s$ is their median absolute deviation (Appendix A). The count $c(v)$ includes every evaluated Refine or Crossover attempt with $v$ as primary parent, including failed and later removed attempts; Explore, reference use, reconnection, and inherited history do not add to it. A route competes through its best node weight, so keeping several versions of one route does not add up to a larger share for that route. Retention in Eq. (3) keeps a node eligible; Eq. (4) decides how often it is actually developed.

**Operators and views.** After selecting a parent, EvoTrace draws Refine, Crossover, or Explore with probabilities $0.45$, $0.25$, and $0.30$. Crossover chooses a reference among other feasible nodes with rank quality at least $0.30$, after excluding the parent and identical normalized source. It prefers parent–reference pairs not used before and, among them, the largest AST distance. Each operator reads the view of the tree that matches its design task (Table 2).

**Table 2. Operator views of the same tree.**

| Operator | What it reads | Design task |
| --- | --- | --- |
| Refine | Current code and fitness; the retained formation path within its route, with a brief account of the route entry and its edge experience; selection experience at the parent; its visible outgoing attempts | Revise the current computation |
| Crossover | Both implementations; within a route, the common ancestor and both diverging paths; across routes, each side's local development and source summary; the parent's visible outgoing attempts | Decide which computation to transfer and how to integrate it |
| Explore | For distinct routes, the entry idea, the idea of the most recently generated retained version, and the node and edge experience of that route, each labeled with its source; then one existing implementation | Construct an alternative whose main computation differs |

An ordinary edge is shown as a code difference relative to the actual parent, with its outcome. A contracted edge is shown as endpoint outcomes, the descendant's description, and its variation experience. If an evolution prompt exceeds the input budget, the view drops whole entries, choosing each time the entry that frees the most tokens. Current code, reference code, and the baselines needed to read differences are always kept. Fitting a view changes only that prompt; it does not change the tree, retention, or experience ownership. Each operator returns an analysis, a complete implementation, and a factual description of the computation the code performs.

### 3.6. Search Procedure

Algorithm 1 summarizes EvoTrace. Initialization generates five feasible programs sequentially; each initialization prompt shows the earlier initial ideas and scores and recent failures. Each attempt has at most one repair, and its final evaluated version becomes at most one node. Repair re-evaluation consumes budget. Attempts without any program evaluation do not enter the tree or advance the clock.

```text
Algorithm 1: EvoTrace
Input: task interface, evaluator, LLM, evaluation budget B
Create a virtual root, an empty tree, and an attempt archive
while budget remains and the stopping rule is not met:
    if fewer than five feasible initial programs exist:
        build an initialization prompt
    else:
        sample a parent (Eq. 4) and an operator; choose a reference if needed
        build and fit the operator's view of the tree
    generate and evaluate a program, with at most one repair
    if the attempt produced an evaluated program:
        attach its final version; update clock and usage records
        if initialization is complete:
            on one snapshot: select R and U; add LCA forks to obtain K
            partition removed nodes into path and side regions
            reflect once per region; reconnect; attach experience
    commit the attempt together with its population update
return the feasible archived program with the best search fitness
```

## 4. Experiments

We organize the evaluation around four questions.

- **RQ1.** Under the same evaluation budget, how does EvoTrace compare with existing LLM-based AAD methods?
- **RQ2.** With the tree held fixed, does reflective experience carry useful material from removed history into the next design, and at what input cost relative to the raw history of the same regions?
- **RQ3.** Does the location where experience is placed matter?
- **RQ4.** How does compression actually proceed during search?

RQ1 measures the whole method. RQ2 and RQ3 test the two parts of the central claim directly, with everything except the tested factor held constant. RQ4 describes the mechanism in operation.

### 4.1. Setup

**Tasks.** We use four tasks, each of which evolves one function inside a fixed solver (Table 3). Each task has 16 training instances for search and 50 disjoint test instances of the same scale. Instances are weighted equally. Generation prompts state the problem, the interface, and the meaning of the score; they do not state instance counts, sizes, data splits, time limits, or solver settings.

**Table 3. Tasks.**

| Task | Evolved function | Objective (minimized) |
| --- | --- | --- |
| TSP | `select_next_node` in step-by-step tour construction | Mean tour length |
| CVRP | Edge prior `heuristics` in ant colony optimization | Mean total route length |
| FSSP | `get_matrix_and_jobs` in guided local search | Mean makespan |
| OP | Edge prior `heuristics` in ant colony optimization | Negative mean collected reward |

**Evaluation.** A program fails if any instance raises an error, returns an invalid solution, exceeds memory, or exceeds the time limit; a failed evaluation has no score. [[EDITOR: state the exact time-limit regime used by EvoTrace and by each baseline. The repaired EvoTrace batch uses a shared 160-second limit over the 16 training instances; the primary baseline runs record `function_budget_scope = mean_instance`. Confirm whether these regimes are the same before writing a single sentence here. See gap G2.]] Each instance runs in an isolated single-threaded process with a 2 GiB memory limit.

**Budget and selection.** Every method receives 1,000 program evaluations per run, counting initialization, failures, and repair re-evaluations. Auxiliary model calls, such as reflection, are recorded separately. After search, each method fixes its best feasible training program without re-evaluation, and that program is evaluated once on the test set. We run three seeds (0, 1, 2) per method and task and report the mean.

**Model.** All methods use Qwen3.8-27B with native thinking disabled. [[EDITOR: the repaired EvoTrace batch records model `Qwen3.8-27B` with generation temperature 1.0 and reflection temperature 0.3; the primary baseline runs record `qwen3.8-27b-awq` with temperature 0.7. Confirm the serving model and resolve the temperature difference before describing the model condition as shared. See gap G2.]] EvoTrace generation allows 16,384 output tokens; reflection allows 1,536.

**Baselines.** [[EDITOR: choose from the primary baseline runs in `experiments_result/primary_runs.json`: FunSearch, EoH, ReEvo, HSEvo, MCTS-AHD, Clade-AHD, PathWise, ShinkaEvolve, DeltaEvolve, RefineEvo, OpenEvolve, CALM. Only methods whose four-task runs satisfy the confirmed common conditions should appear in Table 4.]]

**EvoTrace.** We use the default settings in Appendix A and the frozen source of batch `20261114_103407_evotrace_v2_repaired_co4`.

### 4.2. RQ1: Same-Budget Comparison

**Table 4. Test performance after 1,000 evaluations (mean over three seeds; lower is better).**

| Method | TSP | CVRP | FSSP | OP |
| --- | ---: | ---: | ---: | ---: |
| [[EDITOR: baselines]] | | | | |
| EvoTrace | [[EDITOR]] | [[EDITOR]] | [[EDITOR]] | [[EDITOR]] |

[[EDITOR: report per-seed values and valid-run counts in Appendix G. Add a figure of best training fitness against evaluations for each task. Write the paragraph only after the table is filled. The paragraph should say where EvoTrace leads, where it does not, and what the convergence curves show about when the difference appears. Do not use results from batch `20261113_233010_evotrace_v2_full_co4` or from EvoTrace V1 as results of this method.]]

**Cost.** [[EDITOR: report generation, repair, and reflection tokens per run from the call records; reflection's share of total input tokens; and wall-clock time. The pre-repair batch shows that reflection input is a substantial cost item, so this table is required, not optional.]]

### 4.3. RQ2: Does Compressed Experience Carry Useful Material?

Full-search comparisons cannot separate the effect of experience from the effect of the trajectories it changes. Once generation changes, the retained sets of two runs diverge, even under identical retention rules. We therefore hold the tree fixed.

**Design.** We take tree snapshots from completed EvoTrace runs at states where a recent update removed at least one region. For each snapshot we fix the parent, the operator, and, for Crossover, the reference. We then build three prompts that differ only in how the removed regions relevant to that operator's view are represented:

1. **No history.** The experience on the relevant nodes and edges is removed; current code, path endpoints, and all other material stay.
2. **Raw history.** The removed attempts of the same regions are shown in full form, with code differences and outcome labels, as reflection would have seen them.
3. **Reflective experience.** The tree's actual $S$ and $V$ text is shown, as in EvoTrace.

Each prompt is sampled [[EDITOR: K]] times under the same generation settings, and every program is evaluated once under the training condition. We record score, success rate, and actual input tokens. Cases where the raw history exceeds the input window are reported separately rather than truncated, so that the comparison object stays the same.

**Table 5. Fixed-snapshot generation (per task and operator).**

| Condition | Mean fitness | Success rate | Improvement over parent | Input tokens |
| --- | ---: | ---: | ---: | ---: |
| No history | [[EDITOR]] | | | |
| Raw history | [[EDITOR]] | | | |
| Reflective experience | [[EDITOR]] | | | |

[[EDITOR: the result paragraph should answer two questions. (a) Does reflective experience improve over no history? This tests whether removed attempts continue to contribute. (b) Where does it sit relative to raw history in quality and input length? This tests whether compression gives a useful trade-off. Report the pairing unit and a paired test. The existing 12-pair replay in `2026-11-14-V2最小修复与回放.md` compares old and new prompts, not these three conditions, and must not be used here.]]

### 4.4. RQ3: Does Placement Matter?

**Design.** On the same snapshots, we keep the experience text fixed and change only its location: (a) on its actual edge or node, as in EvoTrace; (b) moved into one global memory block that lists all summaries by recency, with their sources; (c) attached to an unrelated retained node of similar fitness. Each operator then reads the tree through its usual view. Placement (a) against (b) asks whether locating experience in the structure helps beyond having it available. Placement (a) against (c) asks whether the developmental relationship, rather than mere proximity to good code, makes it useful.

[[EDITOR: Table 6 and paragraph. If this experiment is not run, remove RQ3 and weaken the word "jointly determines ... where" in the abstract and contribution 2 to a design statement without an empirical claim.]]

### 4.5. RQ4: Compression during Search

[[EDITOR: compute the following from the repaired batch's committed population updates and generation prompts, per task:]]

- Tree size (algorithm nodes) and number of routes over evaluations, as a figure. The tree has no capacity limit, so the curve itself is a result.
- Number of updates that remove at least one node; numbers of path and side reflections.
- Distribution of newly removed nodes per region, and the share of regions with more than one new node over time.
- Number of times a carrier's existing experience is recompressed, i.e., the depth of recursion per stored summary.
- Exposure: the share of Refine, Crossover, and Explore prompts that contain $S$ text, $V$ text, or both.
- Provenance: for generated programs that set a new best training score, the share whose prompt contained experience originating from removed nodes, and the number of removed nodes behind that experience.

[[EDITOR: write the paragraph from these numbers. A reference point exists in the pre-repair batch `20261113_233010_evotrace_v2_full_co4` (frozen at 5,415 evaluations, same topology code; the 979-state replay in `v2_repairs_20261114_101007` produced identical retention and partition plans before and after the repair). There, 2,621 updates produced 1,180 path and 1,760 side reflections; 1,456 of the 1,760 side reflections and 612 of the 1,180 path reflections added only one newly removed node; 797 side nodes joined a path region in the same update; S or V text appeared in most generation prompts. These figures describe that batch only. Use them in the paper only if the repaired batch is unavailable, and then label them by batch.]]

### 4.6. Ablations

We run the full search with one component changed at a time, under the same budget and seeds.

- **No experience reading.** Reflection and structure stay; generation views omit $S$ and $V$ text.
- **No reflection.** The structural update stays; each region receives the deterministic fallback summary instead of a model reflection.
- **Capacity-triggered simplification.** The tree is simplified only when it reaches a fixed node count (EvoTrace V1 rules: trigger at 64, simplify to 32).

[[EDITOR: Table 7. Full-search ablations carry the same caveat as RQ1: identical rules do not produce identical trees. Interpret them together with RQ2–RQ3, not as standalone attribution. If budget allows only one ablation, "No experience reading" is the most informative, because it isolates the reading path while keeping trajectory-level structure.]]

### 4.7. Case Study

[[EDITOR: pick one case from the repaired batch in which (i) a region was compressed, (ii) its experience was read by a later Refine or Crossover prompt, and (iii) the resulting program changed the computation in a way that the experience refers to. Show the region, the experience text, the relevant part of the generator's analysis, and the code change. A pre-repair example exists (CVRP program 103 in the 11:47 dynamic batch, which read path experience and combined capacity damping removal with savings scaling, 8.82966 → 8.62530), but it comes from an older version and may only be used as an illustration labeled by batch.]]

## 5. Discussion

**What the view explains.** Treating the population update as a decision about history representation explains why two common design choices sit at the ends of one axis. Keeping only the selected programs discards the experience of removed attempts. Keeping every attempt explicit preserves it but leaves its organization to the generator. EvoTrace sits between them: the retained structure decides what stays executable, and reflection turns the rest into experience placed where its developmental relationship lies. [[EDITOR: connect this paragraph to RQ2 and RQ3 once their results are known.]]

**Granularity of compression.** Because every evaluated addition can trigger an update, many regions are small, and a stored summary is often recompressed with one new attempt at a time. Recursive recompression lets a summary absorb history gradually, but it also gives repeated chances to overgeneralize from one comparison. [[EDITOR: use the RQ4 distribution to state how often regions are multi-node and how this changes over the run. If multi-node regions become more frequent later in search, say so; if not, state it plainly and discuss it here.]]

**Reliability of reflection.** Reflection condenses evidence into text, and text can misstate it. Supplying program-computed comparison labels instead of raw scores removed direction errors in our targeted checks, but some summaries still misdescribe the computation that the code performs. [[EDITOR: replace the targeted-case statement with a random-sample audit of the repaired batch (gap G4): the share of summaries whose stated comparison direction matches the archive, and the share whose description of the computation matches the code.]]

**Retention and opportunity.** Retention keeps an implementation eligible; the sampler decides how often it is developed. With a steep quality term, a retained but weak draft can stay eligible for the whole recency window and still receive almost no development. EvoTrace's experience path does not depend on such drafts being developed: their outcomes still enter selection experience when they leave. [[EDITOR: support with the repaired batch's expected-selection statistics, or remove.]]

**Limitations.** Experience in EvoTrace is descriptive text derived from evaluated comparisons; it is not a verified causal model of the solver, and combined changes do not isolate the effect of individual components. The tree size is governed by the retention rules rather than a fixed cap, so memory and context size depend on the task's search dynamics. Our evaluation covers four combinatorial optimization tasks with one model family. [[EDITOR: adjust after results.]]

## 6. Conclusion

We argued that an implementation's value as a starting point and the value of its experience are different, and that a population update should therefore also decide how history is represented. EvoTrace realizes this view with an experience-bearing tree-structured population and dynamic reflective simplification. One retention decision determines which implementations stay executable, which history regions are compressed, and where the resulting experience is kept; later updates recompress experience as the tree changes, and operators read it through task-specific views. [[EDITOR: one or two sentences of findings from RQ1–RQ3.]] The same idea applies wherever a search keeps a population and generates from history: the update that selects candidates can also be the update that organizes what the search remembers.

## Appendix A. Default Settings

| Component | Setting |
| --- | --- |
| Initialization; budget | Five feasible initial programs; 1,000 program evaluations |
| Recent retention | $t-\max(u(v),u(p(v)))<40$, with the parent term only for a same-route tree parent |
| Quality representatives | Rank quality $\geq 0.50$; best-first scan; excludes candidates superseded by a strictly better same-route descendant or covered by a representative |
| Coverage | Identical normalized source, or AST distance $<0.08$; $<0.20$ when fitness differs by at most $10^{-9}$ |
| Fork closure | Pairwise LCAs; no forced retention of initial roots or Explore entries |
| Parent sampling | Sigmoid coefficient 10; discount $1/(1+c(v))$; route maximum, within-route normalization |
| Operator prior | Refine 0.45; Crossover 0.25; Explore 0.30 |
| Crossover reference | Rank quality $\geq 0.30$ after excluding the parent and identical source; prefer unused pairs, then largest AST distance |
| Explore route filter | Distinct normalized entry source and AST distance $\geq 0.12$; ordered by latest retained version |
| Evolution prompt input | 30,000 tokens; whole-entry reduction when exceeded |
| Generation | Temperature 1.0; up to 16,384 output tokens |
| Reflection | Temperature 0.3; up to 1,536 output tokens; one paragraph of about 120 words, shorter for sparse evidence |

The scale $s$ in Eq. (4) is the median absolute deviation of representative fitness, floored at $10^{-6}$. If it is at most $10^{-9}$, EvoTrace uses the median of the absolute deviations larger than $10^{-9}$, with the same floor. Log weights are normalized at both levels; a probability that underflows is represented by the smallest positive float. A reference pool with one candidate assigns it rank quality one. Input tokens are counted with the serving tokenizer when available and otherwise approximated by UTF-8 bytes. A reflection call or parse failure falls back to a deterministic summary of at most 960 characters, which keeps the structural update. The search stops after ten consecutive attempts without a program evaluation.

## Appendix B. Partition Procedure

```text
Input: snapshot tree T, retained set K
removed ← N \ K
for each v in K \ {r}:
    a ← nearest ancestor of v in K
    path(a, v) ← removed nodes strictly between a and v
    if path(a, v) is nonempty: create path region for new edge (a, v)
side ← removed nodes not in any path region
for each x in side:
    y ← first ancestor of x that is not in side
    if y is a removed intermediate: add x to the path region containing y
    else: add x to the side region of y          # y in K, or y = r
each region also takes the experience stored on its removed nodes and edges
```

Each removed node enters exactly one region per update. Old summaries keep their sources when they enter a region, and no material is assigned to two regions.

## Appendix C. Reflection Prompts

Both prompts begin with a task header (problem, interface, score direction). The evidence section lists implementations, step differences, program-computed comparison labels, errors, and earlier experience. The task and output sections are reproduced from `evotrace_v2/reflection.py`.

**Path reflection.**

> Summarize the development from retained start to retained endpoint. The program has computed each comparison label. Use these labels; do not reconstruct scores or numerical differences. Retention does not imply improvement. Earlier prose summaries may be wrong; current measured comparisons take precedence.
>
> Return one JSON object with an "experience" string containing a short factual account: 1. Start with the net start-to-end outcome and name the computational changes together. Do not substitute a last-step gain for the net outcome. 2. Include an intermediate or side comparison only when informative. Name both compared computations and preserve that comparison's measured outcome label. Never say "improved" without specifying which starting implementation it improved upon. 3. If the code suggests an explanation, begin it with "One possible explanation for this comparison is". Keep it specific to the implementations tested. Combined changes do not isolate individual contributions. If the cause cannot be distinguished, say so; no explanation is required. Explore is an initial implementation of a new computational approach, not local tuning of its parent. Its result concerns this draft, not the potential of its entire approach. An unchanged score does not prove two programs equivalent or numerically safe. Successful evaluation does not show that faster code bought extra solver iterations. Do not conclude with general advice, requirements, or future design instructions. Normally use no more than 120 words; sparse evidence warrants fewer.

**Side reflection.**

> Summarize the experiments associated with [receiver]. Compare each trial only to its actual parent in the evidence, never to an independent champion or a different trial's parent. The program has computed each comparison label; do not reconstruct scores or numerical differences. Initial implementations without an algorithm parent have a measured score but no parent-relative improvement. Earlier prose summaries may be wrong; current measured comparisons take precedence.
>
> Return one JSON object with an "experience" string containing a short factual account: 1. Name the starting computation, the tested changes together, and the observed outcome using the measured comparison label. Explore is an initial implementation of a new computational approach, not local tuning of its parent. Its result concerns this draft, not the potential of its entire approach. 2. If the code suggests an explanation, begin it with "One possible explanation for this comparison is". Keep it specific to the implementations tested. Combined changes do not isolate individual contributions. If the cause cannot be distinguished, say so; no explanation is required. Preserve useful earlier observations only when their comparison and conditions are clear. An unchanged score does not prove two programs equivalent or numerically safe. Successful evaluation does not show that faster code bought extra solver iterations. Do not conclude with general advice, requirements, or future design instructions. Normally use no more than 120 words; sparse evidence warrants fewer.

[[EDITOR: add one generation prompt per operator as an example, taken from the repaired batch's call records.]]

## Appendix D. Recovery and Accounting

Each attempt is committed together with its population update, including reflection outputs. Restoring a run replays committed updates without repeating reflection calls. The archive keeps every request, response, program, evaluation, and region input, so every summary in the tree can be traced to its evidence. Token usage is taken from the serving interface; calls without reported usage are counted separately.

## Appendix G. Additional Results

[[EDITOR: per-seed training and test values for every method and task; valid-run counts; convergence curves; RQ2 per-snapshot results; reflection audit table; cost table.]]

## References

[[EDITOR: render from `docs/paper-references.bib`. Keys used in this draft: koza, hh, irace, survey_meta, survey_aad, elm, lmcrossover, funsearch, eoh, llamea, alphaevolve, hsevo, meoh, cdeoh, partevo, shinka, calm, evostage, beam, uct, mctsahd, cogmcts, cladeahd, a2dept, codegraphs, dga2d, gae, dgm, hgm, pathwise, phyloevolve, selfrefine, reflexion, expel, reevo, egr, refineevo, meevo, deltaevolve, memorytree. All 40 keys are used; no new reference was added.]]

---

## 第三部分：实质改进意见

### 意见1：把固定快照三条件对照（RQ2）作为核心证据，并补放置对照（RQ3）

**判断：** 完整搜索对比不能检验本文洞察。洞察说的是“被移除历史经压缩后继续有用，且承接位置有作用”。同预算整体胜负同时混入采样、算子、轨迹分化，规则相同而树不同，无法把收益归给经验。固定树快照下，只改变同一区域历史的表示（无／原始／反思），才直接对应洞察的两半：反思相对无历史检验“经历继续参与”，反思相对原始历史检验“压缩是否形成有效的质量—读取成本取舍”。

**影响：** 如果只有RQ1，摘要和贡献2中“保留结构决定经验承接位置，使被移除尝试继续参与后续生成”只能作为设计描述，不能作为实证主张。RQ3（同文本、异位置）是贡献2里“位置”一半唯一的直接检验。RQ3若不做，正文要相应把“where”降为设计陈述，我已在稿中标出。

**成本：** 不需要新完整搜索。从修复后批次取快照，每任务×算子若干状态，每状态三条件各采样K次，各做一次训练评价即可。建议按“同快照同算子”配对，报告配对检验。

### 意见2：如实呈现压缩粒度，把“压缩”定义为递归增量压缩

**判断：** 旧完整批次（与修复后拓扑代码相同）中，1760次旁支反思有1456次只新增一个移除节点，1180次路径反思有612次只新增一个。也就是说，压缩在多数时候是“已有摘要＋一次新尝试”的重写，不是一次对多节点区域的集中凝练。这是机制的实际形态，不是缺陷，但它改变论文该怎样描述压缩：保留结构确定的“范围”常常是单节点，经验的质量主要靠递归累积，而不是靠单次区域概括。

**影响：** 若论文把压缩写成“把一段历史区域整体凝练”，读者会预期多节点区域主导，而数据会显示相反。稿中第3.4节用“repeated recompression”描述，第5节专设“Granularity”段，并在RQ4要求报告区域大小分布随时间的变化。重复改写也是过度概括的来源（中期审查中跨分支相互冲突的“普遍优于”就是例子）；RQ4的递归深度统计能说明这一点。

### 意见3：用随机抽样的反思保真度审计和经验来源追溯，替代定向案例

**判断：** 现有反思证据来自14个定向案例（含5个历史错误案例），不是随机样本。修复后方向反转在这14例中消失，但仍有3例代码解释或归因错误。论文把反思称为“压缩”，读者自然要问压缩保留了什么。两个可在已有日志上完成、无需新实验的统计最能回答：

1. **保真度：** 从修复后批次的S/V中随机抽样，对照档案核对：（a）陈述的比较方向与程序标签一致的比例；（b）对计算的描述与代码一致的比例。（a）可大部分程序化；（b）需人工或强模型复核。
2. **来源追溯：** 对每个刷新训练最好分数的生成，追溯其提示中经验文本的来源，统计来自已移除节点的比例和对应的移除节点数。这直接量化“被移除尝试继续参与后续生成”，是贡献2最便宜的过程证据。

### 意见4（需父线程优先核对）：基线与EvoTrace的共同条件不一致

**判断：** `primary_runs.json`中基线主运行记录`qwen3.8-27b-awq`、温度0.7、`function_budget_scope = mean_instance`；修复后EvoTrace记录`Qwen3.8-27B`、生成温度1.0，启动记录写“训练16实例共享160秒”。模型名字符串、采样温度和时限口径三项都可能不同。

**影响：** 这决定RQ1能否写成“同模型、同评价条件”比较。三项中任何一项确实不同，Table 4都需要重跑一侧或改变比较口径，不能靠措辞处理。这应在写RQ1正文之前核对。

## 第四部分：证据空缺清单

| 编号 | 空缺 | 影响的论文位置 | 现状 | 补齐方式 |
| --- | --- | --- | --- | --- |
| G1 | 修复后V2完整搜索结果与独立测试 | 摘要结果句、贡献3、RQ1、Table 4、结论 | 批次`20261114_103407`运行中；本地同步时各路约58–90/1000评价 | 等待完成；按三种子均值与逐路值填表 |
| G2 | 基线共同条件：模型字符串、温度（0.7 vs 1.0）、时限口径（mean_instance vs 共享160秒） | 4.1、RQ1全部比较句 | 已在运行配置中观察到差异，未确认是否实际不同 | 核对各批run_config与评价器；不同则重跑或调整口径 |
| G3 | 固定快照三条件生成对照 | RQ2、Table 5、摘要第二个结果分句 | 未做；已有12对回放比较的是新旧提示，不能替代 | 从修复后批次取快照，配对采样评价 |
| G4 | 随机抽样反思保真度审计 | 3.4末段、第5节Reliability | 只有14个定向案例 | 随机抽样S/V，程序核对方向，人工核对计算描述 |
| G5 | 放置对照（同文本异位置） | RQ3、摘要与贡献2中“where”的实证地位 | 未做 | 同RQ2快照上改变经验位置；不做则降为设计陈述 |
| G6 | 压缩过程统计（树规模曲线、区域大小分布、递归深度、曝光、来源追溯） | RQ4、第5节Granularity | 仅有修复前批次数字 | 用修复后批次已提交更新与生成提示重算 |
| G7 | 完整搜索消融（无经验读取、无反思、容量触发） | 4.6、Table 7 | V1批次是旧条件，不能直接充当消融 | 至少运行“无经验读取”一项 |
| G8 | 成本表（生成、修复、反思tokens与墙钟） | 4.2 Cost | 修复前批次显示反思输入是显著成本项 | 由修复后批次调用记录汇总 |
| G9 | 案例研究 | 4.7 | 只有旧批次案例 | 从修复后批次选取“区域压缩→经验被读取→计算改变”完整链条 |
| G10 | 保留与实际开发机会的统计 | 第5节Retention段 | 仅修复前批次统计（848/959未开发初稿累计预期选择<0.01）；修复后质量参照已变 | 修复后批次重算，或删去该段 |

本次没有修改共享paper、源码或其他文件，没有启动实验，没有委派子代理。
