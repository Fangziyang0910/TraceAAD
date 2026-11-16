# EvoTrace: Experience-Bearing Tree-Structured Populations for Automated Algorithm Design

## Abstract

Population competition and derivation trees offer complementary ways to organize automated algorithm design: competition allocates evaluations among candidates, while trees connect successive modifications and branching attempts. We introduce EvoTrace, which combines these roles in an experience-bearing tree-structured population and treats reflection as compression of search experience. Nodes represent evaluated algorithms, and primary-parent edges record their derivations. Reflective tree simplification retains an active working set, the best individual, and the forks connecting them. It partitions the remaining history into path and side-branch regions, then simplifies and reflects on regions that reach a minimum size. Path experience is stored on reconnected edges, and side-branch experience on retained ancestors. Subsequent refinement, crossover, and exploration read path-, branch-, and population-scale views of this same tree. The resulting search process connects population growth and competition with the continual condensation and use of algorithm-design experience.

## 1. Introduction

Large language models (LLMs) support automated algorithm design (AAD) through repeated program generation, evaluation, and revision [@funsearch; @eoh; @elm]. Under a finite evaluation budget, search must choose which candidates to develop and how to use the outcomes of earlier attempts. Population competition directs evaluations toward candidates, while a derivation tree organizes how algorithms are modified and how search branches develop.

Prior work provides population-based evolution [@funsearch; @eoh; @partevo; @shinka], tree- and graph-organized search [@mctsahd; @cladeahd; @pathwise; @phyloevolve], and reflection or memory for subsequent generation [@reevo; @refineevo; @deltaevolve; @memorytree]. These approaches motivate a common design question: **how can candidate competition, tree maintenance, and experience use operate within one evolving search state?** As new programs and attempts accumulate, the representation must support continued search while organizing relevant history for generation.

Our approach combines population competition with derivation structure in an **experience-bearing tree-structured population**. Nodes are competing algorithm individuals; primary-parent edges connect their derivations. We treat **reflection as compression of search experience** during tree simplification. Related attempts are condensed together, and their experience is stored on the nodes and edges through which subsequent generation reads that history.

We implement this approach in **EvoTrace**. After an evaluated insertion, reflective tree simplification retains recently created, directly developed, or crossover-referenced individuals, the best individual, and their lowest common ancestors (LCAs). The retained skeleton partitions other nodes into path and side-branch regions. Only regions reaching a minimum size are simplified and reflected on. Path regions, including their attached side attempts, condense onto reconnected edges; other side branches condense onto retained ancestors. Existing notes participate in later reflections as the tree continues to change.

Search uses the same tree for selection and generation. All retained distinct feasible individuals compete globally according to fitness rank and lineage development effort. **Refine** reads a formation path and local attempts; **Crossover** reads two branches after their common ancestor; **Explore** surveys population designs and experience. Tree maintenance and these views connect the compression of earlier attempts to subsequent algorithm design.

The method has three connected components:

1. **Experience-bearing tree-structured population.** A common representation for competing algorithms, derivation relationships, and node and edge experience.
2. **Reflective tree simplification.** Working-set retention, fork preservation, and batched region reflection jointly maintain the population and condense related attempts.
3. **Search over the maintained tree.** Global parent sampling and operator-specific tree views use the algorithms and experience in this representation.

## 2. Related Work

**Evolutionary populations and parent allocation.** Genetic programming evolves executable structures [@koza], while hyper-heuristics and automated configuration search over heuristic components and solver parameters [@hh; @irace]. LLM-based AAD extends variation to code synthesis and revision [@survey_aad; @elm; @lmcrossover]. FunSearch pairs program generation with an island model [@funsearch], EoH co-evolves natural-language thoughts and code [@eoh], and LLaMEA and AlphaEvolve advance evaluator-guided algorithm evolution [@llamea; @alphaevolve]. Subsequent work improves diversity and parent allocation through harmony search [@hsevo], niching [@partevo], and fitness-and-usage-weighted sampling [@shinka]. EvoTrace retains global competition across feasible individuals, calibrates quality selection via a rank half-life independent of population size, accounts for development effort through the best feasible node on each candidate's retained ancestry path, and embeds individuals in a derivation tree rather than a flat pool.

**Lineage trees and graphs in algorithm search.** MCTS-AHD organizes heuristics in a Monte Carlo search tree [@uct] and conditions expansion on root-to-leaf paths [@mctsahd], while Clade-AHD aggregates descendant evaluations into clade-level beliefs [@cladeahd]. Related frameworks represent search history through evolutionary program trees or directed code graphs [@a2dept; @codegraphs; @dga2d]. PathWise couples an outer population with an inner entailment graph for coordinated planning and reflection [@pathwise]. PhyloEvolve maintains a phylogenetic forest alongside a separate elite pool, pruning stalled subtrees and periodically summarizing trajectories for similarity-based retrieval [@phyloevolve]. In EvoTrace, the population itself is a tree. Working-set retention and region-size thresholds govern path contraction and branch pruning; reflection condenses each selected region onto a retained edge or ancestor.

**Verbal reflection and experience memory.** Iterative self-feedback [@selfrefine] and verbal memory across trials [@reflexion; @expel] motivate language-level learning in AAD. ReEvo combines short-term pairwise comparisons with long-term reflections [@reevo], RefineEvo integrates planning with bidirectional positive and negative experience [@refineevo], MeEvo alternates population evolution with shared-history reflection [@meevo], and BEAM adapts bi-level memory during heuristic search [@beam]. Closer to lineage-indexed memory, DeltaEvolve tracks semantic code deltas in a hierarchical context store [@deltaevolve], and MEMOIR distills branch-local memory upon branch termination [@memorytree]. EvoTrace binds both the scope and the storage location of reflection to simplification of the population tree, preserving provenance-linked comparison catalogs as summaries pass recursively across generations.

## 3. Method

### 3.1. Problem Setup and Overview

We consider AAD for a fixed solver interface. A candidate program implements a target Python function invoked by an enclosing optimization algorithm. Evaluating a candidate on a fixed training instance set returns either a scalar fitness $f(v)$ (where lower values indicate better performance) or execution failure feedback (exception tracebacks, invalid outputs, or timeout diagnostics). Search proceeds under a budget of $B$ actual program evaluations, counting initialization, failed executions, and repair re-evaluations; LLM calls for generation, format or runtime repair, and reflection are logged separately.

EvoTrace maintains an online tree-structured population $T$ alongside an append-only attempt archive. The tree provides selectable parent implementations and structured context for generation, while the archive records every generated program, evaluation result, LLM call, and simplification update. In each post-initialization step, EvoTrace samples a parent and operator from $T$, assembles an operator-specific tree view, generates and evaluates a candidate (with at most one repair attempt), attaches the final evaluated version to $T$, and executes dynamic reflective simplification. At budget exhaustion, EvoTrace returns the feasible archived program with the best training fitness (breaking ties by earlier creation order) for single-shot evaluation on held-out test instances.

### 3.2. Experience-Bearing Tree-Structured Population

Let $T=(\mathcal N\cup\{r\},\mathcal E)$ denote the online population tree rooted at a virtual root $r$ that holds no executable code. Every node $v\in\mathcal N$ represents an evaluated candidate individual and stores: (i) executable source code and a concise factual description of its computation (`design`), (ii) measured fitness $f(v)$ or failure feedback, (iii) direct development effort $d(v)$ (`development_evaluations`), counting actual evaluations spent on attempts with $v$ as primary parent, (iv) cumulative historical `subtree_evaluations` across $v$ and its descendants, (v) last-activity clock $u(v)$ (`last_used`), and (vi) node **selection experience** $S_v$ paired with its supporting comparison catalog. The virtual root $r$ similarly stores root selection experience $S_r$ condensed from culled initial programs.

Each node $v\in\mathcal N$ has a single incoming directed edge $(p(v),v)\in\mathcal E$ from its primary parent $p(v)\in\mathcal N\cup\{r\}$. Initialization attaches initial candidates to $r$; `Refine`, `Crossover`, and `Explore` attach new candidates to their sampled primary parent. `Crossover` additionally records a read-only `reference_id` as metadata without introducing a second structural edge, preserving the tree invariant. Each edge stores its operator label, a boolean contraction flag indicating whether intermediate ancestors were simplified along that link, and edge **variation experience** $V_{p(v),v}$ paired with its supporting comparison catalog.

**Effort and activity accounting.** The global clock $t$ counts total program evaluations consumed so far. When an attempt from primary parent $p$ executes $e\in\{1,2\}$ program evaluations (initial evaluation plus optional repair re-evaluation), $d(p)$ increases by $e$, $u(p)$ updates to $t$, and `subtree_evaluations` increases by $e$ at $p$ and all current ancestors of $p$. This accounting applies uniformly across `Refine`, `Crossover`, and `Explore`: `Explore` specifies a generative shift toward a new core computation, while its offspring competes as an ordinary individual in $T$. Selecting an individual as a crossover reference refreshes its activity clock $u$ without increasing $d$. Merely displaying an individual in a prompt changes neither count nor clock. The newly attached child $v$ begins with $d(v)=0$, `subtree_evaluations` equal to $0$, and $u(v)=t$.

### 3.3. Reflective Tree Simplification

After five feasible initial programs have been obtained, maintenance runs after each evaluated insertion. All decisions read the same pre-update snapshot. The two maintenance parameters are the activity window $W$ (`active_evaluations`) and minimum region size $m$ (`region_size`).

**Active working set.** Let $\mathcal P$ contain feasible nodes in $T$. Collapse identical normalized-source-and-fitness pairs to their earliest retained record, obtaining $\mathcal D=\operatorname{distinct}(\mathcal P)$. Let $v^\star$ be the best feasible individual, breaking fitness ties by earliest ID. The active set is

$$
\mathcal A=\{v\in\mathcal D:t-u(v)<W\}\cup\{v^\star\},
\qquad W=40,
\tag{1}
$$

where $u(v)$ is refreshed at birth, direct development, or selection as a crossover reference. This rule preserves recently used individuals without a fixed quality quota. Exact duplicates do not enter $\mathcal D$ and are excluded from parent selection and the reference pool.

**Fork preservation and region partitioning.** Preserve the virtual root and pairwise LCAs of active individuals:

$$
\mathcal H=\{r\}\cup\mathcal A\cup\operatorname{LCA}(\mathcal A).
\tag{2}
$$

For each non-root skeleton node, follow its original parents to the nearest skeleton ancestor. Any intervening nodes form a path region together with their side branches outside $\mathcal H$. Other nodes outside $\mathcal H$ are grouped by their nearest skeleton ancestor into side-branch regions. Let $\mathcal C$ be this disjoint collection of regions. Skeleton endpoints provide reflection context but are not counted in a region's size.

**Batched simplification.** Simplify only whole regions reaching the size threshold:

$$
\mathcal R=\bigcup_{C\in\mathcal C:\,|C|\geq m}C,
\qquad
\mathcal K=(\mathcal N\cup\{r\})\setminus\mathcal R,
\qquad m=4.
\tag{3}
$$

Smaller regions remain unchanged and eligible feasible individuals can become active again through use. For each selected path region, reconnect its retained endpoints and write the reflected experience to the new edge's $V$. For each selected side-branch region, write the reflected experience to its skeleton ancestor's $S$. Prior notes and comparison catalogs participate in this update.

All selected regions are removed together. A reconnected edge is marked as contracted, with `operator` and `reference_id` cleared because it now represents multiple steps. Surviving individuals retain their original direct effort, subtree effort, and activity clocks. The fixed skeleton keeps regions separate: applying maintenance again without further search produces no additional simplification (Appendix B).

### 3.4. Region Reflection and Provenance

EvoTrace issues one structured reflection call per path-contraction region and one per grouped side-branch region, reading exclusively from the pre-simplification snapshot; updates with no removed nodes invoke zero reflection calls.

**Evidence assembly.** A path reflection prompt presents the retained start and endpoint code (when $a(v)\neq r$), their net code diff and net score change, intermediate implementations and outcomes along the chain, any same-update side attempts attached to those intermediates, and inherited $S$ or $V$ notes carried by removed elements. A side-branch reflection prompt presents the retained receiver's code (when $u\neq r$), each removed trial's code diff relative to its actual parent, measured scores or compact runtime error diagnostics, and the receiver's existing $S_u$. To ground local outcomes, both prompts include a compact `[Related Results]` reference naming the best other evaluated sibling from each shared starting node, together with a program-computed `[Comparison Catalog]` listing every original `parent_id->child_id` transition, operator, measured score change (`improved`, `tied`, `regressed`, or `failed`), and error summary.

**Research-note condensation and provenance binding.** Each reflection call returns a strict JSON object `{"experience": "..."}` containing a self-contained paragraph of roughly 120 words that records the computational ideas tested, informative outcomes, and relevant carried-over experience without referencing transient trial numbers. Interpretations remain tentative and optional, and side-branch prompts explicitly note that population removal reflects capacity management rather than disproof of an algorithmic idea. Alongside the generated text, EvoTrace automatically binds the region's complete `Comparison Catalog` metadata (`selection_observations` or `variation_observations`) to the target node or edge. When a carrier node or edge is simplified in a subsequent update, its note is prefixed with its supporting `comparison_id` keys so original comparison endpoints remain exact across multi-step recursive handoffs. If a reflection request or JSON parse fails, a deterministic 960-character summary rendered from the comparison catalog and prior notes preserves the structural update without loss of provenance.

### 3.5. Parent Allocation and Task-Aligned Generation

**Global rank-decay and lineage-effort sampling.** All retained distinct feasible individuals in $\mathcal D$, including internal fork nodes, compete in one global distribution. Order $\mathcal D$ by fitness and assign each tied block its midpoint zero-based rank $\bar r(v)$. With $M=|\mathcal D|$, normalized rank quality is $q(v)=1-\bar r(v)/(M-1)$ for $M>1$, and $q(v)=0.5$ for $M=1$.

Let $b(v)$ be the best feasible node on the retained root-to-$v$ path, including $v$ itself and breaking fitness ties by earliest ID. Let $s(b(v))$ denote that node's historical subtree evaluation count. Parent weights and selection probabilities are

$$
w(v)=\frac{2^{-\bar r(v)/H}}{1+s(b(v))},
\qquad
P(v)=\frac{w(v)}{\sum_{u\in\mathcal D}w(u)},
\qquad v\in\mathcal D,
\tag{4}
$$

Here $H=10$ is the rank half-life: trailing by $H$ rank positions halves the quality weight. Candidates sharing the same path-best ancestor share its effort denominator. A child that establishes a new path-best fitness becomes its own reference and initially has zero subtree effort. Direct development counts remain recorded separately. This rule is implemented by `lineage_peaks` and `parent_probabilities`.

**Operator and structural-distance reference selection.** Given primary parent $v$, EvoTrace samples `Refine`, `Explore`, or `Crossover` with probabilities $0.45$, $0.30$, and $0.25$. When `Crossover` is chosen, EvoTrace forms the candidate reference pool by excluding $v$ and any feasible node with identical normalized source (`identity(code)`), computing rank quality $q(u)$ within the remaining distinct-source feasible candidates, and retaining those with $q(u)\geq 0.30$ (`reference_quality_floor`). Preferring ordered pairs `(v, u)` not yet recorded in `used_crossover_pairs` (and allowing reuse only after all eligible pairs for $v$ have been tried), EvoTrace selects the reference $u$ that maximizes AST multiset Jaccard distance (`code_distance`, built from docstring-stripped AST parent-child type pairs and traversal bigrams, breaking ties by better fitness then earlier ID). If no eligible reference exists, the step executes `Refine`.

**Task-aligned multi-scale tree views.** Rather than truncating context at arbitrary operator boundaries, the three operators read complementary structural slices of $T$ matched to their design scope:

| Operator | Structural view of $T$ | Design role |
| --- | --- | --- |
| Refine | Current code and score; full retained root-to-parent `[Evolution Path]` with uncontracted diffs, contracted stage summaries $V$, and path-node $S$; root $S_r$; `[Attempts from Current Algorithm]` (parent $S_v$ plus retained valid and failed children) | Improve the current computation using its formation trajectory and local trial outcomes |
| Crossover | Current and reference implementations and scores; true `[Common Ancestor]` (or root `[Origin Context]`); both diverging branches since the LCA (`[Current Branch]` and `[Reference Branch]`) with $S$ and $V$; outgoing attempts from the primary parent | Combine complementary mechanisms informed by how the two lineages diverged |
| Explore | Fitness-ordered `[Population Individuals and Experience]` across retained feasible individuals (designs, scores, parent origins, and attached $S/V$, plus root $S_r$), followed by `[One Existing Implementation]` | Formulate an algorithm with a substantially different core computation |

Uncontracted path steps display score transitions and compact normalized-code diffs alongside baseline code where needed; contracted edges display endpoint score changes, resulting designs, and stage experience $V$.

When a rendered prompt exceeds `max_input_tokens` ($30{,}000$), EvoTrace iteratively halves the optional display window (`max_path_steps`, `max_unconsolidated_attempts`, `max_unconsolidated_failures`, `max_population_members`, or `max_experiences`) whose reduction yields the smallest measured prompt size while keeping at least one path or population entry, current and reference code, and required diff baselines. Context fitting adjusts only the transient prompt view without altering $T$ or its stored experience. Each generation call outputs strict JSON containing `"analysis"`, `"code"`, and `"design"`.

### 3.6. Search Loop

Algorithm 1 summarizes the complete EvoTrace procedure. Initialization and evolutionary steps share the same evaluation budget $B$ and single-repair allowance. Format failures trigger at most one format-repair request without consuming evaluation budget; runtime errors, invalid outputs, or timeouts trigger at most one program-repair request whose re-evaluation consumes budget. Only attempts that execute at least one program evaluation attach a node (the final evaluated version) to $T$ and advance $t$.

```text
Algorithm 1: EvoTrace
Input: task interface, evaluator, LLM, evaluation budget B
Initialize virtual root r, empty tree population T, and attempt archive A
while evaluations_used < B and consecutive_unevaluated < 10:
    if feasible initial programs in A < 5:
        build initialization prompt from task interface and initial archive
    else:
        sample parent v ~ P(v) via Eq. (4) and operator in {Refine, Explore, Crossover}
        if operator == Crossover: select high-quality max-AST-distance reference u
        build and token-fit operator-aligned context view from T
    generate candidate (with at most one format or runtime repair) and evaluate
    log all LLM calls and evaluation outcomes to A
    if at least one program evaluation was executed:
        attach final evaluated node to primary parent in T; update d, subtree, u, t
        if initialization is complete:
            compute distinct D, active A with champion, and LCA skeleton H
            partition nodes outside H into path and side-branch regions
            select whole regions with at least m retired nodes
            reflect on selected regions; bind comparison catalogs
            remove selected regions, reconnect endpoints, and attach updated V and S
    commit attempt record and updated tree snapshot
return archived feasible program in A with best training fitness
```

## 4. Preliminary Evaluation

The evidence in this section comes from an earlier V3 configuration using `quality_slots = 32`, `recent_evaluations = 40`, and direct-effort parent sampling, before the working-set and minimum-region-size rule in Section 3.3. These frozen observations describe that configuration and motivate batched reflection; they do not measure the current maintenance rule.

We evaluate the source-frozen EvoTrace implementation across four standard combinatorial optimization benchmarks and report a frozen multi-seed search snapshot from the 15 November 2026 deployment (captured at 15:36 Beijing time; Appendix D). The deployment comprises 12 independent runs (four tasks across search seeds 0, 1, and 2), each configured with a target budget of $B=1{,}000$ evaluations. At snapshot capture, individual runs had completed between 180 and 256 actual program evaluations (168 to 247 committed generation attempts). We report training fitness across all 12 runs at their largest common horizon of 180 evaluations and audit population maintenance and reflection across all 2,641 committed evaluations.

### 4.1. Benchmark Tasks and Execution Protocol

Each benchmark evolves a key decision function embedded in a fixed optimization framework:

- **Traveling Salesman Problem (`tsp_construct`, TSP):** evolves sequential next-node selection for constructive tour generation on 50-node Euclidean instances, minimizing mean tour length.
- **Capacitated Vehicle Routing Problem (`cvrp_aco`, CVRP):** evolves heuristic edge desirability priors for Ant Colony Optimization (30 ants, 100 iterations), minimizing mean route length.
- **Flow Shop Scheduling Problem (`fssp_gls`, FSSP):** evolves penalty-matrix and job-perturbation scoring within Guided Local Search, minimizing mean makespan.
- **Orienteering Problem (`op_aco`, OP):** evolves edge heuristic priors for Ant Colony Optimization (20 ants, 50 iterations) under a tour-length budget, minimizing negative mean prize collected.

Each training evaluation executes on 16 fixed instances under a pooled 160 instance-second time budget across up to 8 parallel worker processes (2 GiB memory limit per worker, deterministic per-instance seeds offset from `730241`). Failed executions and repair re-evaluations each count toward $B$. A disjoint set of 50 instances (500 pooled instance-seconds) is reserved for single-shot held-out testing after the 1,000-evaluation search concludes. All LLM calls route through a shared `Qwen3.8-27B` pool with native thinking disabled: candidate generation uses temperature $1.0$ and up to $16{,}384$ output tokens, while region reflection uses temperature $0.3$ and up to $1{,}536$ output tokens (full parameters in Appendices A and D).

### 4.2. Search Trajectories at the 180-Evaluation Horizon

Table 1 summarizes the best training fitness reached by each seed at 180 evaluations alongside the three-seed sample mean and sample standard deviation (SD). Figure 1 pairs the common-horizon best-fitness trajectories (top row) with the exact number of retained individuals in $T$ after every committed insertion and simplification step up to each run's observed endpoint (bottom row).

**Table 1. Best training fitness at 180 evaluations (lower is better).**

| Task | Seeds 0 / 1 / 2 | Mean | SD |
| --- | ---: | ---: | ---: |
| TSP | 5.8527 / 5.8362 / 5.9800 | 5.8897 | 0.0787 |
| CVRP | 8.5592 / 8.5845 / 8.2413 | 8.4617 | 0.1913 |
| FSSP | 3928.81 / 3928.25 / 3927.69 | 3928.25 | 0.56 |
| OP | -14.3919 / -14.7719 / -14.6069 | -14.5902 | 0.1905 |

![Early search and retained population](../paper/figures/early-search.png)

**Figure 1. Search progress and population size across 12 frozen runs.** Upper panels plot best training fitness over the common 180-evaluation horizon (solid curves show three-seed means; thin curves show seeds 0, 1, and 2). Lower panels plot the number of retained individuals $|\mathcal K\setminus\{r\}|$ across the first 180 evaluations, verified against committed tree snapshots. Survival selection and LCA closure stabilize active population size once search progresses past the 40-evaluation recency window.

Across all four tasks and all 12 seeds, training fitness improves substantially beyond the five-program initialization phase, with rapid gains during the first 60 evaluations followed by continued refinement and recombination between evaluations 60 and 180 (for example, CVRP seed 2 drops to 8.2413 and OP seed 1 reaches -14.7719). Simultaneously, the lower panels of Figure 1 show the structural effect of survival selection: population size grows roughly linearly during the initial 40-evaluation recency window and then stabilizes tightly, ending between 54 and 70 retained individuals across all 12 runs even at 256 evaluations (`quality_slots = 32` plus active recent trials and connecting LCA forks).

### 4.3. Structural Simplification and Recursive Reflection Dynamics

Auditing all 2,481 committed attempts (2,641 evaluations) across the 12 prefixes documents node removal, region assignment, and reflection under the frozen retention rule:

- **Recorded removal and retained state:** Of the 2,481 nodes attached to $T$, survival updates have removed 1,734 implementations ($69.9\%$) and retained 747 explicit individuals across the 12 active trees, triggered across 1,195 simplifying updates. Every removed node in every update maps to a single region, with zero orphaned nodes or accounting drift.
- **Balanced path contraction and branch pruning:** Once search advances well past $W=40$, both simplification modes operate heavily: the 1,195 updates partition the 1,734 removed nodes into **509 path-contraction regions** (updating edge $V$) and **887 side-branch regions** (updating node or root $S$), totaling 1,396 regions.
- **Incremental recursive condensation:** All 1,396 regions completed structured LLM reflection with **zero fallback summaries** (`0` parse or schema errors). Measured by newly removed nodes per region, 1,127 regions ($80.7\%$) condense 1 newly removed implementation, 204 condense 2, 62 condense 3, 2 condense 4, and 1 condenses 5, integrating each incremental removal with any carried-over experience and provenance catalogs already present on its carrier.
- **Separate reflection accounting:** Across all 2,641 program evaluations, the 1,396 reflection calls consumed 4,897,554 input tokens and 256,594 output tokens in total (roughly 3,508 input tokens and 184 output tokens per region), logged independently of the solver evaluation budget.

## 5. Discussion

**Competition, structure, and experience.** EvoTrace uses the same tree to select individuals, organize related attempts, and present experience to generation. Parent sampling combines individual rank quality with lineage effort. The working set and LCA skeleton determine which parts of the tree remain explicit, while region reflection condenses the related attempts selected for simplification. Path-, branch-, and population-scale views then use this maintained state.

**Why batch reflection.** Under the earlier rule in Section 4, 80.7% of regions contained only one newly removed node. The current threshold groups at least $m$ retired nodes in each reflected region. This changes reflection frequency and the material available for a note; whether it improves the note or subsequent generation requires direct evaluation.

**Evaluation scope.** The frozen runs document early search and experience handoffs under the earlier retention rule. Current working-set simplification has been checked through structural replay and invariant tests. Full-search comparisons and fixed-state experience studies in Appendix E are needed to assess the current method's search performance and how its reflected experience affects generation.

## 6. Conclusion

EvoTrace combines population competition and derivation structure in an experience-bearing tree-structured population. Reflective tree simplification compresses related path and side-branch attempts into node and edge experience as the tree changes. Global parent sampling and operator-specific tree views use that same state for continued algorithm design, connecting population growth, experience condensation, and subsequent generation.

## Appendix A. Default Hyperparameters and Accounting Details

| Component | Default setting |
| --- | --- |
| Initialization and budget | 5 feasible initial programs; $B=1{,}000$ total actual program evaluations |
| Duplicate collapse | Collapse feasible nodes sharing `(identity(code), fitness)` to earliest node ID |
| Active working set (`active_evaluations`) | $W=40$ evaluations since birth, direct development, or selection as crossover reference; always retain champion |
| Batch threshold (`region_size`) | Reflect and simplify whole regions containing at least $m=4$ retired nodes |
| Structural connectivity | Pairwise LCA closure of $\mathcal A$ plus virtual root $r$ |
| Parent sampling (`rank_half_life`) | Half-life $H=10.0$ average-rank positions; lineage-effort discount $1/(1+s(b(v)))$ |
| Operator weights | `Refine` $0.45$; `Explore` $0.30$; `Crossover` $0.25$ |
| Crossover reference rule | Distinct normalized source; rank quality $q(u)\geq 0.30$; unused pair first, max AST `code_distance` |
| Evolution prompt window | `max_input_tokens` $=30{,}000$; iterative halving of optional display windows |
| Generation LLM settings | Temperature $1.0$; `top_p` $0.8$; `top_k` $20$; `presence_penalty` $1.5$; max output $16{,}384$ tokens |
| Reflection LLM settings | Temperature $0.3$; max output $1{,}536$ tokens; target ~120-word JSON `"experience"` note |

**Numerical and edge-case rules.** In `quality_scores`, tied fitness blocks spanning zero-based indices `start` through `end - 1` among $M$ feasible candidates receive normalized quality $q(v)=1-(\text{start}+\text{end}-1)/(2(M-1))$ when $M>1$, and $0.5$ when $M=1$. `parent_probabilities` shifts average ranks by the best tied rank before exponentiation in log space and floors normalized probabilities by `math.ulp(0.0)` so extreme rank gaps never cause numerical zero-division or silent exclusion. Direct effort $d(v)$ counts every evaluated `Refine`, `Explore`, or `Crossover` attempt with $v$ as primary parent (including failed executions and repair re-evaluations), and never changes when ancestors contract or side branches prune.

**Token counting and fallback rules.** When served through the scheduled endpoint pool, `count_tokens` queries the serving chat tokenizer; standalone runs without a tokenizer fall back to UTF-8 byte length and record `"utf8_bytes"` in `input_window`. If a reflection call fails or violates schema, EvoTrace renders up to `MAX_EXPERIENCE_CHARS = 960` characters of measured comparisons from `comparison_catalog` plus clause-truncated prior notes while retaining the complete structured `observations` list. Search halts early only if `max_unevaluated = 10` consecutive attempts fail to produce any program evaluation.

## Appendix B. Structural Properties of One Simplification Update

Consider a finite pre-update tree $T_0$ with active set $\mathcal A$, skeleton $\mathcal H$, candidate regions $\mathcal C$, and final retained set $\mathcal K$ as defined in Section 3.3. The virtual root is always retained. The following properties concern the structural update at a fixed evaluation clock.

**Tree preservation.** Every retained non-root node has a unique nearest retained ancestor in $T_0$. Reconnecting to that ancestor strictly decreases original depth, so repeated parent traversal reaches the root without a cycle. Each retained non-root node has one parent; the result is a rooted tree.

**Unique region assignment.** Chains between adjacent skeleton nodes have disjoint interiors. Otherwise, two such chains would share a fork outside $\mathcal H$, contradicting LCA closure. Every remaining node outside the skeleton reaches a unique nearest ancestor in a chain interior or in $\mathcal H$. In the first case it joins that path region; in the second it joins the side-branch region at that skeleton node. Thus the regions partition $\mathcal N\setminus\mathcal H$. Selecting whole regions by size preserves disjointness, and every removed node belongs to one selected region.

**Champion retention and effort preservation.** The champion belongs to $\mathcal A\subseteq\mathcal H\subseteq\mathcal K$. Reconnection changes parent pointers and experience carriers, while retained nodes' effort counters and activity clocks are copied unchanged. Removed evaluations remain in the global count and the applicable retained ancestors' historical counters.

**One-step idempotence.** Without new search activity, all originally active individuals and their LCA skeleton remain retained. If removing an older duplicate representative makes a deferred copy eligible for the working set, the new skeleton can gain nodes but cannot lose an original boundary. Every remaining region therefore lies within a previously deferred region: adding skeleton nodes can split it, while preserved boundaries prevent merging across regions. Each such region still contains fewer than $m$ nodes, so replanning produces no additional removal.

## Appendix C. Verbatim Reflection Prompt Templates

Both templates in `evotrace_v3/reflection.py` follow a common header supplying `[Task]`, `[Function Interface]`, and `Score direction`, followed by the region evidence (`[Path Evidence]` or `[Branch Experiments]`, including code, diffs, `[Related Results]`, inherited notes tagged with `Supporting comparisons`, and `[Comparison Catalog]`). The `[Task]` and `[Output]` blocks are reproduced verbatim below.

**Variation experience prompt (`VARIATION_PROMPT_TEMPLATE`, path contraction).**

> `[Task]`
> Compress the algorithm-design experience along this transition for a researcher continuing the search.
> Write a short research note preserving what is worth remembering from this development: the computational ideas, informative outcomes, and relevant earlier experience. A faithful account is enough; an explanation or general principle is optional. Keep any interpretation tentative and leave future design decisions open.
>
> `[Output]`
> Return one JSON object with an "experience" string: a concise paragraph, about 120 words, explaining how the computation developed and what is worth remembering from the intervening experiments. Include informative outcomes and earlier experience. The paragraph will accompany the surviving implementation in later design calls; name computations rather than trial numbers. The system supplies endpoint scores.

**Selection experience prompt (`SELECTION_PROMPT_TEMPLATE`, side-branch pruning).**

> `[Task]`
> Compress the algorithm-design experience from these experiments for a researcher continuing the search.
> Write a short research note preserving what is worth remembering from these experiments: the computational ideas, informative outcomes, and relevant earlier experience. A faithful account is enough; an explanation or general principle is optional. Keep any interpretation tentative and leave future design decisions open. These branches were removed to manage the population; their ideas are not thereby disproved.
>
> `[Output]`
> Return one JSON object with an "experience" string: a concise paragraph, about 120 words, relating the computational ideas tried and their informative outcomes, including earlier experience. It will accompany an algorithm in later design calls, preserving experiences its code alone cannot convey. The paragraph must stand on its own without trial numbers.

## Appendix D. Reproducibility and Deployment Accounting

The earlier quality-quota configuration (`quality_slots = 32`, `recent_evaluations = 40`) in batch `20261115_115206_evotrace_v3_revised_co4` froze 463 repository source files, task evaluators, and configurations prior to launch on 15 November 2026. Every committed line in `native/attempts.jsonl` records the generated program, all initial and repair evaluations, sampled `parent_id`, `parent_probability`, `operator`, `reference_id`, `reference_distance`, token-fitted context metadata, simplification triggers, reflection prompts and responses, and the resulting `TreePopulation` snapshot. Resuming from `attempts.jsonl` replays committed tree snapshots deterministically without re-invoking reflection calls and rejects any mismatched method or parameter configuration.

Search runs use seeds 0, 1, and 2. TSP training uses 16 Euclidean 50-node instances (`seed = 2024`). CVRP uses 30 ants and 100 ACO iterations (`aco_seed = 1234`); OP uses 20 ants and 50 ACO iterations (`aco_seed = 1234`); FSSP uses fixed benchmark instances and GLS settings from the frozen task configuration. Training evaluations run with `timeout_scope = "whole_task"`, `time_budget = "pooled_instance_seconds"`, `timeout_seconds = 160.0` (10.0 seconds per instance pooled across 16 instances), `n_workers = 8`, `memory_limit_bytes = 2147483648`, and `evaluation_seed = 730241`. Prompts expose only the task description, score direction, and function template, keeping dataset sizes and solver budgets external to generation instructions.

Calls to `Qwen3.8-27B` (`enable_thinking = false`) are scheduled across a shared local endpoint pool (routing across FP8 and GGUF serving instances and logging per-request endpoint metadata and token counts). Provider transport errors are retried and accounted separately from candidate execution failures. `paper/evidence/snapshot.json` records per-run file paths, SHA-256 checksums, sanitized configurations, and evaluation series; `paper/build.py` and `paper/validate.py` compile and verify both manuscripts directly against this frozen file without refreshing live runs.

## Appendix E. Protocol for Full-Budget and Controlled Ablation Studies

The current maintenance rule requires a separately frozen search batch. The following protocol applies to that batch and its controlled mechanism checks; results from the earlier configuration remain separate:

1. **Matched-budget full search and held-out testing.** Run all compared population, tree, and reflective baselines across seeds 0, 1, and 2 per task under identical training instances, pooled 160 instance-second parallel evaluation limits, `Qwen3.8-27B` sampling parameters, and $B=1{,}000$ actual evaluation accounting. Evaluate each run's best feasible training program once on the reserved 50-instance test split (500 pooled instance-seconds), reporting seed-level test scores, training convergence curves, failure rates, and total generation, repair, and reflection token costs.
2. **Fixed-snapshot experience representation (`include_experiences` ablation).** Sample committed tree snapshots at declared evaluation checkpoints without filtering for downstream improvement. Holding the snapshot, `parent_id`, `operator`, `reference_id`, and token budget fixed, compare repeated candidate generations under three representations of the same removed regions: (a) experience omitted (`include_experiences = False`), (b) raw diffs and outcomes from the removed regions, and (c) EvoTrace's reflected $S/V$ notes. Pair analyses at the snapshot level and report child feasibility, parent-relative win rate, fitness delta, and prompt tokens by operator.
3. **Structural placement versus flat experience list.** Holding the reflected $S/V$ notes and input token budget fixed on identical snapshots, compare EvoTrace's path- and ancestor-localized presentation against a flat chronological list of the same notes with source IDs retained, isolating the benefit of lineage-aligned placement.
4. **Multi-handoff fidelity audit.** Draw a stratified random sample of recorded reflections across tasks, region types (path vs. side), and cumulative handoff depth. Audit each note against `attempts.jsonl` and its bound `Comparison Catalog` for outcome-direction accuracy, code-change faithfulness, and over-generalized causal claims.

## References

References are rendered from `paper-references.bib` in the accompanying PDF.
