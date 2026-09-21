# AAD 高层设计与系统级算法发现综述

核查日期：2026-09-20。范围：本地论文库与单篇精读笔记中，设计对象高于固定求解器内单个 heuristic、scoring function 或 operator 的自动算法设计（AAD/AHD）工作，并补充论文作者公开的原始版本。这里的“高层”有四种不同含义，不能混为一谈：

1. **设计对象变大**：从一个函数扩展到启发式集合、优化器、问题规约、算法组件组合或完整 solver；
2. **搜索控制被设计**：自动决定如何选父代、改写、分配评价预算、切换搜索路线和何时停止；
3. **运行时控制被设计**：LLM/agent 自己决定何时生成、评估、修复、检索记忆或协作；
4. **设计能力被内化**：用微调或强化学习把搜索经验写进模型，使模型直接生成可复用 solver。

这份综述的核心判断是：**该领域正在从 template-bound AHD 走向层次化、程序级和 agentic AAD，但“开放搜索空间”仍被 evaluator、接口、执行契约、预算和任务表示强烈约束。当前最可靠的研究对象不是“完全自由的算法生成”，而是可验证边界内逐层放大设计自由度，并用相同真实评价预算证明每一层的增益。**

## 1. 研究问题：AAD 到底在设计什么

传统 LLM-AHD 可以抽象成：

```text
固定问题表示 + 固定求解器框架 + evaluator
                         └── LLM 设计一个函数 h
```

高层 AAD 把可变对象逐步扩大：

```text
实例解
  → 单个可复用 heuristic
  → heuristic 集合 / portfolio
  → heuristic-optimizer / meta-optimizer
  → 表示、规约和算法结构
  → 完整 solver pipeline
  → 搜索控制器 / agent policy
  → 可复用的算法生成模型
```

这些层次不是简单的性能排行榜。设计对象不同，评价成本、可执行性风险、泛化含义和部署方式都不同。例如：

- heuristic 集合的逐实例 `oracle best` 不等于真实部署性能，部署还需要路由器或运行全部成员；
- meta-optimizer 的得分受内层候选数、任务数和 best-of-k 极值效应影响；
- 完整 solver 的一次结构改写可能同时改变几十个模块，必须处理依赖闭合和信用分配；
- agentic AAD 的收益可能来自更多工具调用，而不是更好的算法设计；
- 微调模型的收益必须与训练成本和推理期搜索成本分账。

因此，领域认知应沿两个正交轴组织：**“设计什么”**与**“如何搜索/控制”**。

## 1.1 先澄清：这是新方向，但不是凭空出现的新问题

把设计对象提升到启发式之上，并不是 LLM 首次提出的研究问题。较早的 **hyper-heuristic** 已经研究“自动选择或生成启发式”；自动算法配置用 ParamILS、SMAC、irace 等方法在算法参数和配置空间中搜索；遗传程序设计和 grammatical evolution 直接演化可执行程序；learning-to-optimize 则把优化器本身作为可学习对象。[hyper-heuristic 综述](https://doi.org/10.1007/s10462-026-11531-8) [自动元启发式设计综述](https://arxiv.org/abs/2303.06532) [Learning to Optimize](https://arxiv.org/abs/1606.01885)

LLM-AAD 的新意主要在于把这些传统问题与三种能力结合起来：

- **语义程序提议**：LLM 可以直接提出带自然语言理由的代码结构，不必把整个空间预先离散成参数或语法规则；
- **开放式表示变化**：问题规约、组件组合和控制流可以通过代码/文本重写改变；
- **自然语言反馈接口**：失败原因、设计意图和历史经验可以进入下一次生成上下文。

所以，MoH、A2DEPT 等工作的正确定位不是“发明了自动设计高层算法”，而是把已有的 hyper-heuristic、程序综合和元优化问题，放进 LLM 驱动的执行反馈闭环中，并把可搜索对象从参数或单函数扩展到更高层的程序结构与控制策略。

这一区分会影响研究问题：真正需要证明的不是“LLM 能不能写完整代码”，而是 **语义生成是否让更高层搜索在相同真实评价预算下发现了固定表示或固定框架无法到达的算法行为**。

## 1.2 统一表示：AAD 是在联合搜索哪些变量

为了避免不同论文使用“算法”“优化器”“框架”时指代不一致，可以把一个可部署的算法系统写成：

```text
z = (R, A, H, controller, router, model)
```

其中：

- `R`：问题表示、规约和数据接口；
- `A`：solver 的控制流、阶段和组件拓扑；
- `H`：具体 heuristic、scoring function 和组件实现；
- `controller`：生成、选择、修复和预算分配策略；
- `router`：部署时的实例选择、portfolio 路由或在线适应策略；
- `model`：生成模型或 agent policy 的参数。

给定任务分布、执行评价器和成本约束，高层 AAD 的真实目标应接近：

```text
maximize expected quality of z on held-out instances
         minus deployment cost and design cost
subject to executable(z) and feasible(z, instance)
```

传统固定框架 AHD 通常只搜索 `H`，并把其余变量固定；MoH 主要扩大到 `controller`；RedAHD 主要扩大到 `R`；A2DEPT、BEAM 和 DGA²D 扩大到 `A` 与 `H`；InstSpecHH 和 EoH-S 扩大到 `H` 与 `router`；AHD Agent 和 CORAL 主要学习运行时 `controller`；CALM、EvoTune 和 Hero 主要改变 `model`。

这说明“高层 AAD”不是单条路线，而是一个**联合变量空间**。两篇论文即使都说自己在做 system-level design，也可能只改变其中一个变量；比较时必须先指出变化发生在哪个坐标。

## 2. 领域谱系与代表方法

### 2.1 组合对象：从单个 heuristic 到算法集合

**EoH-S** 将目标从单条启发式扩展为互补启发式集合 $H=\{h_1,\ldots,h_k\}$。它用实例级性能向量和集合覆盖指标 CPI 选择成员，并以互补种群管理保持不同成员的覆盖范围。这个方向回答的是“一个规则无法覆盖所有实例时，是否应该设计一组规则”。本地精读显示，CPI 是逐实例取最优的潜在覆盖指标，真实部署仍需要选择器或 $k$ 倍运行成本；成绩向量距离也只是互补代理，不能排除某个成员被全面支配。[EoH-S 精读](</home/fang/code/LLM4AD/TraceAAD/docs/05-文献与基线/单篇阅读/10-EoH-S.md:1>)

**InstSpecHH** 更进一步，把生成和使用分开：按实例子类生成候选，再通过特征、最近邻或分类器选择适合当前实例的启发式。它说明高层设计不一定要合成一个全局 solver，也可以设计“算法集合 + 选择策略”的组合。[InstSpecHH 精读](</home/fang/code/LLM4AD/TraceAAD/docs/05-文献与基线/单篇阅读/32-InstSpecHH.md:1>) 但其主要增益来自邻域子类迁移，生成、迁移和选择器的作用不能直接合并解释。

这条路线的研究问题是 **portfolio design**，而不是完整算法结构搜索。它适合实例异质性明显、算法之间存在分工的任务，但必须报告选择误差、选择开销和全量运行开销。

### 2.2 优化器层：设计“如何设计 heuristic”

**MoH（Meta-Optimization of Heuristics）** 把固定的 EC heuristic-optimizer 也变成搜索对象。外层 meta-optimizer 生成多个 heuristic-optimizer；每个内层优化器在多个任务上生成 heuristic；外层按跨任务 utility 选择下一代 meta-optimizer。生成的策略可能类似 EC、ACO、PSO、模拟退火、禁忌搜索或混合策略。[MoH 精读](</home/fang/code/LLM4AD/TraceAAD/docs/05-文献与基线/单篇阅读/31-MoH.md:1>) [原论文](https://arxiv.org/abs/2505.20881)

MoH 的关键变化是：LLM 不只回答“这个 heuristic 怎么写”，还回答“候选怎样挑、如何提出改进、如何生成后代、如何保留精英”。它证明了优化器层可以成为 AAD 的独立研究对象。

但 MoH 仍然保留了任务接口、评价器、双层循环和种群管理等外部骨架。论文的主要泛化证据是跨规模；元优化本身没有与固定 seed optimizer 的严格单变量对照，跨问题结果也只是初步实验。因此它是 **optimizer-level AAD**，还不是完全开放的系统合成。

### 2.3 表示与问题规约层：设计“问题应该怎样被求解”

**RedAHD** 让 LLM 生成从原问题到另一个更易处理问题的规约，包括实例映射、解映射、目标问题描述和代码模板；然后在规约后的问题上运行启发式设计。设计对象由固定 GAF 中的函数扩展为“规约 + 规约后的算法”。[RedAHD 精读](</home/fang/code/LLM4AD/TraceAAD/docs/05-文献与基线/单篇阅读/29-RedAHD.md:1>) [原论文](https://arxiv.org/abs/2505.20242)

它的研究价值不在于某个新算子，而在于把 **representation / reduction** 纳入 AAD：如果当前模板不适合问题，系统可以先改写问题的表示，再进行算法设计。

风险也很明确：规约可能在训练实例上看起来有效却丢失原问题结构；映射合法不等于映射保留了有用的优化性质。未来评价必须同时检查映射正确性、解回译合法性、训练—测试差距和规约成本。

### 2.4 完整程序层：从函数槽位到 solver architecture

**A2DEPT** 直接把搜索对象定义为完整可执行 solver 程序，用进化程序树、微调、宏观重写、语义交叉和程序维护环处理结构搜索、依赖闭合和错误修复。它把 template-bound AHD 明确重述为 framework bottleneck：固定框架把搜索限制在一个 basin 中，可能把真正重要的算法范式排除在外。[A2DEPT 精读](</home/fang/code/LLM4AD/TraceAAD/docs/05-文献与基线/单篇阅读/33-A2DEPT.md:1>) [原论文](https://arxiv.org/abs/2604.24043)

A2DEPT 的主要贡献是把三件事接成闭环：

- 程序树保留结构谱系和可继续扩展的低分 stepping stones；
- micro-tuning 与 macro-mutation 分离局部精炼和范式重写；
- 依赖闭合、死代码裁剪和错误反馈使完整程序有机会执行。

它的结果支持“在某些任务上完整程序搜索能超过固定模板”，但不能把整套配方的收益归因给程序树单一因素。其维护循环主要处理缺失依赖，跨模块类型契约破坏仍是明显天花板；论文自身的主张数字也需要以主表而非摘要为准。

**BEAM** 采用更明确的双层结构：外层 GA 进化高层算法结构和函数占位符，内层 MCTS 实现函数；同时用自适应记忆保存高适应度、新颖、被使用且较新的函数，并以知识增强管线评价完整 solver。[BEAM 精读](</home/fang/code/LLM4AD/TraceAAD/docs/05-文献与基线/单篇阅读/34-BEAM.md:1>) [原论文](https://arxiv.org/abs/2604.12898)

BEAM 的研究问题是 **结构变量与函数变量是否应分层搜索**。它比单层函数进化更接近 solver synthesis，但质量分解假设依赖“结构先正确、函数改进才有意义”；结构选错时内层预算会被浪费。自适应记忆的长期收益也需要把上下文扩大、重复评估和实际复用区分开。

**DGA²D** 用 directed operator graph 表示完整 pipeline：有向 walk 组成算法流程，组件实现池和图边按局部转移信用更新。它研究的是“算法组件如何排列、复用和组合”，而不是单个函数的质量。[DGA²D 精读](</home/fang/code/LLM4AD/TraceAAD/docs/05-文献与基线/单篇阅读/51-DGA2D.md:1>)

DGA²D 的 first-order credit 只描述一个 solver 内部的组件调用拓扑，不能误读成外层演化谱系或提示历史。这个区分很重要：**pipeline 结构、代码谱系、求解轨迹和提示上下文是四种不同的“轨迹”。**

### 2.5 程序进化与工程系统层：可变范围更大，但 scaffold 仍在

**AlphaEvolve**、**ShinkaEvolve** 等系统把 LLM、程序档案、异步候选生成和自动 evaluator 组织成通用的代码进化 harness。AlphaEvolve 支持 patch 或更大范围的代码修改，在算法、芯片、数据中心调度等任务中展示了程序改进；ShinkaEvolve 研究模型调用的 bandit 分配、嵌入去重和长期程序档案。[AlphaEvolve 精读](</home/fang/code/LLM4AD/TraceAAD/docs/05-文献与基线/单篇阅读/12-AlphaEvolve.md:1>) [AlphaEvolve 原论文](https://arxiv.org/abs/2506.13131) [ShinkaEvolve 精读](</home/fang/code/LLM4AD/TraceAAD/docs/05-文献与基线/单篇阅读/13-ShinkaEvolve.md:1>)

这类方法的设计对象可以很大，但仍依赖人类提供的任务表示、接口、测试和 evaluator。所谓“直接修改完整代码”是生成接口，不等于系统脱离了固定边界。它们最有价值的启发是：程序档案、patch、反馈和异步资源管理可以成为通用 AAD 基础设施；它们最薄弱的地方是跨任务可比性和组件因果归因。

**EvoStage** 将设计拆为多个可执行阶段，由 coordinator 分解任务、在中间阶段运行并用中间反馈修正后续设计，再和全局探索/增强算子共同进化。[EvoStage 精读](</home/fang/code/LLM4AD/TraceAAD/docs/05-文献与基线/单篇阅读/55-EvoStage.md:1>) 这是从“修改程序”走向“设计过程结构”的路线，但阶段划分、反馈频率和 coordinator 的职责仍是重要的人工先验。

### 2.6 Agent / multi-agent 层：设计搜索过程本身

**AHD Agent** 把 AHD 变成 agentic RL。agent 在每一步决定是生成/修改 heuristic、调用 evaluator、获取诊断信息还是继续迭代；训练目标是让模型学习多步搜索控制，而非被固定进化循环被动调用。[AHD Agent 精读](</home/fang/code/LLM4AD/TraceAAD/docs/05-文献与基线/单篇阅读/38-AHD-Agent.md:1>) [原论文](https://arxiv.org/abs/2605.08756)

它把高层设计变量从“程序结构”扩展到“何时做什么”。难点是延迟奖励和工具调用成本混杂：如果 agent 只是进行了更多尝试，质量提升不能直接说明控制策略更好。必须保存每一步观察、动作、代码、评价和预算，才能做信用分配。

**CORAL** 进一步把检索、提出、评价、记忆更新和干预的决定权交给多个长驻 agent，通过异步执行和持久共享记忆进行开放式发现。[CORAL 精读](</home/fang/code/LLM4AD/TraceAAD/docs/05-文献与基线/单篇阅读/40-CORAL.md:1>) [原论文](https://arxiv.org/abs/2604.01658)

CORAL 的 best-of-4 独立 agent 对照是这条路线中较有价值的公平性设计：它尝试区分协同收益和单纯增加计算。但共享记忆错误会跨 agent 放大，heartbeat 等控制机制的独立作用尚未充分拆解。

### 2.7 模型层：把算法设计能力写进权重

**CALM、EvoTune、Fine-tuning LLM for AAD、Self-Developing** 等方法不再只在推断期维护候选，而是用搜索产生的程序和反馈构造 DPO、GRPO 或其他训练信号，让模型本身逐步成为 algorithm factory。[CALM 精读](</home/fang/code/LLM4AD/TraceAAD/docs/05-文献与基线/单篇阅读/35-CALM.md:1>) [EvoTune 精读](</home/fang/code/LLM4AD/TraceAAD/docs/05-文献与基线/单篇阅读/36-EvoTune.md:1>)

**Beyond Inference-Time Search / Hero** 则直接训练代码模型生成可一次编译、跨实例复用的 solver，并用 held-out 实例检验 compile-once 行为。[Hero 精读](</home/fang/code/LLM4AD/TraceAAD/docs/05-文献与基线/单篇阅读/41-Beyond-Inference-Time-Search.md:1>) [原论文](https://arxiv.org/abs/2605.18374)

这条路线的核心假设是：昂贵的实例内搜索可以被模型训练摊销。它与程序级搜索正交：一个改变“搜索对象”，一个改变“能力存放位置”。判断是否真的学到了算法能力，必须至少报告：训练成本、推断期调用、冻结模型的 pass@k、固定代码在未见实例上的表现，以及与同预算 best-of-k 的比较。

### 2.8 时间演进：研究对象和瓶颈如何逐步上移

按问题而不是按论文年份看，这个方向大致经历了五个阶段：

| 阶段 | 主要对象 | 典型方法 | 主要瓶颈 |
| --- | --- | --- | --- |
| 1. 固定槽位搜索 | 一个可替换函数 | FunSearch、EoH、ReEvo | 搜索被模板和单任务目标限制 |
| 2. 搜索器与组合对象 | 启发式集合、选择器、元优化器 | EoH-S、InstSpecHH、MoH | 多层 best-of-k、选择偏差和预算方差 |
| 3. 表示与结构开放 | 规约、组件拓扑、完整程序 | RedAHD、DGA²D、A2DEPT、BEAM | 程序爆炸、无效率、依赖和信用分配 |
| 4. 过程自治 | 搜索控制器、阶段和多 agent | EvoStage、AHD Agent、CORAL | 延迟奖励、工具成本、长期记忆污染 |
| 5. 能力摊销 | 生成模型或可复用 solver | CALM、EvoTune、Hero | 训练成本、过拟合和真实 OOD 迁移 |

这个顺序不是简单的“越往后越先进”。每上移一层，搜索空间变大，控制信号更稀疏，验证和部署成本更高。一个可靠的固定框架在某个任务上可能优于开放式搜索；开放式搜索只有在框架先验不匹配、结构变化确实有价值且预算足够时才有优势。

### 2.9 方法边界：名称相近不代表研究对象相同

以下区分对阅读结果非常关键：

- **EoH-S/InstSpecHH 是 portfolio 设计。** 它们扩展的是多个候选及其路由方式，不一定设计一个新的 solver 控制流。
- **MoH 是 optimizer 设计。** 它改变“如何产生和改进 heuristic”的程序，但仍依赖外层固定接口、utility 和双层流程。
- **RedAHD 是表示/规约设计。** 它改变问题落在哪个可搜索空间，不等于已经搜索了任意 solver 架构。
- **A2DEPT/BEAM/DGA²D 是结构设计。** 它们才直接涉及组件拓扑、控制流或完整 solver；其中 DGA²D 的信用是 solver 内部调用关系，不是候选的演化谱系。
- **AlphaEvolve/ShinkaEvolve 是通用程序进化 harness。** 它们的代码编辑范围可以很大，但实际任务仍由接口和 evaluator 界定；不能仅凭“修改完整代码”宣称完全开放。
- **AHD Agent/CORAL 是控制策略设计。** 它们主要学习搜索过程中的行动选择，不一定改变最终算法的结构表示。
- **CALM/EvoTune/Hero 是模型内化。** 训练生成器可能降低未来搜索成本，但模型参数更新本身不等于发现了更好的 solver 结构。

因此，论文标题中的 `algorithm design`、`algorithm discovery`、`heuristic generation` 和 `agentic search` 不能直接作为层级标签。应先回答三个问题：输出物是什么？哪一部分发生了变化？部署时是否还要重新搜索或逐实例选择？

### 2.10 机制层面的真正差异

若把论文中的算法名去掉，现有方法主要在下面五个机制上做选择：

1. **表示机制**：固定函数签名、模块图、程序树、完整文件、问题规约或自然语言计划；
2. **变异机制**：局部编辑、组件替换、宏观重写、语义交叉、阶段拼接或从头生成；
3. **保留机制**：精英选择、Pareto/覆盖目标、退火接受、谱系保存、记忆评分或 agent 自主写入；
4. **分配机制**：固定比例、MCTS/UCT、bandit、枝级 Thompson、agent policy 或 RL；
5. **知识机制**：当前种群、代码差分、反思、行为轨迹、领域检索、共享记忆或模型权重。

高层方法的差别通常来自这五个机制的组合，而不是单一“LLM 更强”。例如 A2DEPT 的优势假设来自宏观结构改写加谱系保留加维护闭环；BEAM 的优势假设来自结构/函数分层加记忆；CORAL 的优势假设来自持久共享记忆加多 agent 异步协同。若实验没有拆开这些因素，结论应写成“完整配方在该协议下有效”，不能写成某一个组件已经被证明必要。

### 2.11 不应遗漏的控制、记忆和行为路线

设计对象变大只是一个轴。另一批工作不一定改动 solver 的表示，却改动了“谁获得下一次评价”和“历史如何进入生成”，它们是高层 AAD 的控制基础设施：

| 方法簇 | 具体方法 | 改变的高层变量 | 主要认识 | 主要缺口 |
| --- | --- | --- | --- | --- |
| 路线/树搜索 | MCTS-AHD、Planning of Heuristics、PathWise、CogMCTS、RefineEvo | `controller` 的路线选择 | 把算法改进看成规划或树上的分支扩展 | 树节点价值通常仍是短期分数，缺少路线 continuation value |
| 枝级与预算信用 | Clade-AHD、Compute Allocation/BaSE、RelayEvolve | 分支、模型和调用预算 | 预算应按枝、模型或改进收益动态分配 | 观测窗口短，容易把早期偶然收益当作长期价值 |
| 轨迹与谱系记忆 | DeltaEvolve、PhyloEvolve、MEMOIR、MeLA、MeEvo | 历史的表示和压缩方式 | 代码差分、谱系和元认知摘要比扁平精英更可复用 | 多数只保存或摘要历史，没有检验形成证据是否改变下一步决策 |
| 行为与状态反馈 | BehaveSim、DyACE、Teacher-Aware Evolution | 求解行为、在线状态和局部行动反馈 | 静态代码相似度不能代表实际求解行为；动态控制需要状态感知 | 行为相似度依赖 probe，teacher 反馈不能成为最终目标 |
| 多样性与生态位 | PartEvo、TurboEvolve、SMCEvolve | 分群、采样和接受规则 | 语言空间需要行为/语义 niche 与有效的探索压力 | niche 特征、候选数和 acceptance 往往成套变化 |
| 知识与诊断 | GeoEvolve、Hercules | 上下文、领域检索和代理评估 | 检索应由当前失败或缺口触发，而非静态堆知识 | 知识库边界和检索成本仍由人工决定，代理预测有误差 |

这些方法共同补充了一个容易被忽略的结论：**高层 AAD 的对象不只有 solver，还包括 solver 生成过程的状态表示、路线价值和预算账本。** 如果只用“最终程序是否更好”评价它们，会看不到它们改变的是搜索动力学而不是单个候选的静态质量。

对 TraceAAD 特别相关的三条证据链是：

- `DeltaEvolve / PhyloEvolve / MEMOIR` 说明保存形成关系和代码变化有价值，但不自动证明 LLM 消费了这些关系；
- `BehaveSim / DyACE / Teacher-Aware` 说明实际求解行为和在线状态可以作为高层控制信号，但它们不能替代设计 lineage；
- `Compute Allocation / Clade-AHD / SMCEvolve` 说明搜索预算和 acceptance 规则本身会塑造结果，必须与上下文机制分开计量。

## 3. 代表方法总表

| 方法 | 主要设计对象 | 主要机制 | 已有证据 | 关键边界 |
| --- | --- | --- | --- | --- |
| EoH-S | 启发式集合 | CPI、互补种群、互补父代 | 集合覆盖在所测任务上改善 | oracle 集合分不等于部署性能 |
| InstSpecHH | 实例专属 portfolio + 选择器 | 子类生成、邻域迁移、特征/LLM 选择 | CVRP/OBPP 上实例匹配有效 | 生成收益与邻域迁移混杂，选择成本需计 |
| MoH | heuristic-optimizer / meta-optimizer | 双层搜索、多任务 utility、自调用 | TSP/BPP 跨规模结果较强 | 外层作用缺少固定优化器单变量消融 |
| RedAHD | 问题规约 + 启发式 | LLM 设计实例映射、解映射和模板 | 多类 COP 上端到端可行 | 映射可能过拟合训练实例，结构保真未充分验证 |
| A2DEPT | 完整 solver 程序树 | micro/macro 操作、混合选择、依赖闭合修复 | 标准基准上超过组件级基线 | 复杂度导致无效率上升，跨模块契约难修 |
| BEAM | solver 结构 + 函数实现 | 外层 GA、内层 MCTS、自适应记忆 | 完整 solver 任务上有端到端提升 | 结构/函数层和记忆贡献仍有耦合 |
| DGA²D | 算法组件有向 pipeline | operator graph、first-order credit | 组件组合与信用消融 | 只涉及 solver 内部拓扑，不是演化谱系 |
| AlphaEvolve | 可变范围程序 | 异步档案、patch、反馈 evaluator | 多个算法/系统案例 | scaffold、evaluator 和工程预算高度任务依赖 |
| ShinkaEvolve | 程序档案与搜索 harness | bandit 模型分配、embedding 去重 | 多任务程序演化案例 | 短期收益可能饿死互补模型或新颖分支 |
| EvoStage | 多阶段算法设计过程 | coordinator、阶段反馈、全局/局部算子 | 芯片布局等任务上有效 | 阶段划分与中间指标仍是人工设计变量 |
| AHD Agent | 搜索控制 agent policy | agentic RL、工具调用、环境合成 | 未见任务与较小模型结果 | 更多动作/评估次数与策略收益难分离 |
| CORAL | 多 agent 开放式搜索系统 | 异步 agent、持久记忆、协作 | best-of-n 对照和多任务案例 | 记忆污染、干预策略和长期稳定性待验证 |
| CALM / EvoTune | 生成模型与算法共同演化 | DPO/GRPO、搜索—训练闭环 | 训练阶段相对无训练基线有收益 | 训练成本、数据选择和权重内化作用混杂 |
| Hero | 可复用完整 solver + 模型策略 | GRPO、可行性门控、compile-once | SDS 上跨实例复用 | 任务结构与 scaffold 特异，跨域证据有限 |

## 4. 这个领域实际上在研究什么

### 4.1 搜索空间问题：固定框架是否成为性能瓶颈

固定框架的好处是可执行、易比较、搜索效率高；坏处是把算法范式、控制流和组件拓扑锁死。A2DEPT、BEAM、DGA²D 和 RedAHD 都在不同层次放松这个约束，但放松方式不同：

- RedAHD 改变问题表示；
- MoH 改变生成 heuristic 的搜索器；
- DGA²D 改变组件的组合拓扑；
- BEAM 改变结构与函数的层次；
- A2DEPT 允许完整程序级重写。

因此，“开放程度”不是二元变量，而是一个需要逐层测量的连续轴。搜索空间越大，发现新范式的机会越多，但无效代码、信用噪声和 evaluator exploit 也越多。

### 4.2 搜索控制问题：有限评价预算应该给谁

高层 AAD 不只是生成更大的程序，还必须决定：

- 继续精炼当前强分支，还是启动新架构；
- 给局部函数实现预算，还是给结构重写预算；
- 使用高质量模型，还是使用便宜模型探索；
- 保留低分 stepping stone，还是立即淘汰；
- 何时触发修复、重启、记忆压缩或跨 agent 迁移。

MCTS-AHD、PathWise、Clade-AHD、Compute Allocation、ShinkaEvolve、AHD Agent 和 CORAL 都在研究这个控制问题，只是使用树搜索、bandit、世界模型、RL 或自治 agent 等不同形式。它们共同揭示：在高层 AAD 中，**搜索控制本身已经是算法设计对象**。

### 4.3 记忆和轨迹问题：保存什么，如何改变下一步

目前文献保存历史的方式包括候选档案、谱系树、patch、自然语言反思、代码差分、行为轨迹和持久共享记忆。它们回答的不是同一个问题：

- **当前种群**回答“现在有哪些好程序”；
- **谱系**回答“候选怎样形成”；
- **patch/delta**回答“改动了什么”；
- **反思摘要**回答“下一步可以怎样改”；
- **行为轨迹**回答“程序实际怎样求解”；
- **共享记忆**回答“其他搜索分支曾经发现什么”。

许多方法保留 lineage，却没有把形成路径的具体证据输入下一次生成；另一些方法只存全局精英，丢失局部失败条件和适用范围。这正是 TraceAAD 可以进入的机制空位：研究“轨迹内容是否改变下一次高层设计决策”，而不是只比较档案大小。

### 4.4 可执行性和验证问题：更高层设计能否运行

从固定函数到完整 solver 后，失败类型从语法错误扩展到未定义依赖、类型契约、控制流死路、无解、超时和 evaluator exploit。现有做法包括固定区与可变区、依赖闭合修复、阶段级检查点、分层奖励、便宜预评估和失败惩罚，但尚无统一的程序维护协议。

一个可信的高层 AAD 报告至少应区分：

1. 生成成功率；
2. 编译/解析成功率；
3. 可执行且满足约束的比例；
4. 真正完成 evaluator 的候选数；
5. 在可行候选中的质量；
6. 修复调用与额外 token；
7. 失败候选是否进谱系、记忆或训练集。

只报告最终 best score 会把“设计能力”和“筛选/修复/重试预算”混在一起。

## 5. 共同证据与主要不足

### 5.1 已有较强共识

- **高层对象有真实收益空间。** 多篇工作在匹配协议下显示，集合、优化器、规约、完整程序或控制器能够发现固定单函数模板难以得到的策略。
- **结构化搜索比完全自由生成更可用。** 程序树、双层分解、阶段分解、组件图和约束修复都是在开放性与可执行性之间建立中间层。
- **历史不应只作为平面精英列表。** 谱系、差分、行为、失败经验和共享记忆各自能提供不同类型的证据。
- **评价器是方法的一部分。** 任何“端到端”结果都高度依赖问题表示、检查器、预算和筛选协议。

### 5.2 当前不足

1. **“完整算法设计”通常仍是受限开放。** 接口、输入输出契约、任务分解、evaluator 和运行环境由人类固定，LLM 只在这些边界内改代码。
2. **整套配方有效，机制因果不清。** 表示、提示、选择、修复、记忆和预算通常一起变化；很多论文没有与固定搜索器或等调用数的简单控制器比较。
3. **预算口径不统一。** 有的限制真实 evaluator，有的限制 LLM 请求、token、墙钟时间或训练步数；训练成本和推理成本常被分开报告，难以横向比较。
4. **泛化常被高估。** 跨规模、跨分布、跨问题、跨实例选择、best-of-run 和 portfolio oracle 是不同命题；不少结果只支持其中一个。
5. **高层搜索的信用分配仍然粗糙。** 一次宏观重写同时改动结构、函数和调用顺序，最终分数无法说明哪个决策有效。
6. **失败处理缺少统一统计语义。** 丢弃、惩罚、修复、保留谱系、进入记忆和进入训练集会产生不同的搜索动力学，但主表很少完整披露。
7. **长期自治和安全边界不足。** 多 agent 共享记忆会传播错误，开放式搜索可能优化 evaluator 漏洞；现有案例多是短期或任务特定压力测试。
8. **理论仍落后于系统。** 少数工作对集合覆盖、信用估计或采样过程给出形式化分析，但对“何时应扩大设计层级、何时应保留可靠模板”还没有一般理论。

## 5.3 主张—证据审计：什么才足以证明“更高层设计有用”

将论文主张拆成可检验命题后，最低证据要求如下：

| 主张 | 容易被误判的结果 | 最小充分对照 | 仍需报告的边界 |
| --- | --- | --- | --- |
| 更大的设计空间能找到更好算法 | 开放方法的 best score 更高 | 相同 evaluator、相同模型/调用预算，固定模板与开放空间配对 | 失效率、代码复杂度、修复成本、OOD |
| 外层 optimizer design 有贡献 | MoH 胜过 EoH/ReEvo | 固定内层 heuristic 生成接口，只开关外层元优化 | 外层候选数、内层候选数和多任务聚合方差 |
| 结构搜索优于函数搜索 | A2DEPT/BEAM 端到端分数更好 | 同一初始程序、同一评估预算，允许结构操作与只允许局部操作对照 | 结构跳变次数、有效率和何时开始领先 |
| agentic 控制优于固定循环 | AHD Agent/CORAL 结果更好 | 固定工具集合和总调用数，比较固定策略、随机策略和 agent policy | 工具调用、重试、并行度、停止策略 |
| 记忆/轨迹带来机制收益 | 有记忆版本最终分数更高 | 固定上下文长度和候选预算，比较无记忆、当前精英、形成轨迹和失败经验 | 记忆命中后的后续增益、错误迁移和过期淘汰 |
| 模型内化了算法能力 | 微调后 pass@k 更高 | 与同搜索预算的 frozen model、best-of-k 和 compile-once solver 比较 | 训练算力、数据泄漏、任务族外迁移、灾难性遗忘 |

由此，现有论文的证据大多属于“整套系统在所测协议下有效”，只有少数消融可以支持某个组件的方向性作用。不能把完整方法胜出自动改写成“开放表示本身被证明优于固定表示”。

## 5.4 统一实验协议：高层 AAD 需要比普通 AHD 多报告什么

对高层方法，普通的最终 gap 不足以描述研究结果。建议每个实验路同时记录：

1. **设计空间**：哪些变量可变，哪些变量冻结，是否允许新增模块和改变控制流；
2. **搜索预算**：真实 evaluator、LLM 请求、输入/输出 token、修复调用、训练步数和墙钟时间；
3. **执行层级**：解析、编译、可执行、可行、完成完整 evaluator 的逐级通过率；
4. **结果层级**：训练 best、重复均值、held-out 单个 solver、portfolio oracle、真实 router 和部署运行时间；
5. **行为层级**：代码结构变化、实际决策变化、求解轨迹变化和搜索控制变化；
6. **失败层级**：失败类型、惩罚值、是否重试、是否修复、是否写入谱系/记忆/训练集；
7. **复杂度层级**：代码长度、模块数、依赖数、调用深度和单实例运行成本。

其中第 4 项尤其重要。一个方法在 `portfolio oracle` 上领先，只说明候选集合有潜在覆盖；一个方法在 `frozen solver` 上领先，才说明它产生了可以跨实例直接复用的算法；一个方法在 `router` 上领先，才说明选择策略具备实际部署价值。

## 5.5 贯穿所有方法的五个混杂因素

### 评价预算混杂

高层方法往往需要更多代码生成、修复和真实评价。若只对齐最终 evaluator 次数，却不计修复和筛选，开放方法可能得到更多有效候选；若只对齐 token，又可能使便宜模型占优。至少需要同时给出 evaluator-equivalent、LLM-call、token 和 wall-clock 四种账本。

### 内层收益混杂

MoH 的外层 utility 依赖内层生成多少个候选；EoH-S 的 CPI 依赖集合大小；InstSpecHH 的效果依赖邻域候选数；A2DEPT 的结构搜索可能得到更多尝试机会。高层对象的分数不能脱离内层 `K`、任务数 `N`、路由候选数和重复次数解释。

### 骨架先验混杂

固定框架不是纯粹的限制，也可能包含强领域知识。A2DEPT 在没有可靠模板的任务上可能占优，在结构先验很强的任务上却可能劣于 AHD。应把“模板质量”作为实验因素，至少比较强模板、弱模板和无模板，而不是默认开放搜索总是更科学。

### evaluator 混杂

当设计对象变成完整程序时，evaluator 同时决定合法性、运行时、资源上限和目标质量。一个候选可以通过弱 evaluator 而不是真正改善算法。高层 AAD 必须有隐藏实例、独立检查器、超时和反作弊测试，并把 evaluator 失败与算法失败分开。

### 选择口径混杂

best-of-run、best-of-k、instance-wise oracle、训练集精英和 held-out 单路结果回答不同问题。尤其是逐实例从候选中挑最优的结果，只能证明 portfolio 的潜在上界，不能证明任意单个程序可以泛化。

## 6. 面向 TraceAAD 的研究定位

TraceAAD 不必把目标表述成“再造一个更大的开放式 solver 搜索器”。更有区分度的定位是：**让高层 AAD 使用有证据链的设计轨迹，学习何时改变层级、何时继续精炼，以及哪些历史证据支持该决策。**

这可以形成四个可证伪问题：

### RQ1：形成轨迹是否比最终程序更能指导高层设计

固定候选程序、LLM、evaluator 和总预算，只改变上下文：

- 当前精英代码；
- 精英代码加父代来时路；
- 精英代码加结构化改动—结果—失败原因；
- 全局档案摘要。

指标不仅是 best score，还应包含单步改进率、重复率、无效率、跨分支采纳率和 held-out 性能。该实验直接检验 TraceAAD 的核心机制是否能影响“下一步怎样改”，而不是只影响最终文本。

### RQ2：何时应从局部函数搜索升级到结构搜索

把搜索层级作为可选择动作：局部精炼、组件替换、结构重写、完整 solver 重构。控制器根据停滞、失败类型、结构冲突和历史收益选择层级；与固定层级、随机升级和始终开放三种策略比较。关键是记录升级动作后带来的真实增益和无效成本。

### RQ3：高层设计能否在相同真实评价预算下胜过固定框架

建立三路严格对照：

1. 固定框架 + TraceAAD 设计单函数；
2. MoH 式优化器层搜索；
3. A2DEPT/BEAM 式结构或完整程序搜索。

在相同 evaluator 次数下比较质量、可执行率、代码复杂度、LLM token、修复成本和 OOD 泛化。这样可以回答“扩大设计对象是否值得预算”，而不是只比较三套不同预算的最终分数。

### RQ4：轨迹能否帮助高层搜索复用而不是重复犯错

把历史条目存成可回链记录：`parent → change → execution evidence → outcome → applicability`。分别测试：只存当前程序、存形成轨迹、存失败条件、存跨任务适用范围。报告记忆命中后的真实增益、误迁移率、上下文长度和过期条目淘汰率。

## 7. 建议的未来方向

### 7.1 自适应搜索粒度

当前方法大多预先决定“只改函数”或“直接改完整程序”。更自然的框架是让系统根据证据选择粒度：局部改动连续失败时才触发结构重写；结构重写后回到局部精炼。需要将“层级切换”作为显式决策，并计入独立预算。

### 7.2 轨迹条件的高层元优化

MoH 已经把优化器作为对象，但其上下文主要是候选和 utility。下一步可以让 meta-optimizer 读取结构化轨迹：哪些改动曾成功、哪些只在某类实例上成功、哪些失败是表示问题、哪些失败是实现问题。这样外层优化的 fitness 不再只是最终 utility，而是“继续产生有效改进的能力”。

### 7.3 合约驱动的完整 solver 合成

完整程序搜索需要统一的接口和验证协议：类型/形状契约、资源上限、可行性检查、阶段性指标、依赖闭合和失败分类。未来应把这些合约作为可组合的设计边界，并测量它们减少了哪些失败类型、增加了多少维护成本。

### 7.4 结构搜索的预算分配与信用分配

应把“架构重写”“组件实现”“错误修复”“真实评价”分开计账，并研究 continuation value：某个低分分支是否仍可能产生高价值结构。A2DEPT 的 stepping stone、Clade-AHD 的枝级信用、BEAM 的双层搜索和 DGA²D 的局部转移信用可以放在同一实验框架中比较。

### 7.5 从搜索到可复用 solver 的迁移

模型训练路线应严格区分：

- 训练期产生候选的成本；
- 推理期生成 solver 的成本；
- 固定 solver 在新实例上的执行成本；
- 重新搜索和每实例选择的成本。

“compile-once + held-out”应成为可复用算法主张的最低验证协议。

### 7.6 实例 portfolio 与完整 solver 的统一

算法集合、实例特征选择和完整 solver 合成可以统一为：

```text
候选 solver portfolio
        → 实例特征/运行状态诊断
        → 选择、组合或继续搜索
```

这条路线可能比强行寻找一个对所有实例都好的 solver 更符合 No Free Lunch 约束，但必须把选择器误差和额外执行成本纳入目标。

### 7.7 公平、可复核的高层 AAD 基准

建议基准同时冻结：任务表示、训练/测试实例、真实 evaluator 预算、LLM token/请求、修复调用、种子、超时和失败语义。结果至少分为：训练 best、重复均值、held-out 泛化、固定 solver 复用、portfolio oracle 和实际选择器性能。

基于上述边界提出的具体研究问题、假设、实验预算、停止条件和方向优先级，见 [AAD 高层设计研究方向与实验方案](AAD高层设计研究方向与实验方案.md:1)。当前优先验证决策相关上下文效应与机会分配效应，再扩展到搜索粒度、完整 solver 合成和模型内化。

## 8. 结论

该领域已经形成一条清晰的升级链：

```text
固定框架中的函数设计
  → 启发式集合与实例选择
  → heuristic-optimizer / meta-optimizer 设计
  → 问题规约与组件拓扑设计
  → 完整 solver 程序搜索
  → 搜索控制器与多 agent 自治
  → 将算法设计能力内化到模型参数
```

其中，MoH 的位置是 **优化器层**；RedAHD 是 **表示/规约层**；A2DEPT、BEAM、DGA²D 是 **结构与完整 solver 层**；AlphaEvolve、ShinkaEvolve 和 EvoStage 是 **程序进化与过程组织层**；AHD Agent、CORAL 是 **搜索控制与自治层**；CALM、EvoTune 和 Hero 是 **模型内化与可复用 solver 层**。

目前还不能说“完整 solver 搜索已经取代固定框架”。更准确的判断是：固定框架提供可靠的算法先验和低成本搜索，开放式 AAD 提供打破框架上限的机会；两者之间的选择取决于任务是否存在可靠模板、结构改写是否可验证、预算是否足够以及 held-out 泛化是否真实发生。未来最有价值的研究不是盲目把搜索空间做大，而是让系统依据轨迹证据决定**何时扩大搜索对象、扩大到哪一层、如何维护可执行性，以及怎样证明这次扩大值得它消耗的预算**。

## 9. 主要原始来源

- [MoH: Generalizable Heuristic Generation Through Large Language Models with Meta-Optimization](https://arxiv.org/abs/2505.20881)
- [RedAHD: Reduction-Based End-to-End Automatic Heuristic Design](https://arxiv.org/abs/2505.20242)
- [AlphaEvolve: A Coding Agent for Scientific and Algorithmic Discovery](https://arxiv.org/abs/2506.13131)
- [A2DEPT: LLM-Driven Automated Algorithm Design via Evolutionary Program Trees](https://arxiv.org/abs/2604.24043)
- [BEAM: Bi-level Memory-adaptive Algorithmic Evolution](https://arxiv.org/abs/2604.12898)
- [AHD Agent: Agentic Reinforcement Learning for Automatic Heuristic Design](https://arxiv.org/abs/2605.08756)
- [CORAL: Towards Autonomous Multi-Agent Evolution for Open-Ended Discovery](https://arxiv.org/abs/2604.01658)
- [Beyond Inference-Time Search: Reinforcement Learning Synthesizes Reusable Solvers](https://arxiv.org/abs/2605.18374)
- 本地方法总览：[AAD 方法比较](</home/fang/code/LLM4AD/TraceAAD/docs/05-文献与基线/专题调研/AAD方法比较.md:1>)
- 本地候选表示对照：[LLM 自动算法设计候选表示协议对比](</home/fang/code/LLM4AD/TraceAAD/docs/05-文献与基线/专题调研/2026-09-12-LLM自动算法设计候选表示协议对比.md:1>)
