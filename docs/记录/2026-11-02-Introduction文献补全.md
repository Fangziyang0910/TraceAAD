# Introduction 文献补全：改进经历用于哪些决策

核对日期：2026-11-02，北京时间。稿件：[paper.md](../paper.md)。本次从本地论文库补入 21 篇文献，参考文献由 10 篇增至 31 篇，中英文同步引用。原有 [1]–[10] 的编号和条目保留，避免改变 `method.md` 等文档的引用含义。

## 补充后的认识

我按引言的论点选择文献：自动算法设计的背景；LLM 与程序演化的结合；搜索机会和计算预算的分配；历史上下文、反思及经验复用。新增条目都在中英文正文中有对应论述。

核对后，选择机制不能再笼统概括为只使用当前适应度。QUBE 使用直接子代成绩，Clade-AHD 汇总后代评价，ShinkaEvolve 将当前表现与已有子代数量用于选择，BaSE 则在完整演化运行之间分配预算。这些工作支持“以历史表现控制搜索投入”的进展。引言的局限因此限定在数值选择依据没有直接表达具体改法及反馈关系，而非未使用历史。

经验学习也已有多种组织方式。HiFo-Prompt 维护设计原则记忆；EvoPH 和 MeLA 依据反馈修改生成提示；MEMOIR 将分支内改进记录与跨分支经验总结分开。它们与原有的 PhyloEvolve 一起说明，轨迹记忆、经验凝练和反馈循环都已有先例。我保留这些进展，将 TraceAAD 的叙事集中于候选的祖先路径与已有子代尝试如何支持预算分配、算法设计和经验凝练。

上述文献核对支持机制与研究进展的描述，不构成 TraceAAD 的性能证据。贡献段及实验结果占位保持原样。

## 新增文献与正文论点

以下链接均指向实际读取的本地原文。TeX 以行号定位，PDF 以页码和章节定位。机制判断来自原文；作者、年份与发表信息另结合本地书目记录核对。

| 编号／工作 | 本地原文定位 | 引言中的用途与表述边界 |
| --- | --- | --- |
| [11] Automated Design of Metaheuristic Algorithms: A Survey | [main.tex](/home/fang/code/LLM4AD/papers/Automated_Design_of_Metaheuristic_Algorithms_A_Survey/main.tex:66)，摘要与引言 | 支持 AAD 减少人工设计及专家知识依赖的背景。 |
| [12] A Systematic Survey on Large Language Models for Algorithm Design | [manuscript.tex](/home/fang/code/LLM4AD/papers/A_Systematic_Survey_on_Large_Language_Models_for_Algorithm_Design/manuscript.tex:324)，摘要与分类 | 支持 LLM 算法设计的研究范围；不将综述的分类当作本文机制的有效性证据。 |
| [13] Toward Automated Algorithm Design | [main.tex](/home/fang/code/LLM4AD/papers/Toward_Automated_Algorithm_Design_A_Survey_and_Practical_Guide_to_Meta_Black_Box_Optimiz/main.tex:54)，摘要 | 补充自动选择、配置、操作与生成算法的总体背景。 |
| [14] Evolution through Large Models | [main.tex](/home/fang/code/LLM4AD/papers/Evolution_through_Large_Models/main.tex:103)，摘要 | 支持用 LLM 构造程序演化算子。原文示例不等于本文组合优化任务的直接验证。 |
| [15] Evolving Code with a Large Language Model | [ELM_GP_jrnl_2023.tex](/home/fang/code/LLM4AD/papers/Evolving_Code_with_A_Large_Language_Model/ELM_GP_jrnl_2023.tex:201)，摘要 | 支持通过提示调用 LLM 实现代码演化算子。 |
| [16] Understanding the Importance of Evolutionary Search | [0-abstract.tex](/home/fang/code/LLM4AD/papers/Understanding_the_Importance_of_Evolutionary_Search_in_Automated_Heuristic_Design_with_L/0-abstract.tex:1)，多方法、多模型评测 | 支持搜索策略影响算法发现效果；不据此声称越复杂的搜索越好。 |
| [17] HSEvo | [aaai25.tex](/home/fang/code/LLM4AD/papers/HSEvo/aaai25.tex:193)，摘要；真实标题和作者在第 146–151 行 | 作为组合优化中 LLM 与演化搜索结合的代表工作；不采用模板中的示例作者。 |
| [18] MEoH | [MEoH.tex](/home/fang/code/LLM4AD/papers/MEoH/MEoH.tex:144)，摘要 | 支持组合优化算法发现的进展；该方法同时考虑质量与其他设计指标。 |
| [19] LACE | [论文 PDF](/home/fang/code/LLM4AD/papers/LACE/s42256-026-01307-8.pdf)，第 1 页 | 支持组合优化中的互补启发式组合设计。PDF 已列 DOI 和接受日期，但上线日期仍为空，不填具体出版日或卷页。 |
| [20] ShinkaEvolve | [03_method.tex](/home/fang/code/LLM4AD/papers/ShinkaEvolve/sections/03_method.tex:9)，父代选择 | 支持以当前表现和子代数量分配搜索机会。模型集成的 bandit 与父代选择是不同机制。 |
| [21] QUBE | [main.tex](/home/fang/code/LLM4AD/papers/QUBE_Enhancing_Automatic_Heuristic_Design_via_Quality_Uncertainty_Balanced_Evolution/main.tex:265)，§3 | 支持以子代质量和不确定性选择父代。统计单位为程序簇，质量项是直接子代的绝对成绩均值，不是父子提升量。 |
| [22] Clade-AHD | [main.tex](/home/fang/code/LLM4AD/papers/Clade-AHD_Clade-level_Selection_for_MCTS_in_Automatic_Heuristic_Design/main.tex:167)，摘要及方法 | 支持将后代评价汇总为分支的概率估计并控制探索；不是让模型直接阅读语义修改轨迹。标题按原文的 Beyond the Node 记录。 |
| [23] BaSE | [bandit.tex](/home/fang/code/LLM4AD/papers/Compute_Allocation_in_Evolutionary_Search_From_Depth_Breadth_to_Multi_Armed_Bandits/bandit.tex:48)，算法 1 及其后说明 | 支持在并行演化运行间分配 LLM 调用预算。每个臂是完整运行，不将其表述为父代选择算法。 |
| [24] Planning of Heuristics | [example_paper.tex](/home/fang/code/LLM4AD/papers/Planning_of_Heuristics_Strategic_Planning_on_Large_Language_Models_with_Monte_Carlo_Tree/example_paper.tex:114)，摘要与方法 | 支持以 MCTS 规划修改、将反思建议作为搜索动作。会议模板不作为已录用证明。 |
| [25] HiFo-Prompt | [templateArxiv.tex](/home/fang/code/LLM4AD/papers/HiFo_Prompt_Prompting_with_Hindsight_and_Foresight_for_LLM_based_Automatic_Heuristic_Des/templateArxiv.tex:161)，Hindsight 与 Insight Pool | 支持从精英候选凝练原则、维护记忆并用于生成。原则池不等于候选祖先链。 |
| [26] EvoPH | [iclr2026_conference.tex](/home/fang/code/LLM4AD/papers/Experience-Guided_Reflective_Co-Evolution_of_Prompts_and_Heuristics/iclr2026_conference.tex:231)，Prompt Evolution | 支持用表现反馈更新提示和选择演化策略。不将这种提示更新表述为模型参数训练。 |
| [27] MeLA | [main.tex](/home/fang/code/LLM4AD/papers/MeLA_Metacognitive_LLM-Driven_Architecture_for_Automatic_Heuristic_Design/main.tex:199)，Iteration Steps 与 Prompt Evolution | 支持依据思考记录、执行错误和适应度反思生成提示。其历史不局限于单次父子比较。 |
| [28] MEMOIR | [论文 PDF](/home/fang/code/LLM4AD/papers/Memory_Guided_Tree_Search_with_Cross_Branch_Knowledge_Transfer/paper.pdf)，第 1 页摘要、§3.2 | 支持分支内改进记忆、分支结束后的反思及跨分支经验共享。不将其归入“只看一次比较”的局限。 |
| [29] LLaMEA-SAGE | [论文 PDF](/home/fang/code/LLM4AD/papers/LLaMEA_SAGE_Guiding_Automated_Algorithm_Design_with_Structural_Feedback/paper.pdf)，第 1 页摘要与方法 | 支持将程序结构和性能的关联转化为语言修改指导；其结构特征不等于演化祖先关系。 |
| [30] Code Evolution Graphs | [main.tex](/home/fang/code/LLM4AD/papers/Code_Evolution_Graphs_Understanding_Large_Language_Model_Driven_Design_of_Algorithms/main.tex:159)，摘要 | 支持分析演化过程中代码结构的变化。它是分析工作，不据此声称已实现本文的轨迹控制机制。 |
| [31] CALM | [main.tex](/home/fang/code/LLM4AD/papers/CALM/main.tex:92)，摘要 | 支持以生成算法的表现为反馈，通过强化学习训练生成模型。与提示演化分别表述。 |

## 书目信息的本地依据

题名和作者优先按正文首页记录。DOI、arXiv 编号与正式发表信息通过下列本地材料核对；本次没有联网检索，也没有在线测试链接。

- [11] 原文首页给出 TMLR、2024 年和 OpenReview 标识 `qhtHsvF5zj`。
- [12] 正文与本地 PDF 对应 `arXiv:2410.14716`；[论文库清单](/home/fang/code/LLM4AD/papers/LLM_优化求解_Paper_List_manifest.json)给出 DOI `10.1145/3787585`；[Fine-tuning 论文书目](/home/fang/code/LLM4AD/papers/Fine-tuning-LLM-Automated-Algorithm-Design/references.bib)及 [Instance-Specific 论文书目](/home/fang/code/LLM4AD/papers/LLM_Driven_Instance_Specific_Heuristic_Generation_and_Selection/main.bib)给出 *ACM Computing Surveys*, 2026。未补写未统一确认的期号和文章号。
- [13] 论文库清单给出 DOI `10.1109/TEVC.2025.3568053`；Fine-tuning 论文书目给出 *IEEE Transactions on Evolutionary Computation*, 2025。
- [14] [QUBE 书目](/home/fang/code/LLM4AD/papers/QUBE_Enhancing_Automatic_Heuristic_Design_via_Quality_Uncertainty_Balanced_Evolution/custom.bib)与 [AlphaEvolve 书目](/home/fang/code/LLM4AD/papers/AlphaEvolve/literature.bib)均给出书章 DOI `10.1007/978-981-99-3814-8_11`、2024 年及 331–366 页。部分旧条目记为 2023 年，正文采用带完整 DOI 的版本。
- [15] AlphaEvolve 书目给出 DOI `10.1007/s10710-024-09494-2`、期刊名、卷 25、文章 21 和 2024 年。
- [16] [组合优化综述书目](/home/fang/code/LLM4AD/papers/Large_Language_Models_for_Combinatorial_Optimization_A_Systematic_Review/references.bib)给出 PPSN XVIII、2024 年、185–202 页及 DOI `10.1007/978-3-031-70068-2_12`；ReEvo 与 HiFo-Prompt 的书目也记录了同一正式会议版本。
- [17] [HiFo-Prompt 书目](/home/fang/code/LLM4AD/papers/HiFo_Prompt_Prompting_with_Hindsight_and_Foresight_for_LLM_based_Automatic_Heuristic_Des/references.bib:643)给出 AAAI 2025、卷页和 DOI。保留原文第一作者完整姓名，避免擅自拆分姓名顺序。
- [18] [AlphaEvolve 书目](/home/fang/code/LLM4AD/papers/AlphaEvolve/literature.bib:1013)给出 AAAI 2025、卷页和 DOI。
- [19] DOI、题名和作者直接来自 Nature Machine Intelligence PDF 首页。
- [20]、[21]、[24]、[25]、[30] 的 arXiv 编号与题名由论文库清单和对应原文互相核对。Code Evolution Graphs 的 TeX 含旧会议模板字段，未采用这些字段推断发表信息。
- [22] 使用同目录 `2602.00549v2.pdf` 和原文题名；[23] 使用同目录 `2605.29268_src.tar.gz` 及原文首页。BaSE 第一作者是 Sixue Xing。
- [26] [MeEvo 书目](/home/fang/code/LLM4AD/papers/MeEvo_Metacognitive_Evolution_for_Automatic_Heuristic_Design/reference.bib:698)与 RefineEvo 书目均记录 `arXiv:2509.24509`。源文件采用 ICLR 模板，不能仅据此标为 ICLR 录用论文。
- [27] [MeEvo 书目](/home/fang/code/LLM4AD/papers/MeEvo_Metacognitive_Evolution_for_Automatic_Heuristic_Design/reference.bib:58)给出 *Expert Systems with Applications*, 330:133022, 2026，DOI `10.1016/j.eswa.2026.133022`；同一书目还记录原始预印本 `arXiv:2507.20541`。机制描述按本地原文核对，引用采用已记录的期刊版本。
- [28]、[29] 的 PDF 首页分别明确标示 `arXiv:2605.17539v2` 与 `arXiv:2601.21511v1`，均为 2026 年。
- [31] [Fine-tuning 论文书目](/home/fang/code/LLM4AD/papers/Fine-tuning-LLM-Automated-Algorithm-Design/references.bib:13)明确记录 ICLR 2026 与 OpenReview 标识 `x6bG2Hoqdf`，论文库清单记录同一链接。

原有 [1]–[10] 沿用此前核对结果；有关轨迹、选择与反思的详细边界见[祖先与子代上下文核对](2026-10-30-祖先与子代上下文文献核对.md)、[预算分配核对](2026-10-30-轨迹预算分配文献核对.md)及[Introduction 元评审记录](2026-11-02-Introduction元评审.md)。

## 纳入范围与一致性检查

本次还筛读了 CogMCTS、MeEvo、BEAM、DGA²D、RedAHD 等原文。前三篇的本地材料尚未补齐可确认的独立书目标识；DGA²D 的主要图结构描述求解器组成，不宜直接归为演化轨迹；RedAHD 的归约机制可用于后续 Related Work，本次没有为增加数量而改写引言主线。

已检查：31 个编号连续且唯一；31 篇均在中文和英文正文中实际引用；两种语言引用集合一致；没有缺失条目或仅列入书目但未引用的文献。原有 [1]–[10] 条目和双语贡献段保持不变。新增内容未改变结果待确认的状态。
