# 代码前文字与算法表示

[相关工作目录](README.md) · [方法总览](AAD代表方法与机制.md)

多数 LLM-AAD 方法会让模型在代码之外写一段自然语言，名称有 thought、description、reflection、plan、rationale、knowledge 等。这些文字在不同方法里承担的功能不同，有的服务本次生成，有的留给后续算子读取，有的是搜索的主要对象。本文按“文字在哪次调用中产生、写在代码前还是后、是否保存、作者认为它起什么作用”比较各方法，并说明与 TraceAAD 格式实验的关系。

## 文字承担的四种功能

| 功能 | 回答的问题 | 时间方向 | 典型做法 |
| --- | --- | --- | --- |
| 回顾反思 | 之前的候选为什么好或差 | 评价后 | `ReEvo` 的短期与长期反思 |
| 动作前决策 | 这一次具体要怎么改 | 代码生成之前；可以来自另一次调用或独立模块 | `EoH` E2 先找共同主干，`MCTS-AHD` s1 先列有益思想，`AutoMOAE` 的目标分析，`PathWise` 的 κ |
| 持久表示 | 这个算法是什么、如何实现 | 保存于种群或记忆 | `EoH` thought、`MCTS-AHD` description、`PathWise` d、`PAEvo` plan、knowledge-first 方法中的 knowledge |
| 事后描述 | 代码实际实现了什么 | 代码后 | `MCTS-AHD` Thought-Alignment |

许多方法用同一段文字承担其中两种以上功能。最常见的是 `EoH` 式“代码前写一句 thought 并保存”，它同时是本次生成的决策和留给后代的表示，因此原文的消融无法区分收益来自哪一种功能。

## 各方法的做法

| 方法 | 代码前写什么 | 产生方式 | 是否保存文字 | 作者对文字作用的解释 | 相关证据 |
| --- | --- | --- | --- | --- | --- |
| `FunSearch` (2024) | 无 | 同次调用，补全函数 | 否，只存程序 | 程序本身就是搜索表示 | — |
| `EoH` (ICML 2024) | 花括号内一句话 thought；E2 先“找出共同主干思想”，再写 thought | 同次调用，thought 在代码前 | 是，thought 与代码一起进入种群 | 思想与代码是同一启发式的两种表示，二者共同进化 | 仅 OBP 上各 3 次，比较只用代码、只用思想和两者兼用 |
| `ReEvo` (NeurIPS 2024) | 生成器只输出代码 | 反思由另一个 reflector 调用生成 | 长期反思累积保存，不是逐个体的描述 | 反思是 verbal gradient，指出后续搜索方向 | 消融去掉反思后性能下降 |
| `MCTS-AHD` (ICML 2025) | 一句话 design idea；s1 先“列出已有算法中明显有益的思想”，再写 idea | 第一次调用写 idea 与代码，第二次调用依据代码重写描述 | 是，保存事后重写的 ≤3 句描述 | 描述帮助推理，但代码前的描述可能与代码“不相关”，需按代码重新对齐 | 去掉 Thought-Alignment：TSP50 gap 10.66% → 11.64%，KP100 0.059% → 0.061%（原版 10 次，变体 5 次） |
| `LLaMEA` (2025) | 简短描述 | 同次调用 | 主要保存代码 | 由性能反馈驱动代码变异 | — |
| `AlphaEvolve` (2025) | 公开材料以代码编辑为主 | 同次调用，可输出 diff | 核心是程序数据库 | 由评价器驱动程序进化 | — |
| `ShinkaEvolve` (2025) | `<NAME>` 与 `<DESCRIPTION>`（“对改动的描述与论证”），再写 diff | 同次调用 | 记录在补丁元数据中 | 说明改动意图，方便后续读取 | — |
| `AutoMOAE` (TEVC 2026 接收作者稿) | 代码生成前，独立分析目标、双亲优势与改进潜力；生成回复只含代码 | Analyse 模块产出指导信息，再传入代码生成与校验调用；§IV-I 明确分析增加 LLM 调用 | 种群个体是代码；正文未将分析结果列为持久个体字段；初始化的一句话 Idea 单独生成后实例化为代码 | 定向处理质量与运行时间的权衡，避免无方向变异或交叉退化 | §IV-C、图 3、表 II：有／无分析模块的精英种群中当代新个体平均占比为 40.25%／15.25%；并非同次回复中的文字格式消融 |
| `PathWise` (ICML 2026) | 由单独的 policy agent 生成 κ（derivation rationale），world model 再写 ≤30 词 description 与代码 | κ 与代码分属不同调用和不同角色 | 是，节点为 (h, κ, d, P, PM)，κ 作为“derivation logic”展示给后续 policy | 把“选哪些父代、怎样推导”作为规划动作，代码生成是状态转移 | 整体框架对比，未单独消融 κ |
| `PAEvo` (PPSN 2026；Springer 引用年份为 2027) | 结构化 plan：子问题、主要步骤、补充内容；原文未规定统一词数上限 | Planner 先生成／进化 plan，Coder 再据新 plan 与父代 plan–code 对生成代码，分属两次调用 | 是；plan 与代码一一对应，保存、共同选择并共享代码评测所得 fitness | 用 plan 分解实现任务并保存搜索洞见，plan 是主要进化对象 | §5.3、表 3：TSP 三重复比较去掉算法进化、plan 进化、角色指令；“去掉 plan 进化”仍生成 plan，未检验完全去掉代码前规划 |
| `Back to the Beginning` (2026) | knowledge/principle | 先进化 knowledge，再实例化为代码 | 是，knowledge 是主要搜索对象 | 代码只是 knowledge 的可执行实现 | 理论上给出 distortion–compression 权衡；作者承认抽象保真度是主要局限 |

`RefineEvo` 的 planner 选择探索还是开发、检索哪类经验，属于机会分配层面的规划，不改变单次生成的文字格式，见[经验记忆与条件生成](经验记忆与条件生成.md)。

## 三种搜索表示

各方法对“进化什么”尚无共识：

- **以代码为中心**：`FunSearch`、`LLaMEA`、`AlphaEvolve`、`AutoMOAE`。候选以程序表示；`AutoMOAE` 的分析主要提供本次生成的目标指导。
- **文字与代码并存**：`EoH`、`MCTS-AHD`、`PathWise`。认为两种表示互补，后续算子同时读取。
- **以文字为中心**：`PAEvo`、`Back to the Beginning`。`PAEvo` 原文明确将 plan 作为主要进化对象，但仍保存对应代码，后续 Planner 与 Coder 都读取父代 plan–code 对；因此是 plan 主导的共同进化。`Back to the Beginning` 以 knowledge 为主要搜索变量，代码用来检验它。

## 两篇新增原文的核对

以下页码采用 PDF 页序；`PAEvo` 的印刷页码另列于括号中。核对日期为 2026-10-01，依据用户补入论文库的两份 PDF。

### AutoMOAE：目标分析先于代码生成

§III-A/B（PDF 第 3–5 页）明确区分初始化的一句话 Idea、`CrossoverAnalyse`／`MutationAnalyse` 的指导信息与代码。交叉分析比较双亲各目标表现，给出目标 $o^*$ 与优势集合 $S_a,S_b$；变异分析以当前分数与该目标历史最好分数的差距确定改进潜力，并选择目标。这些信息进入生成提示，而交叉、变异及校验提示均要求只返回代码。§IV-I（第 10 页）进一步说明目标分析需要额外 LLM 调用。因此应表述为“独立分析后再生成代码”，不能写成与 TraceAAD 相同的“在一次回复中先写 Analysis 再写 Code”，也不能仅凭摘要将其理解为模型先生成一套新的算子实现。正文规定了分析规则，但未给出分析调用的完整提示模板或分析文本在后续调用中的持久保存机制。

§IV-C（第 7–8 页）、图 3 与表 II 已有分析模块消融。有／无分析模块时，精英种群中当代新个体的跨代平均占比分别为 **40.25%／15.25%**；作者观察到有分析时，在难以继续减少颜色数后转向优化运行时间。这里的指标不是严格超过父代比例，也不是代码重复率。正文未在该表报告独立重复数、置信区间或检验，不能把这两个百分比直接当作已确证的效应大小。该消融同时移除了目标选择与指导信息，因而支持整个分析模块有用，尚不能区分收益来自目标信息、额外调用还是代码前的推理计算，更未比较分析长度和是否保存。

§IV-I、表 VI（第 10 页）也显示成本不同：TSP 上 AutoMOAE 为 138 次调用、158,008 token，EoH 为 49 次调用、40,041 token。固定种群与代数不等于固定 LLM 成本。当前 13 页文件包含正文、参考文献与作者简介；正文引用的 Appendix VI-A–H 没有包含在此文件中，附录算法与实现细节仍未核对。

### PAEvo：持久 plan 主导共同进化

§4、§4.1–4.2（第 5–8 页，印刷页 320–323）与算法 1 给出两次调用：`Planner(T, parent pairs)` 生成 plan，`Coder(T, new plan, parent pairs)` 生成代码。plan 包含子问题、主要步骤与补充内容；只明确要求前两部分，没有规定统一长度，允许补充搜索中形成的建议。plan 与代码一一对应，保存在两类种群中，共享代码的 fitness，下一代选择不同的 plan 及对应程序。这支持“plan 是主要进化对象”，同时说明代码仍是后续决策的输入。

五个算子也有区别：C1 比较父代对并融合，C2 产生尽可能不同的 plan，M1 修改任务分解与关键步骤，M2 把参数变化先写入 plan，再实现代码；R1 保留最好 plan，重新采样代码，避免一次较差实现使好 plan 被忽略。这不同于只把一两句 Design 作为参考卡保存。

§5.3、表 3（第 11–12 页，印刷页 326–327）在 DeepSeek-V3.2、TSP 构造框架、三重复上报告消融。TSP50／100／200 的平均路径长度（越小越好）为：

| 变体 | TSP50 | TSP100 | TSP200 |
| --- | --- | --- | --- |
| 完整 PAEvo | 6.215 | 8.591 | 11.992 |
| 去掉算法进化 | 6.452 | 8.857 | 12.352 |
| 去掉 plan 进化 | 6.740 | 9.401 | 13.197 |
| 去掉角色协作指令 | 6.314 | 8.756 | 12.172 |

“去掉 plan 进化”仍由 Planner 根据父代代码生成 plan，只是不再进化父代 plan；“去掉算法进化”则只进化 plan，Coder 仍为评价其质量生成程序，但不使用算法进化提示。因此结果支持两类进化信息结合的价值，未隔离“先规划”与“保存规划”的全部作用，也没有测试文字长短。表中未报告波动、置信区间或显著性检验，尽管作者使用了 significantly，本文只报告均值差异。§5.1 固定每任务最多 400 个启发式样本，不能据此视为等 token 或等调用预算。

§5.5、表 4（第 13–14 页，印刷页 328–329）的 “w/o heuristic description” 去掉的是任务提示中的外部启发式框架说明，仍保留 plan；TSP50／100／200 从 6.215／8.591／11.992 改善为 5.849／8.165／11.438。作者描述候选改为先构造完整路径、用 2-opt 改进并缓存，再响应逐步选点接口。因此这里还涉及实际求解方式的改变，不能将它读成“算法存档描述没有价值”或 Design 长度的消融。

## 文字与代码不一致

这一问题已被明确提出。`MCTS-AHD` 指出 `EoH` 式“先描述后代码”会因幻觉产生与代码不相关的描述，因此改为由代码事后重写描述；但其 s1 仍保留代码前的“列思想再写 idea”，即保留代码前的决策，同时改为在代码后写存档。`Back to the Beginning` 从更抽象的层面讨论同一问题：同一条 knowledge 可对应性能差别很大的实现，抽象过粗会丢掉决定性能的细节。

目前各方法把不一致当作存档准确性的问题处理。如果代码前文字的作用是本次决策，那么它与最终代码不一致未必有害，只要它不进入存档。

## 与 TraceAAD 格式实验的关系

[格式实验](../03-机制探索与验证/2026-10-01-写代码前的决策与Design格式.md)的结论与文献的对应如下：

| 本地结论 | 文献中已有的部分 | 本地实验补充的部分 |
| --- | --- | --- |
| 修改型算子在代码前先做简短针对性分析，可减少重复、提高改进率 | `EoH` E2、`MCTS-AHD` s1 在生成指令中要求先分析；`AutoMOAE` 用独立目标分析指导代码生成，且已有有／无分析模块消融 | 用同父代配对对照研究同次回复的格式；测了分析长度与内容，以及重复和超父，而非仅观察整个模块的搜索结果 |
| 分析不保存，Design 一两句话即可 | `PathWise` 区分 κ 与 d，但 κ 由单独调用生成且会保存；`MCTS-AHD` 用事后描述替代代码前描述 | Design 长度（16–145 词）、参考卡长度（31–327 词）对后续生成均无可测影响 |
| Explore 不需要分析 | 文献中多数方法对所有算子使用同一格式 | 迁移实验显示分析的价值依赖算子：Explore 只写 Design 名次最好、成本最低 |
| 长 Idea 被当作草稿本、与代码不一致 | `MCTS-AHD` 已指出代码前描述可能与代码不相关 | 观察到草稿式推敲反而压低重复，说明不一致的文字仍可能具有决策功能 |

因此，“写代码前先分析或规划”本身不能作为 TraceAAD 的新意，`AutoMOAE`、`PAEvo` 与 `MCTS-AHD` s1 已有不同实现，而且前两者都有相关组件消融，不能再概括为“已有工作只评价整体框架”。在本次核对范围内，仍未找到在固定上下文和可比成本下，同时区分本次决策、后续读取、长度与算子差异的对照。本地实验只在单次生成层面给出初步答案；短 Design 的长度无可测影响，不能推广为 `PAEvo` 的结构化 plan 无用，完整搜索与 held-out 尚未验证。

一个可检验的推论是 Analysis → Code → 事后 Design：代码前保留针对性分析，存档由代码事后生成，以兼顾决策与存档准确性。代价是每次多一次调用；收益取决于事后 Design 是否比代码前 Design 更准确，以及更准确的存档是否改善后续生成。由于本地实验中 Design 长度和参考卡长度都没有可测影响，后一点的先验并不强。这只是待检验的假设，不是已采用的协议。

## 资料来源

- `FunSearch`：[Nature 2024](https://www.nature.com/articles/s41586-023-06924-6)，本地 `papers/Mathematical_discoveries_from_program_search_with_large_language_models/`
- `EoH`：[arXiv 2401.02051](https://arxiv.org/abs/2401.02051)，E2 提示见 `reference_code/EoH/eoh/src/eoh/eoh/evolution.py`
- `ReEvo`：[arXiv 2402.01145](https://arxiv.org/abs/2402.01145)，生成器提示见 `reference_code/ReEvo/prompts/common/`
- `MCTS-AHD`：[arXiv 2501.08603](https://arxiv.org/abs/2501.08603)，本地 `papers/MCTS-AHD/icml2025.tex`（Thought-Alignment 见第 251 行与附录，s1 提示见附录 Action s1，消融见第 436 行）
- `LLaMEA`：[arXiv 2405.20132](https://arxiv.org/abs/2405.20132)
- `AlphaEvolve`：[arXiv 2506.13131](https://arxiv.org/abs/2506.13131)
- `ShinkaEvolve`：[arXiv 2509.19349](https://arxiv.org/abs/2509.19349)，diff 格式见 `reference_code/ShinkaEvolve/shinka/prompts/prompts_diff.py`
- `AutoMOAE`：*AutoMOAE: Multi-Objective Auto-Algorithm Evolution*，[OpenReview 原稿](https://openreview.net/forum?id=G8tP1Z9dLy)、[IEEE TEVC 正式入口](https://doi.org/10.1109/TEVC.2026.3737437)。2026-10-01 已依据用户提供的 [本地 PDF](../../../papers/AutoMOAE_Multi_Objective_Auto_Algorithm_Evolution/paper.pdf)核对方法与消融；文件为 13 页的 TEVC 接收作者稿，并非 ICLR 录用证明。证据位置：§III-A/B、算法 1（第 3–5 页），§IV-C、图 3、表 II（第 7–8 页），§IV-I、表 VI（第 10 页）。正文引用的附录不在此 PDF 中，仍待补充。
- `PathWise`：[arXiv 2601.20539](https://arxiv.org/abs/2601.20539)，本地 `papers/PathWise/example_paper.tex`（节点定义见第 174 行，动作定义见第 185 行，world model 提示见附录）
- `PAEvo`：*PAEvo: Plan-Algorithm Evolution with LLMs for Automatic Heuristic Design*，[Springer 出版页](https://link.springer.com/chapter/10.1007/978-3-032-36217-9_20)，作者 Zhiyuan Chu、Zikang Yu、Jiahai Wang、Jinbiao Chen、Zizhen Zhang。PPSN 2026，LNCS 16989，316–331 页；首次上线为 2026-08-25，Springer 建议引用年份为 2027。2026-10-01 已依据用户提供的 16 页 [本地 PDF](../../../papers/PAEvo_Plan_Algorithm_Evolution_with_LLMs_for_Automatic_Heuristic_Design/paper.pdf)核对表示、调用流程、算子和消融：§4.1–4.2、算法 1（第 6–8 页），§5.3、表 3（第 11–12 页），§5.5、表 4（第 13–14 页）。论文说明了角色和算子行为，但没有给出完整提示模板。
- `Back to the Beginning`：[arXiv 2605.06123](https://arxiv.org/abs/2605.06123)，本地 `papers/Back_to_the_Beginning_of_Heuristic_Design_Bridging_Code_and_Knowledge_with_LLMs/paper.pdf`
