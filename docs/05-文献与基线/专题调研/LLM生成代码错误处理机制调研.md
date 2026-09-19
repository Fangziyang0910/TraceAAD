# LLM 生成代码的错误处理机制调研

调研问题：AAD/AHD 各方法中，LLM 生成的算法代码发生**解析错误**（提取不出代码、签名不符、格式错）或**评估错误**（运行异常、超时、输出非法值）时，论文给出什么处理方式。依据 `papers/` 库 38 篇论文原文逐篇核对，每项结论附原文定位。

## 核心判断

1. **错误处理在文献中系统性欠披露，但“未披露不等于丢弃”。** 38 篇中约半数对“生成代码评估失败怎么办”零着墨或仅一笔带过（如 EoH-S、HiFo-Prompt、RoCo、Hercules、MCTS 系多篇）。必须严格区分四种证据状态：**“原文明确”**、**“代码确认”**、**“从流程推测”**与**“未知”**。未明确披露处理流程的方法只能归入“未知”或“从流程推测”，不能武断推断为“默认丢弃”。
2. **重试与修复改变有效候选分布，绝非“零机制成本”。** 解析容错与限次报错重试即便没有调用主 evaluator，也直接消耗了 LLM 调用次数、Token 预算与生成墙钟。更重要的是，重采样或带报错回喂的修复使最终送入评估的候选转变为**条件修复后的有效提议分布**，而不是无偏的原始采样。因此它构成了实质性的生成机制改变，绝非零成本的工程附属层。
3. **错误信息属于环境观察或提示上下文，不是天然属于算子 $o_t$。** 必须保持严密的形式化分层：选择算子（如 Refine, Pivot, Fuse, 或执行 Repair）是控制器的**动作（Action）**；执行代码后捕获的 traceback 或异常类型是环境给出的**观察（Observation）**；将报错信息结构化组装并放进下一次请求是**上下文构造（Context Construction）**。混淆动作与观察会导致状态空间与动作空间定义紊乱。
4. **保留失败不等于把所有失败塞进 prompt：六类失败状态的显式解耦。** 事实库应当分别记录：解析失败（Parse Error）、运行错误（Runtime Error）、超时（Timeout）、不可行（Infeasible）、有效但低分（Valid but Suboptimal）、重复候选（Duplicate）。分别记录后，“是否进入生成上下文”、“是否影响选父权重”、“是否参与 credit/bandit 质量统计”，是三个完全正交的下游控制决策。
5. **预算口径与计账边界必须透明。** 修复调用是计入 LLM 预算还是评价预算、失败评估是否扣减主预算，必须清晰分账（如 TurboEvolve 失败全额计入 N_eval，A2DEPT 修复计入全局 LLM 预算，AHD Agent 评价预算只算实际执行）。跨方法比较时，不能让未披露的重试成本破坏评价公平性。

## 一、各家做法总览

生成物列标注失败面大小：单函数 < 组件 < 完整求解器。

### 奠基与种群线

| 方法 | 无效处置 | 错误信息的去向 | 证据强度 |
|---|---|---|---|
| ELM | 隐式丢弃（diff 不可应用/程序不可运行即无个体） | valid/runnable 比例被测量并作为微调变异模型的收益指标 | 无效率是结果变量，处置是实现默认 |
| LMX | 丢弃（"parsed 或 raised exception 即 discarded"） | validation rate 是三个主报告指标之一 | 消融充分（签名强制/父代数/模型规模） |
| Evolving Code/ALFAECLLM | 算子级默认回退：解析失败回退父代、选择失败退随机 | 错误率按算子×方法做图分析并归因 | 正式分析对象，但无策略比较 |
| FunSearch | 丢弃（超时/超内存/非法输出不进 programs database） | 无；防错靠骨架+空函数头补全收窄失误面 | 主文一句话规则（本地无补充材料） |
| EoH | 丢弃（"added to population if feasible"） | 无；prompt 固定签名+禁随机+禁多余解释 | 一个条件从句 |
| MEoH | 继承 EoH | 附录图注承认 illegal code 导致种群空白 | 图注级 |
| HSEvo | 丢弃 | 评估时限 50/100s 显式化（源码注释：应对 infinite loops） | 参数表一行 |
| QUBE | FunSearch 式丢弃（exceptions or timeouts 不保留） | UIQ 不受污染（质量项只由保留样本构成），但无效 offspring 对质量证据的稀释未讨论 | 继承性实现细节 |
| EoH-S | 未知（未提及） | 未提及 | **未知**（正文完全未提及） |
| CDEoH | **反思修复**：$h' \sim R_{LLM}(h, e)$，报错+thought+code 回喂，预算 B 内重试 | 错误信息是修复 prompt 的输入 | **原文明确**（有专门消融） |
| MoH | 容错下放给生成的优化器代码（try/except 跳过+全批无效回退现有最优） | 经 seed optimizer 与 prompt 隐式传递给后代 | **代码/附录明确**（附录示例代码） |
| ReEvo | 丢弃（父代从 successfully executed 中选） | 无；反思只消费成功个体的优劣对比，7 个 prompt 模板无报错字段 | **原文明确**（两处一句话规则） |
| Hercules（Efficient Heuristics Generation） | 未知（未提及代码错误处置） | 火力在评估端：代理预测置信度分层+不可信回退真实评估 | **未知**（代码有效性未提及） |

### 树搜索与分配线

| 方法 | 无效处置 | 错误信息的去向 | 证据强度 |
|---|---|---|---|
| MCTS-AHD | 从流程推测（节点定义为 executable 实现，失败 reward 未定义） | 无 | **从流程推测**（未披露具体处置细节） |
| PoH | 从流程推测（early stopping 剪低分路径，无效处置未定义） | 无 | **从流程推测**（写明提取机制，处置未提及） |
| CogMCTS | 从流程推测（继承 MCTS-AHD） | 负知识库 $K^-$：无改进经验入库为 avoidance cues（性能失败，非运行错误） | **从流程推测**（负知识有消融，运行错误处置未知） |
| Clade-AHD | 未知（失败 outcome 未定义，α/β 由归一化分数驱动） | Beta 信念形式上为失败计数预留 β 通道，未启用；失败若映射 0 分与"极差但可运行"混同 | **未知**（结构预留但未实现） |
| PathWise | 从流程推测（argmax rollout 入图，其余含无效者只作备位；全部消耗预算） | worst-vs-best critic 把当步最差 rollout 转为语言反馈 | **从流程推测**（critic 有消融，无效处理未提及） |
| Compute Allocation/BaSE | invalid 显式映射 fitness 0.0 | 0 分 pull 改变 bandit 臂估计，预算转离停滞轨迹；无效段长度进案例研究 | **原文明确**（唯一把无效当统计对象并进分配决策） |

### 反思与记忆线

| 方法 | 无效处置 | 错误信息的去向 | 证据强度 |
|---|---|---|---|
| HiFo-Prompt | 未知（正文未提及，注释稿暴露 EoH 式 null 过滤） | 记忆只存成功精英的规律 | **未知**（正文完全未提及） |
| MeLA | **报错回喂修复**：错误 prompt 回喂，至多 M 次重试，最优有效候选替换 | 错误历史进元认知 prompt（"avoid the errors"）；附 12 条典型错误清单 | SR 表佐证（EoH 53–89%、ReEvo 41–96%、MeLA 93–99%），无关停对照 |
| EvoPH | 失败赋大负值，隐性淘汰出精英 | 失败→分析报告入经验库；演化后的 prompt 自建 `error_reason`/traceback 字段与"先纠错后优化"分层指令 | 有可执行率图（70–80% vs 20–45%）+消融 |
| MeEvo | 执行失败 $f(h)=\infty$，排除出父代池（可行父代<2 则跳过该代）；COR=2 次修复，无错且更优才替换 | ERR 是跨代共享历史四元组之一，驱动收敛诊断与"勿重复致错策略"约束 | 有伪代码，修复未单独消融（作者自认） |
| RefineEvo | 未定义 | validity rate 作为算子统计触发算子精化；负经验=性能退化，无报错字段 | 负经验库有消融，错误处理是触发器之一 |

### 现代系统线

| 方法 | 无效处置 | 错误信息的去向 | 证据强度 |
|---|---|---|---|
| AlphaEvolve | 评估级联：小规模预测试过滤 faulty program；LLM 反馈可整解丢弃 | 提示渲染执行结果（隐式反馈）；无修复循环 | 实现细节（传闻的 build/repair 级联在白皮书中不存在） |
| ShinkaEvolve | **评估前限次 patch 重采样**：解析反馈（Reflexion）回喂，上限 3（MoE 任务 10） | archive 每程序存文本反馈（可含错误信息）；知识提炼只用成功样本 | 方法节一句+超参表，无消融 |
| DeltaEvolve | 评估器层映射：违约 0 分、NaN/超时惩罚分、两阶段协议 Stage 1 作 validity filter | 失败改动编码为 "Degraded" delta 进入历史上下文 | 评估器附录规则 |
| PhyloEvolve | **最完整修复级联**：语法/类型检查→沙箱执行→根因诊断→最小修复→3 次重试→回滚或上报 Designer 重构 | 失败路径在树中显式标记+错误摘要作负例；selective failure retention 保留信息量最大的失败 | 具名方法小节，无消融 |
| TurboEvolve | OpenEvolve 可行性检查（编译/运行/超时/硬约束）→丢弃 | validity rate 按生成排名报告（头名更高）；失败执行全额计入 N_eval | 唯一报 validity 曲线+预算口径显式 |
| SMCEvolve | invalid=最差奖励（0 分），MH 接受概率随退火温度连续压缩，拒绝时保留父代 | 被拒（含无效）提案留在全量历史作灵感来源 | 形式化核心，invalid 映射只在任务附录 |

### 系统级合成线

| 方法 | 无效处置 | 错误信息的去向 | 证据强度 |
|---|---|---|---|
| RedAHD | 逐问题解检查清单验证+丢弃；LR 有效性由下游解是否合法定义 | 删最易错算子 E1；规约与代码拆两次调用防幻觉 | 附录实现细节 |
| BEAM | **组件级修复**：MCTS 逐函数填充循环内嵌 Fix（traceback 回喂，max_fix_try=3，低温 0.7） | 修复下沉到单函数粒度；生成物自带 MAX_TIME 超时协议 | 算法伪代码内，无消融 |
| A2DEPT | **依赖闭合修复循环**（静态可判定、保证终止，预算耗尽记 0 分）+死代码裁剪；评估失败 $-\infty$ 但仍插入搜索树 | 失败个体进树保留谱系；non-executable rate 随复杂度上升的曲线+失败案例解剖 | 核心贡献之一，无 w/o-maintenance 消融 |
| EvoStage | 阶段级检查点：中间指标反馈纠正设计方向；失败 run 计为失败自然淘汰 | 中间反馈只消费性能指标（wirelength/overflow），不消费报错；组件分工+coder 低温 0.2 防语法错 | Pass Rate 进主表（78%/89% vs EoH 23%、AlphaEvolve 25%） |
| DyACE | 失败代 retry（次数与预算未展开） | 解耦诊断 agent 消费轨迹特征压幻觉 | 附录一句话 |
| Evolutionary RL Algorithms | 训练崩溃经 max-return+clip 评价口径软性吸收进 0 分下限 | **Runtime Errors 槽位进变异 prompt**（"justified by the failure patterns"）；NaN/Inf 禁令+稳定性指标（gradient norm）回喂 | prompt 附录级设计 |
| AlgoPilot | 奖励塑形防错：TLM 对无算法模式的轨迹给负奖励 | 轨迹 LM 用随机（多数无功能）程序轨迹做算法样态先验 | 防错前移到生成过程内部 |

### 智能体与错误专题

| 方法 | 无效处置 | 错误信息的去向 | 证据强度 |
|---|---|---|---|
| AHD Agent | **分档惩罚进 RL reward**：提取失败 -2.0、执行失败/不可行 -1.5、可行得改进量 | 错误作为环境 observation，修复是策略在多轮 revise 中的涌现行为 | reward 设计核心，档位无消融 |
| CORAL | 五状态判定（improved/baseline/regressed/crashed/timeout），后两者 null score | 文件级操作消解解析层；失败 attempt 全量入共享记忆；本地测试在消耗评估预算前拦截编译失败；"what NEVER worked" 防重访 | 失败入记忆有消融，状态机是实现细节 |
| RoCo | 未知（未提及代码级无效处置） | critic 只反思性能回退（"avoid..."反馈进角色记忆），代码级无效不在职责内 | **未知**（正文完全未提及） |
| Experience Memory Graph | 离线编译：失败轨迹与专家轨迹图匹配→最短编辑路径→条件纠正规则，单次执行零试错 | 错误知识结构化存储、检索复用、跨任务迁移 | 核心机制（小模型上 53.6% vs 迭代反思 27–39%）；前提是失败+专家成对轨迹 |
| Where LLM Agents Fail | AET 五模块错误分类+关键错误检测（最早可翻转成败的步骤）+限次迭代调试 | 级联失败是可靠性主瓶颈；action/system 模块与 AAD 的签名/运行错误同构 | 核心机制，但实验全部在通用 agent 任务，未用于 AAD |
| AutoSND | 失败候选排除出 parent；三状态执行证据全量保留 | 负例按劣质/慢/无效三分，与 Pareto 正例做结构频率对比，编译成 avoid/bound 约束在前端压无效率 | 核心机制（去结构引导 validity 98%→76.5%）；validity 99% vs 对照 23.6% |
| DGA2D | 四层验证门（AST/签名/隔离运行/组合冒烟）入池；失败拿最低 fitness | 失败沿有向 walk 回传为算子/实现/边的负信用，最低信用点触发带 `{diagnostic}` 的定向修复 | 信用分配有消融，验证门是实现细节 |

### 分析与证伪补充

- **LLaMEA 的修复循环在本库不可考。** 库内没有 LLaMEA 原论文；LLaMEA-SAGE（含原作者）只有 1 小时评估上限，唯一痕迹是同团队 Code Evolution Graphs 的一句"guided by ... error correction"转述，无轮数与预算。BEAM 声明其 Fix 继承 LLaMEA，是另一个方向的间接证据。
- **MWV**（Mutation Without Variation）披露最完整的重试规格：失败原因回喂+每模型 5 次重试+fallback 模型再 5 次；GPT-5 Mini 需提到 15 次才能跑完 300 步链，作者自认修复过程会贡献虚假变异、污染算子偏置测量。
- **Understanding the Importance of Evolutionary Search** 批判了 baseline 强度、LLM 依赖、重复检测三个方法论问题，唯独漏掉无效样本偏差；参数表里有"应对 invalid heuristics（如死循环）"的 50/20 秒时限。
- **REx**（Code Repair with LLMs，NeurIPS 2024）把修复形式化为以随机反例为条件的生成分布，修复过程构成无限树，用 Thompson Sampling 调度（pass rate 注入 Beta 先验，h=0 的不可运行程序留在树中永不硬淘汰）。其引用的 Olausson et al. 结论——greedy/BFS/fixed-width 朴素修复的收益与独立重采样相当——是"修复天然有价值"这一假设最强的反向证据。
- **A2DEPT 的修复天花板**：依赖闭合只修"缺函数"类错误，跨模块类型契约破坏修不了——生成复杂度上升引入的新失败类恰好落在现有修复能力之外。

## 二、处理方式的机制清单

按从生成前到搜索后的顺序分层，各层可组合：

**1. 预防：把失败面从可变空间切掉。** FunSearch 骨架+空函数头（"减少重建已知结构时的失误面"）；EoH/ReEvo 固定签名+禁随机+禁解释；ShinkaEvolve/BEAM 的 EVOLVE-BLOCK 不可变区；A2DEPT 的 Preface 固定+Immutable/Mutable 角色化解析；EvoStage 组件级分工+coder 低温 0.2；RedAHD 删最易错算子+拆分两次调用；RL-Algorithms 把类名/方法名/返回形状契约写进 prompt；ShinkaEvolve 的 HT prompt 预先注入失败模式警告（degeneracy warning）。这一层全部论文都有，是唯一的全员共识。

**2. 解析容错与限次重采样（评估前重试）。** MWV 失败原因回喂 5 次+fallback 模型；ShinkaEvolve patch 无效（含动了不可变区）以 Reflexion 解析反馈重采样 3–10 次；TurboEvolve 返回不足 K 个候选该轮 re-query；Evolving Code 解析失败回退父代（恒等变异，失败变无害空转）。**机制与资源定性**：重采样虽然未消耗主 evaluator 预算，但直接消耗 LLM 调用与 token，且通过改变进入后续环节的有效候选分布对搜索机制产生实质影响，绝非“零机制成本”。

**3. 失败个体的处置。** 三种形态并存：
- **丢弃**：FunSearch、ReEvo、LMX、TurboEvolve 等明确规定无效个体不进种群/数据库/父代池。树搜索语境下部分方法可能隐式不入树，但未披露的方法应严格归为“未知”，不能一律推定为默认丢弃。
- **惩罚值**：MeEvo 执行失败 $f=\infty$ 并排除出父代池；A2DEPT 评估失败 $-\infty$ 但仍插入搜索树（谱系保留，配合 Boltzmann 全树历史采样）；AHD Agent 的 -2.0/-1.5 分档；DeltaEvolve 各任务 0 分/惩罚分；Compute Allocation 的 0.0；SMCEvolve 的最差奖励+MH 连续接受（把"丢弃 vs 惩罚"的差别消解为退火温度的连续压缩）。惩罚值的关键设计点是统计语义：进不进均值、进不进 UCT/Beta 信念、与"极差但可运行"是否混同——除 A2DEPT 和 SMCEvolve 外没有论文处理过这一点。
- **限次修复**：CDEoH（预算 B）、MeLA（M 次，最优有效候选替换）、MeEvo（COR=2，无错且更优才替换）、BEAM（max_fix_try=3，组件级）、PhyloEvolve（3 次+回滚/Designer 升级路径）。触发条件从"评估抛错"到"动了不可变区"不等；修复失败者的最终去向普遍没写。

**4. 错误信息作为搜索资源。** 四个去向：
- **进生成 prompt**：EvoPH 的 `error_reason`/traceback（演化出的 prompt 自建"先纠错后优化"指令）、RL-Algorithms 的 Runtime Errors 槽位（要求变异"justified by the failure patterns"）、DGA2D 的 `{diagnostic}` 注入。
- **进跨代记忆**：MeEvo 的 ERR 四元组（驱动收敛诊断）、MeLA 的错误历史（"avoid the errors"）、CORAL 的失败 attempt 全量档案+失败方向笔记。
- **进统计与信用**：Compute Allocation 的 0 分 pull 改变 bandit 臂估计（失败率作为任务特征进入分配决策的唯一实证雏形）；DGA2D 的负信用沿管线回传到算子粒度；AutoSND 的失败结构频率对比编译成 avoid/bound 约束；Clade-AHD 的 β 通道（预留未用）。
- **进训练信号**：AHD Agent 把无效处置写成 reward 梯度；AlgoPilot 用轨迹先验在生成过程内部压制无模式输出。

**5. 评价口径吸收。** 不设处置分支，把失败压进 fitness 定义：RL-Algorithms 的 max-return+clip 让训练崩溃自然落到 0 分下限且保留最佳 checkpoint 分数；SMCEvolve 的奖励下界锚定随机基线；AlphaEvolve 的评估级联让无效程序死在便宜的一级（方向与其他家相反：修的不是程序，是评估成本）。

## 三、无效率与失败统计：文献中仅有的数字

无效率几乎从不被报告，仅有的数字反而说明问题规模：

| 来源 | 数字 |
|---|---|
| LMX | CodeGen-6B 约 30% 有效；去掉强制签名后单亲变异 0% 有效；父代顺序可造成 50% 以上成功率波动 |
| MeLA（SR 表） | EoH 53–89%、ReEvo 41–96%、MeLA 93–99%（四任务） |
| EvoPH | 有 prompt 进化 70–80% 可执行，无 prompt 进化 20–45% |
| EvoStage | Pass Rate 均值 78%（ISPD）/89%（ICCAD），EoH 23%、AlphaEvolve(OpenEvolve) 25% |
| AutoSND | validity 99.0% vs Codex-Evo 23.6%；去掉结构族约束 98.0%→76.5% |
| TurboEvolve | validity 随生成排名递减（头名更高，尾部更多样），且随 LLM 后端变化 |
| Compute Allocation | 单任务连续 42 代无效后首个有效配置；卡死 run 465 个子代全 0 分；Llama-HT 512 次调用零有效配置 |
| MWV | GPT-5 Mini 需 15 次重试预算才能完成 300 步链（其余模型 5 次） |

两个跨论断：无效率是模型属性与任务属性的乘积（TurboEvolve 的后端差异、MWV 的模型差异、Compute Allocation 的任务差异互为印证）；无效段长度直接塑造搜索曲线（Compute Allocation 的 HT 恰是深度收益最大的任务，前 42 代无效没有阻止后期 +0.333 的深度增益）。

## 四、预算口径

修复与失败评估怎么计账，直接决定"修复是否划算"能否被评价。文献中的四种口径：

1. 失败执行全额计入评价预算（TurboEvolve：N_eval 明确含编译错误/异常/超时/违约；Compute Allocation：retries 的 FLOPs 全计）；
2. 修复计入全局 LLM 调用预算、与评价预算分离（A2DEPT 的依赖闭合；ShinkaEvolve 的 patch 重采样在评估前）;
3. evaluator 预算只算实际执行，诊断工具调用免费（AHD Agent；CORAL 的本地测试在提交前拦截编译失败）;
4. 每个提案无条件先评估、用概率接受代替丢弃（SMCEvolve：N·K·T 全额计价，无效的代价均匀摊进预算）。

其余论文口径未说明。EoH 家族普遍把无效个体计入生成预算（LLM query 已花）但排除在分数统计外，无一篇核算无效样本浪费的预算比例。

## 五、谱系与脉络

**奠基期（2022–2024 上）从测量走向定型。** ELM 第一个撞上生成无效，区分 diff 可应用/程序可运行两层并把比例画进图，应对是微调变异模型与接口内置约束；LMX 把 validation rate 升格为主指标并坦率报告 70% 无效，同时点破本质：LLM 变异无法像经典遗传算子那样由构造保证合法性。处置策略在此时定型为丢弃。Evolving Code 把"Check"写进算子三步规范并给每个算子配默认回退。FunSearch 面对可廉价判定正确性的任务，用操作化的"不正确"定义（超时/超内存/非法输出）+骨架防错+10^6 采样吞吐把无效率摊薄为成本项。

**分化期（2024 下–2025）：两条平行路线各自出现修复。** 种群线内部，CDEoH 批评家族默认（"prematurely eliminate candidates that are promising at the level of algorithmic ideas but flawed only in implementation details"）并引入报错条件修复，是家族内唯一有独立消融的转折点。反思线从 ReEvo 的"反思只看成功个体对比"出发（错误在进入反思前就丢失），MeLA/EvoPH 在 2025 下半年平行引入报错回喂（两者与 CDEoH 互不引用，属平行发明），MeEvo 把错误升格为跨代历史记录。系统线 ShinkaEvolve 把修复放在评估前的解析层，PhyloEvolve 给出最完整的修复级联与升级路径。MWV 披露的重试规格与 REx 的形式化（修复=带调度的搜索）证明这一层可以做成被实验评价的机制问题，而实现层面的朴素修复收益存疑。

**收敛期（2025 下–2026）：从处置失败转向利用失败。** 错误信息沿着"进 prompt → 进记忆 → 进统计/信用 → 进训练信号"的方向逐级结构化：EvoPH/RL-Algorithms 的报错槽位、MeEvo/CORAL 的错误记忆、Compute Allocation/DGA2D/AutoSND 的失败进分配与信用、AHD Agent 的错误进梯度。与此同时生成复杂度上升（完整求解器、程序树、pipeline 图）催生分层防错（A2DEPT 依赖闭合、BEAM 组件级 Fix、EvoStage 阶段级指标反馈、DGA2D 四层验证门），并暴露修复能力的天花板（跨模块契约破坏修不了）。无效率也开始被个别论文当作系统指标报告（TurboEvolve、EvoStage、AutoSND），但主流仍是盲区。

**智能体化的结构性差异。** 文件级操作（CORAL）让解析错误这一类整体消失；修复时机从框架规则变为策略自主决策（本地测试 vs 提交 vs 修复）；失败以知识对象形式跨 agent、跨代复用。EMG 代表最彻底的形态——错误离线编译成条件纠正规则、测试时零试错，但其前提（失败+专家成对轨迹）在 AAD 场景没有天然对应物。

## 六、对本仓库的可借鉴点（TraceAAD 的设计启示）

> **核心判断收束**：
> **错误处理影响候选可执行率、有效提议分布和单位资源下的搜索收益。先明确失败语义与计账边界，再决定哪些失败信息值得用于生成或分配。**

围绕这一认识，TraceAAD 应在形式化抽象与系统实现上落实以下四点：

1. **严格解耦动作、观察与上下文构造**：
   - **动作（Action）**：选择算子 $o_t \in \{\text{Refine}, \text{Pivot}, \text{Fuse}, \text{Repair}\}$ 是控制器的决策；
   - **观察（Observation）**：执行候选代码后从沙箱环境获得的 traceback、报错文本或超时信号是环境反馈 $e_t$；
   - **上下文构造（Context Construction）**：将报错信息结构化提取并组织进下一次请求的提示词中，是上下文工程策略。
   - **形式化边界**：错误信息属于观察或上下文，**绝非天然属于算子 $o_t$ 本身**。混淆动作与观察会导致状态与动作空间定义紊乱。

2. **重采样与修复不是“零机制成本”，会实质改变有效提议分布**：
   - 在解析层加入报错重试虽然避免了将坏代码送入耗时的主 evaluator，但直接消耗了额外的 LLM 调用与 Token 资源；
   - 更重要的是，重试与修复改变了进入评估池的有效候选提议分布 $K_\phi(x \mid C, o_t)$，使其转变为带有报错纠正偏置的后验分布。这属于实质性的生成机制改变，必须在资源核算中明确分账（同时记录 LLM 调用与 evaluator 评估）。

3. **保留失败不等于把所有失败塞进 prompt：六类失败状态的显式解耦**：
   - 事实库中应当分别记录六类失败事实：
     (1) **解析失败**（Parse Error: 提取不出代码/语法非法）；
     (2) **运行错误**（Runtime Error: 异常崩溃）；
     (3) **超时**（Timeout: 死循环/超计算时限）；
     (4) **不可行**（Infeasible: 违反问题硬约束/输出格式不合规）；
     (5) **有效但低分**（Valid but Suboptimal: 算法质量劣质）；
     (6) **重复候选**（Duplicate: 语义或代码与历史记录重合）。
   - 分别记录后，必须将下游的三项设计选择正交拆开：
     - **是否进入生成上下文**：例如将运行 traceback 整理后选择性放入下一步修复 prompt，而将低级解析失败在前端消化；
     - **是否影响选父权重**：例如严重退化或不可行节点降低选父权重，但仍保留在谱系中；
     - **是否参与质量与信用统计**：例如失败节点是否折算为 0 分或负信用，避免将运行崩溃与可运行的低分算法简单混淆。

4. **无效率作为任务/分支诊断量与资源分配信号**：
   - 无效段长度和失败类型分布反映了当前代码分支的修改容错度与模型的表达边界；
   - 在预算分配层，连续产生无效候选的分支可被分配器识别为脆弱或停滞区域，从而指导探索预算向更具可行潜力的分支转移；在跨任务比较时，无效率也是衡量任务难度与 Prompt 约束有效性的关键诊断指标。

覆盖 `papers/` 库 38 篇；LLaMEA 原论文与 FunSearch 补充材料不在库内，相关结论以库内转述为限。
