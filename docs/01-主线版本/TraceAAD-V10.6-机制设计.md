# TraceAAD V10.6 完整机制设计

公共任务修订：任务描述与模板统一来自公共任务定义，恢复VRPTW为旧基线/V9.16文本，移除V10.6私有任务描述与具体时限注入。两次调用使用相同任务信息。

摘要与算子修订：依据用户进一步要求，摘要目标约500字，单条Prompt上限1024 tokens，历史最多8条、8192 tokens；借鉴BehaveSim的实现与行为案例，对Refine/Pivot/Fuse作有限的决策语义澄清。下文为修订后的唯一推荐配置。

提示词修订：V10.6方法新增的指令采用正向行动表述；公共任务原文按历史比较版本保留，每个信息块承担一个职责。任务调用、算子方向和输出格式分别表达；研究诊断与失败案例保留为本文设计依据。

生成视角修订：每次调用都是一个自洽的算法设计任务。算子用“沿当前思路改进、尝试不同思路、结合两份算法的有用思路”表达；行为分析术语留在研究解释和确有需要的任务调用说明中。

## 1. 核心决定与目标

**V10.6第一次调用先生成设计Idea、再生成完整Code，第二次调用重新生成实现Idea；以真实父子关系和评价构成形成历史；预算保留V10.5的质量与次数基座、算子条件分布和逐评价重选。** 为满足先选父后定算子的顺序，用等价联合分布重排抽样，不改变相同可扩展集合和基础权重下的目标概率。新Prompt会改变哪些长程序能够容纳，因此不把整个版本的实际搜索路径称为与V10.5等同。

优化目标是在固定真实评价预算下获得更强的最终算法，并确认未知实例与跨规模表现。轨迹首先帮助生成器理解当前实现的形成过程。摘要准确率、局部改善率、谱系数量都不是最终优化目标。

本版撤下个体EI/Beta估计及20%机会混合通道。相近fitness、出生算子和回撤量能否支持增益迁移尚未得到验证；不能为了凑出可计算的评分，用这些特征假装识别了潜力。群体统计并非原则上不能用于个体，但需要相关性、条件可迁移性和不确定性校验，目前不具备这种证据。

本版也不使用活跃代表、候补池、固定多步保活、行为聚类、LLM价值critic或在线算子成功率调整。退步后的多步潜力仍未解决；保留旧分配是当前行动选择，不是宣称旧分配最优。

## 2. 已有证据如何约束设计

- V10.5前750次评价的15路中，5847次Refine涉及2762个父节点，2001个只获得一至两次Refine（72.4%）。这一政策分布下的数据不足以支撑逐节点独立潜力曲线。文献核查
- CVRP node363的depot偏置在零值上相乘，文字声称的机制未生效，而实际程序获得较好fitness。应修正历史描述，不因意图未实现就自动修改或淘汰程序。
- V10.4的计划先行、长文本历史与实现上下文问题，不等同于代码后描述。MCTS-AHD在代码生成后另调模型提取至多三句描述；本稿只借鉴其方向，单次调用的效果不能由其结果保证。[原文§3.1、Appendix E.2](https://arxiv.org/html/2501.08603v3)
- 已有形成来时路实验支持保留与当前节点匹配的近期路径；加入所有直接子代尝试没有稳定额外收益。生成上下文经验
- BehaveSim Figure 5/§4.2指出，Code 6因bug使future_potentials全为零，Code 8因距离项权重占主导，两者都表现为最近邻；Figure 2展示argmin/argmax的微小替换可以改变选择行为。本版由此完善实现摘要，未引入其行为度量，也不声称论文验证了本版摘要方案。[BehaveSim原文](https://arxiv.org/html/2603.02787v1)
- 过去的landing、历史贡献分及固定开发块没有形成跨任务稳定优势；不把“没开发过”写成“开发价值高”。预算分配经验

## 3. 搜索对象、初始化与入档

沿用单父程序树。节点继续保存`id, code, idea, fitness, evaluation_id, parent_id, operator, donor_id`。V10.6的`idea`字段存第二次调用的实现Idea；前置设计Idea保留在请求/响应和候选事件中，渲染标签统一为`Implementation Summary`。

初始化从头生成8个有效根，Init也采用Idea→Code→独立实现Idea。初始化正式评价计入总预算。生成无效不获得根名额；预算用完仍不足8根时如实报告初始化未完成，不额外补预算。正式比较不从历史赢家热启动。

每次非初始化迭代只产生一个子代。有限有效fitness的候选全部入档，包括退步、持平和重复代码。Fuse仍以选中的parent作为唯一父节点，donor只记录引用。摘要缺失不影响入档；低分、代码长和“看起来不像Pivot”都不成为额外语义门槛。

最终输出全档案中fitness最大的程序；当前不能装入生成上下文的程序仍可成为最终输出。

## 4. 预算分配：概率不增加新的潜力假设

### 4.1 基础分布

令A为有限有效fitness、最低完整父代Prompt能够放入上下文的节点集合，N=|A|，q(n)为越大越好的原始fitness。沿用既有ESS校准：

$$
E=\min(N,\max(0.1N,2)),\qquad
w(n)=\exp[\beta(q(n)-q_{max})].
$$

β使质量归一化权重的ESS尽可能达到E；若并列最好节点数超过E，沿用可达下界处理。令c(n)为该node ID已经被选为父代的次数：

$$
p_0(n)=\frac{w(n)/\sqrt{c(n)+1}}{\sum_{j\in A}w(j)/\sqrt{c(j)+1}}.
$$

次数只作为沿用的机会分散规则，不解释为成功率或潜力。一次候选选定后加一；同一pending请求的传输重试、恢复和token计数不重复增加。质量ESS、次数修正后的ESS分别记录，不声称前者控制了后者。

### 4.2 先选父，再定算子

V10.5目标联合分布为：R/F用p0选父；P用0.5p0+0.5/N；请求算子比例为0.50/0.15/0.35。V10.6按下面完全等价的方式执行：

$$
m(n)=0.925p_0(n)+0.075/N.
$$

先按m选择父节点n，再按以下条件概率选择请求算子：

$$
P(R\mid n)=\frac{0.50p_0(n)}{m(n)},\quad
P(P\mid n)=\frac{0.075p_0(n)+0.075/N}{m(n)},\quad
P(F\mid n)=\frac{0.35p_0(n)}{m(n)}.
$$

三者之和为1，边际请求算子比例仍为50/15/35；这是抽样次序变换，不是节点条件价值学习。它保持相同A与p0下的联合概率，不保证同一随机seed产生与旧版相同的实现路径。parent日志记实际边际m(n)、条件算子概率和`parent_route=joint_marginal`，不虚构旧版quality/uniform隐变量。

### 4.3 Fuse donor与回退

请求Fuse后，在档案中排除parent自身、祖先、后代，再按完整Fuse最低上下文能否容纳过滤。从合法候选中取fitness最高5个（不足5个则全部，平分按ID），均匀选donor。该规则只是沿用谱系约束，不宣称保证实际机制互补。

没有合法donor时，将这次Fuse执行为Refine，parent不变，不重抽、不生成第二个候选；记录requested=Fuse、executed=Refine。此时实际R/F比例由可用donor决定；不能声称实际执行比例始终严格50/15/35，也不把缺失Fuse质量转给Pivot。

## 5. 任务描述与模板：统一公共定义

共同任务信息只来自 `evaluation.task_description` 和 `evaluation.template_program`。V10.6采用与V9.16一致的目标函数呈现：保留函数签名和docstring，清空示例函数体，交由模型实现。父代与参考算法仍展示代码。

五个公共任务文本以旧基线批次为准；VRPTW已恢复到8月22日使用的文本，其余四任务原本一致。方法层不设置按任务名称选择的描述表，不替换公共模板措辞，不追加仅供V10.6使用的求解器规则或具体评测秒数。评价器、数据与真实超时限制保持原有配置。历史一致性已核查确认。

任务块与方法的算子、历史及输出指令分别承担职责。生成与校准两次调用复用同一任务块；公共原文中既有的表达按历史定义保留，方法新增指令采用正向表达。

## 6. 生成协议与算子

### 6.1 默认：Idea→Code→独立实现Idea

第一次调用接收任务、当前代码、需要时的参考代码与最近形成历史。先用一个简洁自然段说明主要决策方法和关键计算，再输出完整代码：

````text
First describe your proposed algorithm in one concise paragraph, explaining its
main decision method and key calculations. Then implement it in this format:

Idea: <design idea>
```python
<complete target implementation, including all required imports and helpers>
```
````

静态检查通过后，第二次调用接收同一公共任务块、本次设计Idea、最终完整子代代码，以及存在时的完整父代代码。原设计Idea是意图参考，最终代码的运算决定实现描述。第二次输入不包含历史摘要、fitness或控制器调度信息。

```text
Explain the algorithm implemented by the final code for the stated task.
The design idea provides the intended approach; the code operations determine
the implemented method. Describe its main decision rule, the calculations that
determine its output, and the important parameters and conditions. Derive each
preference from the computation and the supplied task contract. State directly
established effects as facts and expected performance benefits as hypotheses.
Write approximately 500 words in 2–3 paragraphs, scaled to the implementation's
complexity. Return your implementation idea as: Idea: <implementation idea>
```

有父代时追加：`Describe the important implementation changes relative to the parent code.`

第二次调用负责说明而非修改代码。其结果进入节点idea字段和后续历史；前置设计Idea仅保留在过程记录中。两次调用合起来生成一个子代，进行一次正式评价。没有新增语义验收、自动修复、个体潜力分数、edit或RL。

后置Idea目标约500 words、2–3段，简单实现可更短；字、词和token分别计量。摘要重点是实际决策方法、影响输出的公式与参数、关键条件及父代变化。可由代码确定的作用写为事实，预期性能收益写为假设。它仍是模型的实现说明，不是经过执行验证的组件因果报告。

单条说明进入后续Prompt的上限为1024实际tokens。超长时保留能容纳的完整前缀段落与省略标记；第一段已超限则该次展示标不可用。原始响应完整留档，不追加压缩调用。根、donor和历史节点采用同一规则。

### 6.2 四种生成行为

保留Init/Refine/Pivot/Fuse及中等强度的生成自由度。生成模型负责根据本次提供的任务、算法和历史，设计并实现一个算法。算子说明它与输入算法之间的设计关系。BehaveSim帮助我们认识到实现细节的重要性；这一认识通过完整代码、必要的调用说明和实现摘要进入上下文，通用算子使用直接的算法设计语言。

| 算子 | 正向行动指引 |
| --- | --- |
| Init | 为给定任务设计一个有竞争力的算法，并实现提供的函数接口 |
| Refine | 沿当前算法的主要思路设计更好的版本，自主选择最有希望提升表现的实现修改 |
| Pivot | 为同一任务设计采用不同主要思路的算法，将当前算法作为参考 |
| Fuse | 结合当前与参考算法中的有用思路，选择并调整适合共同使用的部分，争取超过两份输入 |

具有形成历史时采用下列说明。质量目标与接口约定由公共任务块表达，此块说明提供的历史是什么，以及它与本次设计算法的关系：

```text
The development history describes how the current algorithm was built and how
previous versions performed. Use it as context for this algorithm design task.
```

四条算子指令如下。每次Prompt装入被选中的一条，统一替换旧指令；Init仍承担原有职责，措辞一并收敛：

```text
Init:
Design a competitive algorithm for this task and implement it using the provided
function interface.

Refine:
Build on the current algorithm's main idea to design an improved version for
this task. Choose the implementation changes most likely to improve its performance.

Pivot:
Design an algorithm for this task using a different main idea from the current
algorithm. Use the current algorithm as a reference for developing a promising
new approach.

Fuse:
Design an improved algorithm for this task by combining useful ideas from the
current and reference algorithms. Choose and adapt the parts that work well
together, aiming to outperform both algorithms.
```

### 6.3 Prompt组装与信息职责

正式Prompt由任务与调用说明、Current Algorithm的完整代码与评价分数、需要时的Reference Algorithm、匹配的发展历史、本次算法设计要求、输出格式组成。历史解释紧邻实际历史展示，历史为空时自然省去。设计要求块使用`# Algorithm Design Task`标题和所选指令正文。Init使用任务、初始设计要求和公共输出契约。各块按既有顺序组装，每种指令出现一次。

每次调用的材料足以回答：解决什么问题；函数提供哪些信息、输出怎样被使用；已有算法是什么、表现怎样；这次沿用、替换还是结合哪些算法思路；返回什么。模型通过这些材料完成当前设计。种群、预算分配、节点价值、行为距离及算子调度由外部搜索程序负责；其名称与数值保存在控制器日志中。

写作要求是“目标＋行动＋必要条件”，并从生成模型收到的这一次任务出发检查用语。共同目标放在Task Contract，算子定义本次算法设计方向，Summary描述完成代码中的算法思路与实现。BehaveSim的案例、历史失败清单及本文设计论证供科研与实现人员查阅；生成模板以本节标明的正式指令块为准。长摘要的容量用于具体算法内容。

长度计数、代码完整性、接口合法性、评价和恢复由现有程序设施校验。接口中的数学条件、真实代码和历史数据按事实保留。此次指令调整遵循用户的提示设计偏好，效果仍由实际生成与搜索结果检验。

研究解释上，OBP对同一分数作严格单调变换，argmax通常不变；ACO中即使边权排序相同，调整相对权重仍可能改变归一化转移概率。这解释了为什么实现细节需要保留。确有必要的argmax或ACO转移公式放在对应任务的调用说明中，说明候选函数的输出如何被使用。通用算子采用第6.2节的算法设计表述，不把“概率变化、激活条件、尺度调整”设成所有任务共同的生成目标。

Refine不要求行为完全不变，Pivot也不要求修改大量代码。二者仍可能产生重叠的候选，区分的是本轮生成方向，而不是事后证明其属于哪个语义类别。发现一个未生效项不触发自动修复、回滚或免费重生成；代码完成后Summary如实描述，正式fitness决定结果。

生成器看到当前代码、真实质量与匹配的形成历史；Fuse另见donor代码、摘要与质量。沿用“历史失败在限制被处理后可重访”的原则；提示只说明整体修改与评价相伴，不下组件因果结论。

算子日志表示实际使用了哪条指令，不表示已通过语义分类器验证。上述改变不新增执行probe、行为阈值、算子验收器或新颖奖励，也不改变预算分配与算子比例；模型未充分遵循Pivot或Fuse时仍按真实代码评价。这保留了算子自由度，也避免把尚不可观察的行为差异硬编码成拒绝门槛。

## 7. 历史组织与上下文裁剪

沿用最近最多8条形成边，包含当前节点自己的形成事件，按时间从旧到新展示；历史区上限为8192实际tokens（`history_tokens=8192`）。每条按真实父子关系取出修改前后质量和child的代码后摘要；Fuse形成事件额外显示当时参考算法的fitness。给模型展示“先前版本、修改后版本、参考算法”，父子ID、donor ID和算子代码保留在存档及`history_ids`等追踪字段中。

示意（数字为假设，不是实验结果）：

```text
Step 3
Previous version fitness: -9.3
Reference algorithm fitness: -8.9
Resulting version fitness: -9.0
Implementation Summary: <该次完成代码的实现摘要>
```

参考算法的历史补充仅使用当时的评价分数；完整谱系和历史代码保留在档案中。当前非根摘要只在最后形成事件展示一次；根与当前donor的摘要放在各自代码块旁。历史区被裁掉时，当前完整代码仍足以作为生成输入，不通过额外复制摘要绕过预算。

精确计数沿用模型服务tokenizer，满足`prompt + 16384 output + 256 margin <= 32768`，即完整输入最多16128 tokens。8192是历史上限，不是硬预留；通常按未触及单条上限的摘要组织8条，最终以实际计数为准。8条摘要若都达到1024，再加事件头就会超出8192，按规则移除最旧事件，不截断最新摘要。历史用满8192时，任务、当前代码、donor及其他输入最多还可用7936 tokens，长代码仍可能迫使移除旧事件，不保证每次保留8条。处理顺序固定：

1. 所有摘要先按第6节1024-token上限生成完整段落视图；组装历史仍超8192时，逐条移除最旧事件，不把每条重新压回一句话。
2. 完整请求超限，继续移除最旧历史事件。
3. 无历史仍超限，依次移除donor、根当前节点的辅助摘要（若有），原始摘要留在档案。
4. 不截断current/donor代码、任务接口或输出契约。最低Prompt的可容纳性按无摘要、无历史的完整代码判定。
5. donor最低Prompt过长则跳过该donor；parent最低Prompt也过长则只暂时不可扩展，仍留在档案；全无可扩展节点则报告未完成。

Prompt代码视图沿用现有普通注释过滤，原始候选模块完整保存和评价。摘要缺失时只显示`[Summary unavailable; refer to implementation.]`，不得拿旧Idea或自动猜测的机制填充。

记录原始/实际展示的摘要tokens、超长省略原因、历史tokens与保留事件数，用来检查实际上下文是否满足设计，不作为在线奖励。

## 8. 解析、预算与恢复

第一次响应包含设计Idea及唯一闭合的Python代码块。检查目标函数名、参数接口、AST与compile；完整模块直接存档和评价，保留装饰器、辅助函数及模块级语句。若finish_reason为length但代码围栏已闭合且静态检查通过，仍可进入校准和评价；代码未完成或格式不明确则只记生成失败。

第二次响应采用 `Idea:` 标签。空白、格式错误、非正常结束或length截断均记实现说明不可用；传输失败记录错误，完整候选仍进行正式评价。单条进入历史的1024-token上限不等同于生成输出硬上限，两次调用沿用现有输出预留与精确计数。

校准请求使用完整父子代码并进行精确计数。若超过完整请求预算，记录 `context_exceeded`，保留候选并将实现说明标为不可用；不截断代码、不把前置计划冒充实现Idea。这个边界单独记入过程数据。

两次调用分别记录stage、call_id、请求、响应、usage、耗时与结束原因。pending先持久化生成响应，再持久化校准响应；恢复时复用已有响应。尚未获得并持久化返回结果的远程调用无法保证恰好一次，恢复只重试未完成阶段。校准传输错误作为一次说明失败记录，不进入自动反复重试循环。

沿用V10.5评价提交与收据：每次真实评价计一个slot，包括失败和超时；纯LLM失败不虚构评价。已确认收据不重复执行；提交状态未知时保留unknown reservation并停止该路。连续50次无可评价代码沿用原停止规则。

checkpoint版本仍为106，generation标识改为 `idea_code_then_implementation_idea`，任务与源码指纹同时更新。原单次调用检查点与新协议不兼容，不能将旧run恢复成新配方。旧运行结果和启动记录保留原含义。

## 9. 完整流程

```text
initialize independent RNG, empty single-parent tree and evaluation journal
while budget remains:
    if valid root count < 8:
        select Init, no parent or donor
    else:
        A = nodes whose minimum complete parent prompt fits
        p0 = existing ESS quality weights corrected by selection counts
        select parent n from m(n) = .925*p0(n) + .075/len(A)
        select requested R/P/F from the conditional probabilities in section 4
        increment n's selection count once
        if requested Fuse:
            select a fitting top-5 cross-lineage donor, or execute Refine

    build task + current code + optional donor + recent formation events
    checkpoint this selection, exact prompt and post-selection RNG state
    make first model call: design Idea, then complete Code
    durably save response and finish metadata
    extract design Idea and validate complete code
    if code is not evaluable:
        log generation failure; checkpoint; reselect
    else:
        make independent implementation-Idea call using the same task, plan and final code
        durably save the description response or its failure; classify availability
        submit one formal evaluation using the complete archived module
        persist evaluation receipt; charge one slot
        if fitness is finite and valid:
            store a root or one child, with optional post-code summary
        log outcome; checkpoint; reselect
return best valid program in the full archive
```

## 10. 唯一首发配置

| 项目 | 配置 |
| --- | --- |
| 主比较 | 五任务，每任务3次独立运行，seed=0/1/2；每路1000次真实评价，初始化计入 |
| 初始化 | 8个有效根，从头生成 |
| 模型与采样 | 同V10.5正式后端的Qwen3.8-27B配置；temperature=1，top_p=.95，top_k=20，thinking关闭；固定并记录真实权重/量化/服务信息，不仅凭模型别名宣称完全一致 |
| 输出 | 首次Idea→Code，独立调用生成实现Idea；约500字的目标（英文约500 words、中文约500字，分别计量），简单实现可更短；Prompt摘要硬上限1024实际tokens |
| 分配 | 第4节的等价父代先行分布；请求R/P/F=.50/.15/.35 |
| ESS/次数 | fraction=.10，minimum=2，保留并列处理；1/sqrt(c+1) |
| donor | 合法且可容纳的top-5均匀选；缺失则Fuse转Refine |
| 历史 | 最近最多8条真实形成边，8192 tokens；Fuse事件展示参考算法quality，IDs与算子代码在日志中追踪 |
| 上下文 | 32768总上限；16384输出预留；256余量；整事件裁剪 |
| 评价器 | 原有五任务定义、训练数据、求解设置和各自超时；不改变评价标准 |
| 额外控制器 | 无新增潜力模型、代表集合或保护预算；仅新增实现Idea调用，无价值critic |

这些沿用参数和工程限额不是由新数据识别的最优参数。本版不同时引入调参网格。

## 11. 实现范围与必要验证

实现时建立独立`traceaad_v10_6`入口，复用V10.5的树、ESS、统一评价、计数、日志和恢复设施。主要变更点明确为：

1. Prompt：使用公共历史任务说明与模板，移除专用覆盖；采用Idea→Code及独立实现Idea协议；采用第6.2节面向单次算法设计的指令，按当前材料组装历史说明与摘要比较句。
2. 解析/候选推进：两次调用分别解析与持久化；解除实现Idea对可评价代码的绑定；按第8节处理length及恢复。
3. 调度：父先行的等价联合抽样；日志记录实际边际和条件概率。
4. 上下文：1024-token摘要完整段落视图、8192-token历史区；辅助摘要可移除，最低可容纳性只依赖完整代码和必要协议。
5. 版本/运行：106 checkpoint、机制指纹、独立run与批次入口。不能仅改METHOD字符串而继承硬编码105的加载校验。

实现采用独立版本目录，复用现有评价器与基础设施，本轮修改后旧检查点因协议和指纹差异拒绝按新配置恢复。实现与本地验证后，用户另行授权停止旧批次、删除旧数据并启动新批次；此次正式启动已完成，未混用旧检查点。

落地后的必要验证按真实风险组织，复用现有测试设施：

- 正常代码后摘要、无摘要、摘要过长、代码前Idea、多代码块、错误接口；确认存档模块与实际评价模块相同。
- length在代码中与摘要中两种位置，前者不执行，后者只执行一次完整代码且不使用残缺摘要。
- 父子事件、根、Fuse donor及上下文裁剪；覆盖1024-token边界、首段就超长、完整前缀段落与省略标记、8192历史及16128总输入边界；不重复当前摘要，不截断代码，不把较强donor贡献隐藏成单亲突破。
- 枚举不同p0（包括零权重、并列与单节点），核对父先行联合概率与V10.5目标完全相同；检查Fuse回退不改变Pivot请求质量。
- pending响应与评价收据恢复，证明不重复生成/评价/增加选择计数；旧版恢复行为不回归。

单元验证和真实服务少量冒烟只确认实现协议与成本，不先开展一整套轨迹消融。然后运行一个完整V10.6配方，在共同真实评价前缀及1000终局比较训练best、运行失败与实际生成成本；冻结后测试未知实例和跨规模。三重复不能支撑强统计断言，不能用最好单路替代稳定SOTA结论。

## 12. 当前仍需诚实保留的边界

独立生成的实现Idea可能仍然描述错误；这是本版要尝试改善的生成行为，未升级为行为验证。质量分配仍可能不给低分结构足够的后续机会；本版没有解决多步潜力识别。ESS随档案增长、节点ID次数重置也仍有局限。

选择这些边界，是为了先实现一条明确的改进链：**明确设计 → 完成实现 → 重新描述实际代码 → 保存匹配的形成历史 → 辅助下一次改进 → 用完整搜索终局决定保留。** 如果用户要求本版同时解决退步节点的开发机会，需要先讨论接受哪种明确的探索规则，不能偷偷用不受支持的潜力估计补齐。
