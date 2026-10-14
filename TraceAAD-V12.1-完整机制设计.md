# TraceAAD V12.1：以可执行改动实验积累算法设计经验

**English working title:** *Beyond Successful Programs: Cumulative Algorithm Design through Executable Change Experiments*

**日期：**2026-10-13  
**规格标识：**`traceaad-v12.1-executable-experiments-r1`  
**状态：**完整研究与实现规格；尚未实现、运行或验证。  
**范围：**一次任务内的 LLM 自动算法设计。保留现有可编辑接口与评价器，不引入任务专家规则、外部教师或模型训练。

---

## 0. 版本决定

V12.1 不再把研究核心设为“给历史加一种表示”“再设计一个父代分数”或“保护所有首版较弱的想法”。

**它研究：搜索怎样从一次具体改动中取得可复验的效果证据，并让该改动脱离原程序的总分，进入后续开发与跨程序复用。**

核心假设是：

> 一份程序没有击败现有最好程序，不等于它包含的每项新计算都没有价值。只依据整份程序的质量筛选后续材料，会漏掉一部分有正向作用的改动；只依据描述总结历史，又不足以辨认这些改动。通过可执行对照识别改动在当前环境中的作用，再在目标程序中重新检验其作用，可以把部分低质量候选转化为有用的改进经验。

这是待检验的机制假设，不是对历史档案已经证明的总体结论。

方法只增加三个相互依赖的构件：

1. **可执行改动实验。** 把一次待检验计算写成共享背景下的处理版与对照版，由宿主程序构建并评价两者。
2. **成果与证据分开保存。** 程序按真实总分竞争；改动按具体环境中的对照结果保存。低分程序不自动丢失提供后续材料的资格。
3. **带证据的改动迁移。** 迁移的是具体代码改动、原对照及适用条件，而非任意复制整个 donor。目标环境必须重新评价，不能继承源环境的收益数值。

主循环是：

```text
提出一个具体计算变化
        ↓
构建可执行处理／对照
        ↓
测量整份程序收益与该改动的条件作用
        ↓
分别更新程序成果与实验记录
        ↓
继续实现 / 迁移至另一程序 / 提出新改动
```

三者共同服务一个目标：**让算法搜索积累可继续使用的改进证据，而不只是积累高分程序。**

---

## 1. TraceAAD 一直在研究什么

### 1.1 原始问题不只是“怎样选父代”

TraceAAD 最初的直觉是：人设计算法时，不只看当前代码和分数，还会记住怎样走到这里、哪些修改试过、为何继续或转向。这个直觉包含两个不同的问题。

**证据问题：**一次生成、评价和后续改写，究竟告诉了我们什么？

**决策问题：**这些证据应当改变下一次的修改、参考选择、继续开发或停止吗？

早期路线信用主要处理第二个问题。后期 Design、实现、失败与材料诊断逐渐暴露出第一个问题：当一次观测不能说明究竟检验了什么计算时，复杂的调度器也没有可靠的输入。

因此，V12.1 不是放弃轨迹或预算研究，而是先改变预算决策能够读取的证据单位。

### 1.2 历史版本留下的有价值内容

| 阶段 | 最值得保留的认识或机制 | V12.1 的继承方式 |
|---|---|---|
| V1–V3 | 多层控制可以同时运行，却未必产生可归因的独立作用 | 不恢复多智能体、多分量价值或多层奖励的组合 |
| V4–V5 | 当前程序、形成过程和参考路线是不同的信息；跨路线可以传递具体计算 | 保留来源边和跨程序来源，但以实际改动及对照为复用对象 |
| V6–V8 | 可执行程序身份、历史身份与繁衍资格不能混成一件事 | 相同代码共享评价缓存；不同实验来源仍独立记录 |
| V8.3、V9.1–V9.4 | 历史最大值、局部恢复和祖先成功，不能直接当成下一次投入的收益 | 不回传永久祖先信用；不把恢复幅度加入“潜力分数” |
| V9.3、V9.16 | 部分有用过程不是单调的；连续开发也有机会成本 | 工作实现可以落后；续投有边界，每次测量后回到全局决定 |
| V9.7–V9.8 | 形成历史对不同生成目标的作用不同 | 提新改动、修同一改动、迁移改动读取不同的实验切片 |
| V9.17 | 已兑现质量和近期开发响应承担不同职责，不必相加 | 程序成果与实验进展分开；近期进展只用于决定是否继续本实验 |
| V9.19 | 来源不同不等于计算不同；代码不同不等于行为不同 | 不用根编号、代码距离或嵌入直接认证“同一思想” |
| V9.21–V9.22 | 一份文本提案可以有不同实现；最好成果与工作实现可以不同 | 同一实验保留实现版本、稳定对照和工作版；不以一次失败否定整个提案族 |
| V10.19 | 应检查改动在原算法中是否真正工作，而不是只看新程序总分 | 把“受控改动”从提示要求落实为可构建的程序对照 |
| V10.24 | 问题字段、长开发段和结束标记不自动形成有效实验 | 必须有可执行比较；暂停是状态更新，不强迫交付重复代码 |
| V10.25 | 净差异可能掩盖失败后被修复删除的计算 | 失败、修复与撤回逐事件保存，不被成功终点覆盖 |
| V12.0 | 已有实现不必通过被选为父代才能进入下一次生成 | 保留第二条继承通道，但传递对象升级为经过对照的具体改动 |

历史依据见 [H1]–[H8]。这个表说明设计的来源，不把所有历史结果解释成同一个原因。

### 1.3 哪些历史事实最有支撑力

V9.8 的固定状态实验有 72 个锚点、三次重复和“历史 × 动作”四种条件，共 864 次响应。父代形成路径对 Refine 的平均即时效果在四个任务上方向一致，对 Explore 没有同样稳定的方向。它说明材料作用应结合当次任务理解，不证明历史必然压制探索。[H2]

V9.17 的 Adaptive 与 FixedCycle 使用相同的首次开发 checkpoint，只改变后续 sweep 调度。CVRP 的三个测试规模、三个重复均偏向 Adaptive，但其他任务并不一致。这是局部开发响应可以参与调度的证据，不是任意正增益都值得续投的证明。[H3]

V9.21 的历史首跑报告 5,495 个两次实现均有有限分数的配对。其中 1,653 组结果类别不同，504 组出现“一次改善、另一次未改善”。这说明“相同提案文本与冻结上下文”仍可对应不同实现结果；它不是已经控制实际计算语义和评价随机性的设计方差估计。[H4]

V10.17 的 CVRP 案例中，声称使用节约法的首版并未忠实实现该计算，随后改写补上容量可行构造。另有计算被写入代码，却在下游归一化中失去作用的案例。[H6]

**V12.1 在这些证据之上新增的判断是：应直接取得可复验的改动效果，而不再要求单个总分回答它无法回答的问题。**“较差候选中有多少可迁移的有用改动”仍需要前瞻测量。

---

## 2. 与相关工作的关系：真正改变哪一步

### 2.1 不把领域压成一个弱基线

已有工作已经覆盖思想与代码共同演化、历史检索、反思、子代价值、谱系统计、多次实现和因果推理。因此，V12.1 不认领这些宽泛功能的首次提出。

| 相关工作 | 已经处理的内容 | V12.1 要单独识别的差别 |
|---|---|---|
| EoH、FunSearch、AlphaEvolve、ShinkaEvolve | 程序生成、种群或档案、参考程序、修改与筛选 | 参考材料不只由完整程序质量决定；实际改动的条件效果成为独立证据 [R1–R4] |
| ReEvo、RefineEvo、EvoMem | 从评价历史获得可复用经验，并用于后续生成或规划 | 不只保存对改动的解释，还保存能够重新运行的处理／对照代码 [R5–R7] |
| DeltaEvolve、PhyloEvolve | 语义改动、形成路径及轨迹条件生成 | 不把父子分差直接解释成某个局部计算的效果；需要明确比较对象 [R8–R9] |
| QUBE、MCTS-AHD、Clade-AHD | 由后代结果、树或分支证据改变投入机会 | 不再主要改写质量与不确定性的选择公式，而是改变输入证据的实验单位 [R10–R12] |
| PAEvo、PathWise、Back to the Beginning | 计划／知识与实现分离；PathWise 还显式采样多个 rollout | 多份独立实现主要增加重复观测；一份匹配对照则回答具体计算是否起作用 [R13–R15] |
| **LabBook** | 单代理维护记忆、检索完整实验记录、生成程序 | 仅改善记录到上下文的通道还不够；V12.1 主动生成此前记录中没有的比较证据 [R16] |
| **CausalEvolve** | 构造结果和程序因素，估计因素效果并更新 causal scratchpad | V12.1 在可执行代码中建立显式的同背景比较，测量一个实际替换，而非仅从历史程序因素估计效果 [R17] |
| **HeurAgenix** | 在求解轨迹中替换操作、寻找改善解的关键操作并指导启发式 | V12.1 干预算法设计中的代码计算，并保存跨程序再实施的实验关系；不是首次在 AAD 使用反事实 [R18] |

这里按实际机制区分，不断言相关工作完全没有控制变量或实验思考。尤其不能把 CausalEvolve 简化成“只看分数”，或把 PathWise 说成每项设计只实现一次。

传统进化计算也已研究依赖关系与局部组合。例如 GP-GOMEA 使用 linkage learning 决定传播哪些程序模式。因此，“把代码分块再组合”本身同样不是 V12.1 的贡献。这里要检验的是：在没有预先固定基因位点的 LLM 程序生成中，由模型提出计算边界、宿主建立真实对照、再按条件效果选择并复验改动的闭环。[R19]

### 2.2 最值得承担的创新主张

候选主张不是“我们提出了因果思想”，而是以下闭环：

> **让程序搜索直接生成可执行改动实验；用同一背景中的对照结果保留改动；将改动及其比较条件跨程序迁移，并在新背景重新检验。**

这个闭环同时改变了经验的产生、保存和使用。各个工具——代码替换、对照、检索、生成代理——都可以沿用已有技术。增量需要来自这个闭环的独立效果，而不是给旧工具换名称。

最接近的三个比较对象是 LabBook、CausalEvolve 和“受限编辑＋常规 crossover”。前两个决定文献定位，第三个决定复杂的对照机制是否确有必要。

---

## 3. 核心研究问题与可证伪假设

### 3.1 研究问题

**有限预算的算法搜索，能否通过保存并复用具体改动的对照证据，比只复用高分程序或语义经验，更有效地累积算法改进？**

该问题有三个可区分的部分：

- 一份较差程序中，是否存在在其当前环境下产生正向作用的计算？
- 这种证据能否改善下一步“继续实现还是移到另一个程序”的决定？
- 额外测量和迁移消耗的预算，能否由最终搜索收益补偿？

### 3.2 三项主假设

**H-A：总分与局部作用存在有用的不一致。** 一部分实现没有超过来源程序，但其中的焦点改动优于匹配对照。只看总分会把它们与完全无效的改动混在一起。

**H-B：对照证据改善复用选择。** 在候选池、目标程序和迁移生成器固定时，根据对照证据选择的改动，比根据源程序总分或文字解释选择的改动，更常带来目标程序的实际改善。

**H-C：该作用能够进入完整搜索。** 在候选、真实评价和模型成本均可核对的比较中，闭环改善最终质量或达到同一质量所需的成本。

H-A 不蕴含 H-B，H-B 不自动蕴含 H-C。实验分别给出答案，论文主张以三者实际覆盖的范围为准。

---

## 4. 数学对象：不估计一个脱离实现的“思想真值”

### 4.1 程序与评价

设任务的完整可执行程序为 \(P\)，评价协议为 \(\pi\)。协议包含任务接口、训练实例、随机种子、评价源码和计算限制。所有目标统一为越小越好的损失：

\[
L_\pi(P)=\frac{1}{n}\sum_{i=1}^{n}\ell(P,x_i;\pi).
\]

只有全部要求满足的程序才有有限分数。失败不填入一个任意大损失参与效果计算。

### 4.2 可执行改动实验

一个实验为

\[
e=(h,v,B,S,K^-,K^+,\pi),
\]

其中：

- \(h\)：待检验改动的显式记录 ID，不是嵌入聚类标签；
- \(v\)：该改动的实现版本；
- \(B\)：本次开发所参照的已评价来源程序；
- \(S\)：两臂共用的程序背景；
- \(K^-\)：合法的对照计算；
- \(K^+\)：本次待检验的计算；
- \(\pi\)：冻结的评价条件。

宿主构造

\[
P^- = \operatorname{Build}(S,K^-),\qquad
P^+ = \operatorname{Build}(S,K^+).
\]

不能让模型分别输出两份任意程序，然后用一句“只有某个机制不同”认证对照。共享背景由同一份源码生成；两臂之间只允许已声明的计算包不同。

### 4.3 三个量，各有用途

\[
G_{\rm total}=L_\pi(B)-L_\pi(P^+),
\]

\[
G_{\rm focal}=L_\pi(P^-)-L_\pi(P^+),
\]

\[
G_{\rm background}=L_\pi(B)-L_\pi(P^-).
\]

它们满足恒等式：

\[
\boxed{G_{\rm total}=G_{\rm background}+G_{\rm focal}.}
\]

正值表示改善。

这个等式不假设各个算法组件独立，也不假设算法性能是可加模型。\(G_{\rm focal}\) 是在当前 \(S\) 中，将 \(K^-\) 换为 \(K^+\) 的有限协议比较；它包含该改动与共享背景的交互作用。\(G_{\rm background}\) 包含其余变化，不自动等同于“编码错误”。

**把 \(K^+\) 移到另一份程序以后，必须重新测量；不能把这里的 \(G_{\rm focal}\) 直接相加到目标程序分数上。**

### 4.4 两种关键不一致

| 观测 | 能说什么 | 下一步候选 |
|---|---|---|
| \(G_{total}\le0,\ G_{focal}>0\) | 整体没赢，但焦点改动在当前背景中有帮助 | 修正其余变化，或把焦点改动移到成熟程序 |
| \(G_{total}>0,\ G_{focal}\le0\) | 新程序变好，但不能把收益归给焦点改动；对照可能更好 | 保留优质程序与对照，不写入虚假的正面机制经验 |
| 两者都正 | 本次改动与整份程序改善一致 | 继续精炼或在另一背景检验 |
| 两者都非正 | 本实现未给出正面效果 | 修正有证据的实现问题，或暂停此具体实验 |

如果 \(P^-=B\)，那么 \(G_{focal}=G_{total}\)。此时比较就是通常的受控父子修改，不获得额外识别能力。V12.1 应复用这类便宜对照，而不是为了显得新颖制造第二个程序。

---

## 5. 一个完整示例

以下数值为说明机制的虚构例子，不是已有实验结果。

已有程序 \(B\) 的损失为 5.80。模型提出一种路线末段改进计算，同时改变了初始路线构造。新背景的初始路线较差。

| 程序 | 实际计算 | 损失 |
|---|---|---:|
| \(B\) | 原成熟路线构造 | 5.80 |
| \(P^-\) | 新背景，不启用末段改进 | 6.45 |
| \(P^+\) | 同一新背景，启用末段改进 | 6.20 |

于是：

\[
G_{total}=-0.40,\quad G_{background}=-0.65,\quad G_{focal}=+0.25.
\]

仅看总分会说“这次候选更差”。可执行对照给出的信息是“末段改进在这个背景中减少了 0.25 的损失，但其余变化带来的损失更大”。

系统保留完整候选，也把末段改进的代码、对照和依赖写入实验记录。随后将该计算接入成熟程序 \(B\)，以 \(B\) 本身为对照，真实得到目标程序 5.70。这个目标环境中的 0.10 才是迁移收益。

```text
较差候选中的局部正效应
    → 保留具体实现和比较条件
    → 找到兼容的成熟程序
    → 实施该改动
    → 在目标环境再次对照
    → 获得新的可交付程序
```

这不是保护较差程序直到它自己追上，也不是复制整个较差 donor。它是**把已检验的改动从原宿主中取出来，重新参与搜索。**

---

## 6. 对照怎么构建：两种模式，一个实验语义

### 6.1 模式 A：锁定来源程序 `locked`

适用于可以直接在现有程序上实施的修改。

- 对照 \(P^-\) 是一个明确的已评价程序 ID。
- 宿主冻结其 AST，并给可替换位置分配稳定 locator。
- 模型返回被替换的完整语句块或函数，以及声明的新增依赖。
- 编译器只在指定位置替换；对照就是原 AST。
- 对照的代码、协议和随机种子完全一致时复用已有评价。
- 新产生一份处理版，消耗一个候选槽。

这与普通程序编辑兼容。优势不在编辑本身，而在后续实验记录与复用闭环。

**不能把整个函数都替换后，声称只识别了其中某一行的作用。**实际替换包有多大，效果的归属就有多大。

### 6.2 模式 B：共享背景的成对构建 `factorized`

适用于新的计算需要共同的结构调整，或者希望把复合改写中的一项计算单独检验。

模型输出三个完整源码片段：

```text
Shared:     两臂相同的背景、接口、调用位置与共同依赖
Control:    在焦点位置执行的旧计算或合法中性计算
Treatment:  在同一位置执行的待检验计算
```

概念例子：

```python
# Shared: 两臂逐字共享的主体

def solve(instance):
    plan = make_initial_plan(instance)
    plan = focal_step(plan, instance)
    return plan

# Control: 固定的合法中性后处理

def focal_step(plan, instance):
    return plan

# Treatment: 本次待检验后处理

def focal_step(plan, instance):
    return improve_selected_part(plan, instance)
```

上例只是表示方式，不向模型预先指定具体任务算法。实际导出的是两份自包含程序，运行时没有实验开关、另一臂源码或访问档案的权限。

新共享背景、Control 与 Treatment 在一次实现请求中产生。宿主分别将 Shared 与两臂绑定。新生成两份程序，消耗两个候选槽。第一臂的评分不进入另一臂生成。

### 6.3 对照来源必须在评分前确定

默认允许的对照只有三类：

1. 已评价程序中实际存在的计算，由程序 ID、AST locator 和源码哈希指定；
2. 有明确语义的中性操作，例如合法完整路线上的 identity 后处理；
3. 整体替换时，原程序本身。此时效果对象是整个替换包。

“删掉可行性检查”“把必要输出置零”“让对照故意用错数据”不是合法的中性对照。新算法替换旧算法，比较的是两种合法选择，不是完整程序与人为破坏程序。

若旧计算需要适配，适配应成为两臂共享背景的一部分。不能单独把对照改写成更差的实现。无法建立合理比较时，扩大焦点包直到比较成立，或者把本次记为未识别；不能由 LLM 的解释填补。

### 6.4 不把可控边界误做成局部搜索限制

焦点包可以包含多个必须共同工作的函数、数据结构和计算步骤。允许一次更换整个算法框架。要求的是比较对象清楚，不是只允许改一行或调一个参数。

必须共同变化的部件一起形成一个包。V12.1 不假设算法能够分解成彼此独立的小模块。

---

## 7. 编译、验证与实际计算边界

### 7.1 宿主可以严格验证的事情

- 两臂共享源码哈希相同，导出接口一致；
- 差异仅位于声明的计算包；
- 所有导入、辅助函数与常量有明确归属；
- 对照源码与其声明来源相符，允许的 alpha-renaming 有映射记录；
- 两臂使用同一任务、数据和评价设置；
- 两份完整程序分别通过接口、合法性及运行约束；
- 评分和差值由宿主计算，模型不能改写。

`locked` 模式使用源 AST locator，而非自由文本 SEARCH/REPLACE。`factorized` 模式拼接完整顶层定义，并检查导出符号与重复定义。源码经过语法验证后输出最终自包含程序。

禁止程序读取 evaluator 源码、测试实例、环境中的实验标签、另一臂源码或档案内容。禁止通过动态执行或源码反射识别自己属于哪一臂。

### 7.2 这些验证不等于语义证明

编译器可以证明实际替换的代码边界，不能自动证明“这确实是 Clarke–Wright”或者“此实现保留了所有算法思想”。

因此实验记录有两个分开的字段：

- `declared_change`：模型提出的计算说明；
- `executed_change`：真实两臂代码、定位、依赖和哈希。

正式实验的抽样代码审查检查说明与实现是否一致。主效果始终属于实际执行的代码包；说明失配不能让代码对照消失，也不能成为更宽语义主张的证据。

在线准入只依赖可复算的源码来源、结构边界和评价契约检查，不引入人工批准或一个假定全知的语义裁判。“这个对照是否足以检验所声称的算法概念”由盲化的离线审查评价。未经该审查，在线标签只表示实际代码包的条件效果。完整搜索结果保留全部在线运行，不按事后审查结果挑选或删除不利路次。

### 7.3 计算没有进入决策时怎么办

相同输出与相同分数可能意味着焦点未被调用、被下游变换消去，也可能只是当前实例未触发差异。主系统不凭零收益宣布整个设计无效。

需要区分时，允许创建一个带明确目的的诊断运行，检查实际调用、输入输出或某个中间量。诊断不能修改正式评价器；额外执行和模型调用单独计入真实成本，并在主比较中给所有方法同样的接口权限。默认正式搜索不依赖一套复杂的求解轨迹探针。

---

## 8. 状态：一个成果档案，一份实验账本

这不是两个人工设计的独立进化种群。

### 8.1 `ProgramArchive`

保存所有有效程序：

```text
program_id
code_hash / complete_code
protocol_id
training_score / per_instance_scores
validity / measured_cost
source_experiment_id / arm
creation_order
```

全局最好程序只按真实目标决定。Treatment、Control 和与语义说明不一致但合法的程序，都可以成为成果。实验归因有问题不自动抹掉实际程序成绩。

完全相同的代码共享程序记录与评价缓存，但保留全部生成 occurrence。缓存键必须包含程序、评价器、数据、种子及资源协议身份。

### 8.2 `ChangeRecord`

记录需要检验与复用的计算变化：

```text
change_id
question                         # 一个具体替换及比较对象
origin_experiment_id
declared_input_output_contract
source_code_locators
required_dependencies
parent_change_id                 # 显式继承，允许为空
implementation_versions
context_experiment_ids
```

`change_id` 由系统创建；后续请求显式声明修订或迁移哪个 ID。相似文字只用于查找，不能自动合并记录、评价和信用。

改变焦点问题时创建新 ID。对同一计算包修复实现、调整实现细节时保存版本。族标签是组织线索，不是算法等价真值。

### 8.3 `ExperimentRecord`

```text
experiment_id / change_id / implementation_version
mode                             # locked / factorized
reference_program_id             # B
shared_hash
control_provenance / control_program_id
control_bundle_hash / treatment_bundle_hash
protocol_id / dataset_hash / seed_bundle
scores / per_instance_differences
G_total / G_focal / G_background
status                           # unresolved / positive_here / nonpositive_here
failure_details
generation_cost / evaluation_cost
revision_of / transferred_from
```

所有评价事实 append-only。重新评价产生新记录，不覆盖旧数值。

### 8.4 `WorkItem`

```text
work_id
change_id / fixed_context_id
best_valid_treatment_id
latest_failed_artifact_id
best_treatment_score_in_context
nonprogress_streak
state                            # active / paused
pause_reason
```

同一 `fixed_context_id` 锁定 Shared、Control 和评价协议。修订 Treatment 时，不能偷偷改变对照来制造正效应。背景改变即建立新 context。

最新失败源码可以用于一次修复；不能覆盖最好有效工作版。历史失败与修复链完整保留。

### 8.5 不把跨背景效果汇总成“思想成功率”

同一改动在三个背景中的收益可以不同。三次结果分别保存，不自动平均成通用质量。相同背景中的连续修订也不是三个独立思想样本。

摘要可以展示各 context 的结果和适用条件，但不能用它们拟合一个未经验证的全局潜力值来主导搜索。

---

## 9. 三个动作与统一信息路径

### 9.1 `OPEN`：提出一个新改动实验

读取任务、一个已评价程序、少量与问题相关的实验记录。提出具体替换、比较对象和预期改变的行为。

默认优先使用 `locked`。当实质重构需要共享适配，或要在新背景中隔离焦点计算时，使用 `factorized`。可提出整个新算法，以已有程序作为完整替换对照，不要求先找到一个小模块。

OPEN 不读取整条优秀祖先链作为默认示范，也不要求首版立刻超过全局最好。目标仍是改善最终搜索成果，但本次首先应构成有意义的程序或实验。

### 9.2 `REVISE`：推进同一实验的实现

读取固定 Shared、Control、当前最好 Treatment、最近两次实际尝试与结果。修改 Treatment，不改变当前问题与对照。

若上次无效，只修明确的交付或执行问题。若局部效果非正，可以提出另一种实现；不是把整项计算判死。若局部效果为正而整份程序仍弱，需要在继续修 Treatment 与迁移到成熟背景之间作选择。

每次修订后重新回到全局决策，不强迫执行固定三步链。

### 9.3 `TRANSFER`：在另一个背景重新检验同一改动

读取源实验的实际差异、依赖、处理／对照结果，以及目标程序完整代码。

要求：

- 明确指出目标程序中哪个计算位置接受该改动；
- 保留不受改动影响的成熟计算；
- 引用源实现版本，列出必要适配，而不是重新自由生成一个“类似思想”；
- 目标对照优先就是目标程序本身；
- 若必须改变共同背景，创建新的 factorized context；
- 目标环境的收益重新测量，源环境收益不作为目标评分。

默认复用候选来自有正向局部结果的记录，但源程序不需位于质量前列。尚未识别的旧实现可以作为 OPEN 的灵感，不伪装成已有正证据的 TRANSFER。

### 9.4 为什么不再设独立 Crossover

普通 crossover 往往同时重写背景、复制 donor 和引入新计算。V12.1 把其中需要研究的部分收束为“迁移哪项改动、在目标中检验什么”。

同时采用多项源改动时，把它们作为新的联合包检验。不能假定多个正效应会直接相加。

---

## 10. 决策器：让 LLM 选择实验，不让它伪造证据

### 10.1 一个模型，两个必要阶段

使用同一冻结生成模型：

1. **选择并定义实验。** 从事实中决定 OPEN、REVISE 或 TRANSFER，确定目标代码、焦点及对照。
2. **构建程序。** 按已经确定的实验规格生成可执行源码。

两阶段不是两个拥有独立记忆的代理。不存在额外 critic、反思委员会、世界模型或在线奖励模型。实验后的数值摘要由宿主自动生成，下次决策直接读取。

### 10.2 决策读取什么

默认初始视图包含：

- 当前最好程序的完整代码、得分和协议；
- 最近一次实验的两臂结果、真实差异与状态；
- 最多 8 张简短实验卡；
- 剩余候选预算及各动作预计需要的候选数。

8 张卡以确定性轮转形成四组：最近实验 2 张；局部正效应但整体未超过来源程序 2 张；仍在改善的未完成实验 2 张；其余未显示的历史实验 2 张。去重后不足的席位由最近记录补齐。这里采用轮转，不按最大的局部恢复幅度排序。

这只是有限上下文的入口，不是对档案的永久删选。决策器可进行最多两轮只读查询，每轮返回至多四个记录。

### 10.3 只读接口

```text
search_experiments(query, status=None, limit=4)
read_experiment(experiment_id)
read_program(program_id)
list_programs(order="quality", offset=0, limit=4)
```

检索默认使用关键词与显式符号、接口字段。可换为嵌入检索，但排序器不是贡献，且必须做检索身份记录。关键词或嵌入不能更新 `change_id` 的语义身份。

待提交的实验规格中引用的完整程序和完整源实验必须已被读取。引用尚未读到的内容时，宿主先返回内容，再让同一决策器提交最终规格；此调用计入查询上限。不能据一张描述卡直接假装已核对代码适配。

### 10.4 预算分配规则

默认采用**单代理的离散实验选择，加一条明确的探索保底**：

- 每执行三个非 OPEN 实验后，下一次强制为 OPEN；
- 无可用记录时自动 OPEN；
- 其余时刻由模型根据具体代码、对照结果、适配关系和成本选择动作；
- 不把总分、局部增益、访问数、相似度线性相加；
- 不因某个源程序总分较低而禁止选择其正效应改动；
- 不因一次负效应就禁止整个 change 在别的环境中再次检验。

保底限制的是连续实验类型，不声称最优探索比例。`3` 是公开默认值，实验比较去掉保底及改为 1、7 的敏感性。收益主张来自实验表示和复用，而不是这个常数。

### 10.5 如何暂停，又不让工作无意义中断

同一固定 context 的最好有效 Treatment 损失下降，记为实现进展。失败后生成首个有效 Treatment 也记为一次进展；重复返回同一个程序不算进展。

连续两次 REVISE 没有产生新的有效改善时，将该 WorkItem 暂停。第一次非正结果仍有再次实现的入口。暂停不删除原代码或证据。

同一 context 只有出现以下新条件才自动重新进入可选集合：引用了此前未使用的实现依赖或源实验，或者在另一个 context 得到了新的有效实现。单纯重新陈述问题不重置计数。转向新的目标程序形成新的 context，但保留源实验与过去暂停原因。

模型也可以主动暂停，以状态更新表达，不生成“原样代码”占候选数。最终决策响应必须同时选出一个可执行的下一实验，不能通过连续 STOP 请求空转。

### 10.6 工程性降级

元数据格式错误最多纠正一次；仍错误时，采用当前最好程序上的 locked OPEN，只生成一个完整目标函数替换。它被记录为 fallback，不计作精细改动归因的成功。

这让暂时不会遵守结构化实验协议的模型仍能进行普通搜索，但不能把普通搜索产物包装为验证过的局部改动。

---

## 11. 评价、更新和失败处理

### 11.1 冻结评价条件

两臂使用相同训练实例、外部评价种子和资源协议。两臂从独立进程开始，不能共享可变全局状态。执行顺序用搜索种子的独立派生随机流决定。

共同种子控制了外部条件，但两种程序的随机调用顺序可能不同；不承诺自动得到最小方差的配对估计。

缓存只复用已经采用的训练观测，不代表获得一次新的独立证据。固定随机种子也不保证所有程序完全确定；依赖墙钟、自适应停止或不受控随机源的程序需标记 `stochastic_or_runtime_sensitive`。对此类程序，需要效果解释时重新执行必要的两臂，并记录真实评价成本；不把旧 Control 与新 Treatment 的一次偶然差异写成稳定效果。

### 11.2 何时更新

一个完整 paired trial 的两臂都完成或明确失败后，才更新实验状态与下一步选择。编译通过且成功评价的每份程序，都进入成果档案。

`locked` 对照是同协议的已有结果，只需执行 Treatment。`factorized` 需处理两个生成臂。

每个 trial 的开始时刻冻结 `reference_program_id` 和全局最好值。之后可同时记录相对来源程序和相对该时刻全局最好的收益，不能用未来最好值改写当时判断。

### 11.3 数值状态

数值容差定义为：

\[
\epsilon=10^{-12}\max(1,|L(P^-)|,|L(P^+)|).
\]

- 两臂有效、结构检查通过且 \(G_{focal}>\epsilon\)：`positive_here`；
- 两臂有效、结构检查通过且其余情况：`nonpositive_here`，保留零值与负值；
- 任一必要条件不满足：`unresolved`。

这是当前有限训练协议上的观测标签，不是统计显著性标签。新种子上的稳健性由专门实验报告，不能让 `epsilon` 冒充置信阈值。

### 11.4 不完整实验

| 情况 | 程序成果 | 实验判断 |
|---|---|---|
| ON 有效、OFF 无效 | ON 可以参与成果竞争 | 未识别；不能把无效 OFF 当作无限损失 |
| ON 无效、OFF 有效 | OFF 可以参与成果竞争 | 未识别；保留实现错误 |
| 两臂均有效但背景匹配失败 | 两者仍可作为普通候选 | 不计算受控局部效果 |
| 两臂代码相同 | 共享缓存与来源 | 记为 no-op，不记为新经验或进展 |
| 文本声称的计算与代码不符 | 实际合法程序仍有成绩 | 只保留代码级比较；语义标签待修正 |
| 控制条件不合法，例如删除必要可行性机制 | 有效候选可保留 | 对照不成立，不能晋升可迁移的正证据 |

### 11.5 修复

每个 trial 最多一次修复调用。修复针对交付或运行失败，使用明确错误。

如果修复只改变一个臂，另一个臂的背景与对照保持不变。若修复 Shared，原 paired trial 结束为 unresolved，建立新 context 并重新构建必要的两臂；旧结果不混用。

修复产生几个候选，就额外消耗几个槽。预算不足不启动修复。主系统不把多次修复藏在同一个免费 primary slot 中。

---

## 12. 成本与候选计数

### 12.1 主预算

默认每路 `candidate_budget=1000`。计数定义遵循“实际请求生成的完整候选”，不是所有方法都按迭代数计费。

| 操作 | 候选槽 |
|---|---:|
| 初始化输出一个程序 | 1 |
| locked：一个新 Treatment，直接引用已有 Control | 1 |
| factorized：同一请求构造两个臂 | 2 |
| REVISE：一个新 Treatment，固定已有对照 | 1 |
| TRANSFER locked | 1 |
| TRANSFER factorized | 2 |
| 重新生成的重复、解析失败或无效程序 | 按本次预定输出臂数计入 |
| 仅查询档案、选择动作、说明或元数据纠正 | 0 候选；模型与工具成本完整记录 |
| 修复 | 按新生成臂数追加 |

开始生成前预留相应槽位。factorized 的任一臂重复或无法解析，不退回预留槽；否则会奖励难交付的实验。明确引用的历史控制程序不是“再次生成”，因此不新扣候选槽。未经预先声明的重复结果仍扣费。

剩余一个槽时，只允许 locked 请求。无可用合法来源时停止，不能超预算。

### 12.2 三条成本曲线同时保留

- 最好质量—候选槽；
- 最好质量—真实完整程序评价数及候选 CPU 时间；
- 最好质量—累计模型 token 与实测服务成本。

缓存仅减少真实执行，不使“生成了重复程序”的候选消失。内部编译、检查和检索时间另记。美元成本只有实际服务费率与 usage 记录完备时计算，不以虚构统一 token 价格推算。

### 12.3 为什么不固定每个提案生成两份代码

若已有来源程序就是正确对照，再生成一份相同程序没有新增信息。若需要分离共享背景和焦点计算，第二臂才提供额外识别。

因此两种构建模式不是两套不同方法，而是同一实验语义在“已有对照”和“需要新对照”条件下的成本实现。

---

## 13. 完整运行过程

### 13.1 初始化

建立 8 个有效且源码不同的初始程序，初始化最多 32 次候选尝试，全部计入预算。各方法的初始化条件在对照实验中一致；能够共享 bootstrap 时，直接冻结并共享相同程序及成绩。

新正式实验不从历史强程序中偷偷注入种子，也不把旧任务或旧模型结果加入新协议的在线证据。任务提供的标准起始代码可以按所有方法共同协议使用。

重复单位仍是完整搜索运行。初始化失败不能被静默替换成历史强程序；若 32 次内只产生部分有效根，按实际根数继续；全部失败则明确返回初始化失败。

### 13.2 主循环伪代码

```python
initialize_archives_and_budget()
bootstrap_programs(target_valid=8, max_attempts=32)

while budget.remaining > 0 and archive.has_valid_program():
    board = build_fact_board(archive, ledger, work_items)
    allowed = eligible_actions(work_items, budget)
    if consecutive_non_open_trials >= 3:
        allowed = {OPEN}

    spec = same_model_choose_experiment(board, allowed, max_read_rounds=2)
    spec = validate_ids_and_freeze_comparator(spec)

    if spec.is_malformed_after_one_correction:
        spec = locked_whole_function_fallback(archive.best)

    required = 1 if spec.mode == LOCKED else 2
    if required > budget.remaining:
        spec = choose_one_slot_locked_fallback_or_stop()
    if spec is STOP:
        break

    snapshot = freeze_context_protocol_and_rng(spec)
    budget.reserve(required)
    artifacts = same_model_implement(snapshot)
    arms = build_and_validate(artifacts, snapshot)
    outcomes = evaluate_or_reuse_explicit_control(arms, snapshot)
    save_all_raw_artifacts_and_costs()

    if needs_delivery_or_runtime_repair(outcomes) and repair_is_affordable():
        outcomes = repair_once_with_explicit_budget(outcomes, snapshot)

    archive.add_every_valid_program(outcomes)
    evidence = compute_effects_only_if_pair_is_valid(outcomes, snapshot)
    ledger.append(evidence)
    update_fixed_context_work_item(evidence)
    update_open_cadence(spec.action)
    checkpoint_all_state()

finalists = training_top5_unique_programs()
reevaluate_on_training_under_frozen_protocol(finalists)
freeze_best_valid_finalist()
run_held_out_test_once()
```

### 13.3 必须保持的状态不变量

1. 每个局部效果都能追溯到两份完整代码和相同协议。
2. 对照只在看分数之前确定，不能事后选择最差臂制造最大差值。
3. 同一 context 的 Shared 与 Control 不变；变更时新建 context。
4. 源环境的效果不能直接成为目标环境的效果。
5. 程序总分与实验标签由不同字段管理，不相互覆盖。
6. 工作版可以落后，最终交付仍只选真实最佳合法程序。
7. 失败、修复和重复都有成本；暂停不要求生成重复代码。
8. 模型可以解释证据，不能写入未经执行的评分。
9. 任一缓存结果都带完整身份，旧协议不可复用。
10. `change_id` 的关系必须显式记录，不能仅由语义相似度自动合并。

---

## 14. 默认配置

这些是 r1 的确定默认值，不是宣称经实验得到的最优参数。

```yaml
mechanism_id: traceaad-v12.1-executable-experiments-r1

search:
  candidate_budget: 1000
  initial_valid_programs: 8
  initial_attempt_cap: 32
  actions: [open, revise, transfer]
  max_consecutive_non_open_trials: 3
  pause_after_nonprogress_revisions: 2
  repair_calls_per_trial: 1
  final_candidates: 5

experiments:
  modes: [locked, factorized]
  default_mode: locked
  freeze_control_within_context: true
  comparator_selected_before_scoring: true
  relative_numeric_tolerance: 1.0e-12
  invalid_arm_has_numeric_score: false
  semantic_auto_merge: false
  source_effect_reused_as_target_effect: false

context:
  initial_experiment_cards: 8
  card_groups: [recent, masked_positive, progressing, rotating_history]
  cards_per_group: 2
  max_read_rounds: 2
  records_per_read: 4
  max_input_tokens: 24000
  decision_output_tokens: 1024
  implementation_output_tokens: 8192
  full_source_required_before_spec_commit: true

model:
  backend: frozen_run_model_spec
  temperature: 1.0
  top_p: 0.95
  enable_thinking: false
  same_model_for_decision_and_implementation: true

protocol:
  task_suite: co6
  training_instances: 16
  held_out_instances: 50
  evaluation_revision: research-protocol-v6-20261012-raw-objective
  function_budget_scope: mean_instance
  fixed_framework_mean_cpu_seconds: 2
  full_solver_mean_cpu_seconds: 10
  instance_wall_seconds: 120
  count_repairs_as_candidates: true
  record_actual_evaluations_and_all_model_costs: true
```

若模型服务无法承受声明的上下文，应在实验启动前统一更改配置并记录身份；不能在某些臂中静默截去必要 Control 或源代码。输入过长时先删除未选卡片，再删除旧的解释文本。共享背景、选定两臂、比较对象和必要接口不能删。仍过长则改为可容纳的较小实验或暂停，不伪造已完成比较。

---

## 15. 接入当前仓库

建议新建独立目录，不通过多层继承重用旧版本的调度隐含规则：

```text
traceaad/v12_1/
  config.py
  schema.py                 # Program / Change / Experiment / WorkItem
  builder.py                # AST locators; locked 与 factorized 构建
  controls.py               # 对照来源与共享背景约束
  archive.py                # 程序身份、缓存、最终成果
  ledger.py                 # append-only 实验与效果记录
  retrieval.py              # 事实卡、查询、来源追踪
  prompts.py
  controller.py             # 三动作、暂停与探索保底
  traceaad.py               # 主循环及事务边界

experiments/traceaad_v12_1/
  run.py
  launch_host.py
  report.py
  audit_pairs.py

tests/method/
  test_traceaad_v12_1_builder.py
  test_traceaad_v12_1_evidence.py
  test_traceaad_v12_1_budget.py
  test_traceaad_v12_1_resume.py
```

共享使用现有模型客户端、任务注册、评价器、CPU 调度、成本记录和最终测试工具。具体 API 名称以现有接口实现为准；上表是职责划分，不声称文件已经存在。

### 15.1 不复用的旧机制

不继承 V10.17 的质量乘成功率选父、随机 1/8 三步开发、按 Design 近邻认证同计算、修复不计 primary 的旧逻辑，以及 V10.24 的文本 edit 结束协议。

保留原始评价数据与历史工件，但不将旧记录改写为 V12.1 的合法对照。旧父子边只有在代码和比较条件满足本规范时，才能作为另一个明确标记的离线重建实验；不能直接加入新正式搜索。

---

## 16. 最小实现验收

实现验收检查以下事实，而不是只检查模型是否返回了几个字段。

| 检验 | 预期 |
|---|---|
| locked 回退 | 重建 OFF 的 AST 与来源程序完全一致 |
| 共享背景 | 两臂不同的 AST 节点全部属于声明包 |
| 隐式依赖 | Treatment 引用未声明的 helper 时拒绝构建 |
| 非法对照 | 删除必要合法性逻辑不能认证为中性对照 |
| 实际分数 | 用小型确定性测试任务验证三个收益的恒等式 |
| 有效单臂 | 任一臂失败时仍保存另一臂，但不生成数值效果 |
| 低分正效应 | 构造 B 优于 ON、ON 优于 OFF 的例子，确认改动仍能被检索和迁移 |
| 伪正贡献 | ON 优于 B、OFF 又优于 ON，确认不把焦点标成正贡献 |
| 换背景 | 源效果为正、目标效果为负，两者分别保存 |
| 重复与计费 | factorized 重复仍扣两槽；显式历史 OFF 不重复扣槽 |
| 修复 | Shared 变更必须新建 context，不能沿用旧对照分数 |
| 暂停 | 不输出重复程序表达 STOP，不出现无限零候选循环 |
| 崩溃恢复 | pending 两臂不被重复请求；已预留槽与已完成评价不重复计数 |
| 数据隔离 | 无测试反馈、跨实例状态或旧协议结果进入搜索 |

这些验收只证明实现符合规格，不证明搜索有效。

---

## 17. 实验设计：先检验机制特有的预言

### 17.1 E0：确认“总分与局部作用不一致”确实存在

使用当前模型和当前协议，在预先固定的搜索状态上随机抽取复合改写。不能只回溯最终成功程序。按源程序质量与候选退步程度分层，但保留每层分母。

对每个可分离焦点建立合法对照。先完成代码与对照审查，再读效果标签；审查者尽量不知道哪臂更好。

报告：

- 能形成合法对照的比例；
- 不可分离、依赖不满足与语义失配比例；
- \(G_{total}\le0, G_{focal}>0\) 的比例；
- \(G_{total}>0, G_{focal}\le0\) 的比例；
- 各类变化中的分布，而不是只给全池均值；
- 新种子上的方向稳定性，作为独立补充复评，不偷偷修改在线训练协议。

r1 的筛查规模固定为六任务各 12 个冻结状态，早、中、晚预算位置各 4 个；在可用程序质量层内预先随机取样。每状态提出 2 个 factorized 实验，共 144 对。来源至少覆盖每任务三路基础搜索，统计按来源运行与锚点分层，不把 288 个程序当成 288 次独立搜索。

这一步的关键不是证明“负程序也能有更负的对照”，而是检查比较对象合理，并且这种不一致能预测后续有用行动。

### 17.2 E1：固定迁移器，测试证据是否帮助选对改动

冻结相同的源实验池、目标程序池和 TRANSFER 提示。只改变候选改动的选择依据：

- 源完整程序质量；
- 源对照效果与适配条件；
- 来源和成本匹配的随机选择。

所有被选源代码都由同一迁移器处理，目标 ON 与目标 OFF 都评价。主指标是目标程序的净改善，不是源局部增益。

r1 使用至多 80 个冻结的目标状态，每个状态先按接口与依赖建立同一候选池，再向三种选择规则分别提供规定信息。候选池不能预先只保留正效应改动，否则无法公平评价总分选择。每种规则每状态实施一次迁移；因合法候选不足而缺失的状态逐项报告，不在看结果后补选容易成功的目标。

这个实验直接区分“多了一种 crossover 提示”与“对照证据让系统选择了更有用的迁移对象”。

### 17.3 E2：一份对照与另一份独立实现，哪种预算用途更好

固定提案与背景，比较：

- 一份实现＋一份匹配 Control；
- 两份独立实现；
- 一份受限、尽量不改无关部分的普通编辑，并把省下的预算用于下一次编辑。

每条臂使用相同可计费候选上限；所有有效程序都可参与最终择优。不让对照组少拿候选，也不让 V12.1 独占完整评价反馈。

读取的不只是即时最好分，还包括用这些记录作后续 REVISE 或 TRANSFER 后的净产出。r1 选 48 个独立于 E0 调参状态的提案／背景，每个比较臂总计 4 个候选槽：前 2 槽取得各自的初始证据，后 2 槽用于继续开发或迁移。模型调用与 token 单独对齐报告，不把同样的候选数说成全部成本相同。

### 17.4 E3：实验记录是否优于摘要记忆

保持候选程序、必要对照计算和预算不变，只替换后续上下文：

- 完整实际改动＋配对结果；
- 同 token 量级的语义改动摘要＋整份程序分数；
- 完整代码与分数，但不显示哪个背景与哪组结果构成配对。

所有臂均支付相同评价成本，避免把“少花了一次对照评价”混进记忆作用。

### 17.5 主实验

先在现有六任务协议上筛查，三路种子用于工程筛选。论文正式比较预先固定为至少五路搜索重复，使用两个独立生成模型；参数冻结后不按任务单独调 controller。

比较组优先级：

1. V12.0 base 与 linking，恢复后的 V9.17 简洁控制基线；
2. 受限编辑＋普通 crossover，以及 V9.21 式独立双实现；
3. EoH、ReEvo、MCTS-AHD、ShinkaEvolve 等现有可用基线；
4. 最接近的新工作 LabBook、CausalEvolve、DeltaEvolve、RefineEvo，依据可取得的正式代码进行忠实复现。没有公开实现时，明确标注按论文重建，不能把重建版本当作者官方结果。

不是把全部历史版本都重新跑一遍。历史版本用于恢复有解释力的对照，近期方法用于检验真实增量。

主报告同时覆盖：最终同规模测试质量、预算曲线、模型成本、有效率与对照合法率。扩大规模属于附加泛化实验，不能改变主协议后再与历史数值直接排名。

### 17.6 最有说服力的一张机制图

建议展示同一条可核验链：

```text
低分源程序
  ├── 对照更差：定位具体有效计算
  ├── 保留该计算的代码与依赖
  └── 迁入另一个高分程序
         ├── 目标对照：无该改动
         └── 目标处理：有该改动，真实提高最终质量
```

旁边给出总体发生比例、目标迁移成功率和预算成本，不只展示一个挑选出的故事。

---

## 18. 论文主张与叙事

### 18.1 最强的问题表述

> LLM 算法搜索通常借助已评价程序推进，但完整程序的表现并不总能说明其中具体改动的作用。程序级选择因此可能遗漏有用的计算，也可能把背景带来的改善误记为焦点改动的收益。我们研究能否把算法设计轨迹转化为可执行、可复验的改动实验，并通过目标环境中的重新检验累积改进。

“通常”不等于所有方法都没有记忆、反思或多次实现；相关工作中已有这些能力。本文聚焦的是可执行比较和复用通道。

### 18.2 三个候选贡献

**经验发现。** 在前瞻抽样的算法搜索中量化程序总收益与焦点改动效果的不一致，并检验这种不一致是否对应被常规程序级筛选遗漏的迁移机会。

**方法。** 提出可执行改动实验表示，使用共享背景与显式控制构建比较；把程序成果与改动证据分开管理；在目标程序中复验迁移后的效果。

**结果。** 在固定函数与完整求解器任务、不同模型及明确的成本预算下，检验该闭环相对程序复用、语义记忆和独立重采样的增量。

第三项只有在真实结果出来后才能写成完成式。方法规格可以现在冻结，结果性结论不能预支。

### 18.3 不承担的主张

不估计任意算法思想的“真实潜力”；不保证语义等价识别；不声称获得通用因果规律；不将共同种子当作消除全部噪声；不把每项模块都包装为首次提出。

论文最有力的表达应是**发现并利用一种可测的程序级信用盲点**，而不是“我们首次让 AI 像科学家一样思考”。

### 18.4 与 V12.0 的根本区别

V12.0：已经准备写一个计算，找到看起来属于同一计算的旧程序，帮助这一次实现。

V12.1：一次程序变化提供了什么可复验的改动证据？该证据说明应继续完善原实现，还是把其中的有效计算接到另一个程序上？

前者改变一份实现能看见的参考；后者改变搜索产生、保存和再使用经验的单位。V12.1 不要求模型重复提出同一计算，系统也能主动让一项已经有局部证据的改动跨背景继续发展。

---

## 19. 关键决策与失败信号

本节用于改变下一步研究决策，不用于提前削弱论文。

**第一优先检查：有效局部作用能否迁移。** 如果 E0 找到大量正局部效果，E1 却不优于按总分选改动，那么不能用“发现了隐藏价值”解释完整搜索。应检查适配条件或比较对象，而不是增加一堆潜力特征。

**第二优先检查：对照是否只是制造了弱基线。** 若收益依赖把 Control 写坏、删掉必要机制或大幅改变计算限制，应修正实验构造。合法可执行不自动等于有意义的比较。

**第三优先检查：普通受限编辑是否已经足够。** 如果保持成熟背景、只改目标计算的简洁方法获得同样收益，那么额外 factorized 构造没有独立价值。保留已经有效的可执行改动复用，移除没有作用的比较开销。

**第四优先检查：调度是否真正使用了证据。** 记录模型每次选择引用的实验 ID、源代码和目标。若它忽略 Control，只选高分 ON，则系统实际上退化为普通程序检索，应先调整信息通道或简化决策，而不是声称理论机制已生效。

这些信号区分的是实验构造、复用和控制三处原因，不把一个整包结果解释为整个研究方向成立或终结。

---

## 20. 最终规格摘要

V12.1 保留 TraceAAD 最初关心的“过去试过什么，以及它怎样改变下一步”。但它不再要求模型只凭过去的文字和总分，猜出每个计算的价值。

它把一次设计尝试变成一个明确的可执行比较，把较差程序中的有效改动和较好程序中的无效归因区分开来。随后，系统可以继续实现这项改动，也可以把它带到新的程序背景中重新检验。

**一句话：V12.1 不只继承成功程序，而是继承经对照得到的改进证据，并通过再实施让这些证据产生新的程序成果。**

---

## 附录 A. 决策与实现提示契约

以下为可直接转写进 `prompts.py` 的语义契约。代码内容用代码块提交，少量元数据用结构化对象提交；不要求模型生成庞大的嵌套日志。

### A.1 决策提示

```text
You design algorithms under a fixed candidate budget.
Choose one next experiment: OPEN, REVISE, or TRANSFER.

Use measured program results and executable comparisons.
A program's total score is not the measured effect of every computation in it.
A positive effect in one shared background is not evidence of a positive effect
in a different background.

OPEN: define a new computation change and a meaningful executable comparator.
REVISE: improve the treatment implementation of one existing experiment.
        Keep its shared background and comparator fixed.
TRANSFER: apply an existing, positively measured change to a compatible program.
          Define and evaluate the comparison again in the target program.

Read the exact source code and source experiment before committing to them.
Do not choose a source only because its whole program has a high score.
Do not assign numerical results that have not been measured.
Do not discard a whole idea because one implementation failed.

Return:
- action
- source program or experiment IDs
- target program ID
- one concrete change to test
- control provenance and control computation
- locked or factorized mode
- required dependencies and intended code boundary
- one reason this experiment is preferable to an available alternative

Keep the decision specific. A change may be small or may replace a whole method.
Do not add a component just to make the design appear novel.
```

### A.2 locked 实现提示

```text
Implement the selected change in the frozen source program.
The source program is the control.

Return complete replacements for the declared AST locations and any declared
private dependencies. Do not change code outside this bundle.
The host will build the new standalone program from these replacements.

The experiment measures this actual replacement bundle, not a broader claim.
If the requested change needs a larger boundary, report the required boundary
before implementation; do not silently alter the shared code.
```

边界扩大需重新提交规格；相应模型成本计入，未生成代码不扣候选。每个实验最多允许一次边界修订，之后使用整体函数替换或暂停，避免无限规划。

### A.3 factorized 实现提示

```text
Build two standalone programs from one shared background and two alternative
implementations of the declared computation.

Return three code blocks: Shared, Control, Treatment.
Shared must be identical for both arms.
Control must implement the predeclared comparator from its stated source,
or the predeclared valid neutral operation.
Treatment must implement the new computation.

Do not weaken feasibility, remove a necessary operation, or intentionally damage
Control. Both arms must satisfy the original task contract.
Do not expose an arm flag, alternate-arm code, or evaluation metadata to either arm.
Include all dependencies. The host will compile and evaluate both arms.
```

### A.4 TRANSFER 补充提示

```text
Reuse the executed change from the cited source experiment, not merely its name.
Map its inputs, outputs, and dependencies to the target program.
Keep mature target computations that the change does not replace.

The source effect is conditional on the source background.
Measure the target effect with its own control.
Do not claim success from copying a donor or from merely beating a weak source.
```

---

## 附录 B. 事实卡示例

以下数字同样是机制示例，不是运行输出。

```yaml
experiment_id: e37
change_id: h12
implementation_version: 2
context_id: ctx9
mode: factorized
reference_program_id: p18
question: Compare a new route postprocessor with identity postprocessing.
control:
  provenance: registered_identity
  program_id: p51
  score: 6.45
treatment:
  program_id: p52
  score: 6.20
reference_score: 5.80
effects:
  total: -0.40
  focal: 0.25
  background: -0.65
status: positive_here
interpretation_scope: this code bundle in ctx9 under the recorded protocol
available_next_steps:
  - revise treatment without changing ctx9
  - transfer the executed postprocessor to a compatible program
```

`available_next_steps` 由规则生成；它不声明哪一步必定有效。卡片中的解释范围不可被模型更新为全局结论。

---

## 附录 C. 证据来源

### C.1 TraceAAD 原始研究记录

以下历史统计引用文档报告；本设计没有重新执行旧搜索或逐条复算旧原始日志。

- **[H1]** V1–V9.4 机制诊断。固定 Git 版本 `0814dc72ff1277f01d98412f96a8955a7eadbc90`。https://github.com/Fangziyang0910/TraceAAD/blob/0814dc72ff1277f01d98412f96a8955a7eadbc90/docs/analysis/TraceAAD-V1-V9.4机制诊断.md
- **[H2]** V9.8 机制诊断，同一固定 Git 版本。https://github.com/Fangziyang0910/TraceAAD/blob/0814dc72ff1277f01d98412f96a8955a7eadbc90/docs/analysis/TraceAAD-V9.8机制诊断.md
- **[H3]** 2026-08-24 自适应调度。https://github.com/Fangziyang0910/TraceAAD/blob/main/docs/03-现象与检验/2026-08-24-自适应调度.md
- **[H4]** V9.21 机制首跑分析。固定 Git 版本 `bbf4b61dbe9c25adbea6e27bd96b80b6f46a7db5`。https://github.com/Fangziyang0910/TraceAAD/blob/bbf4b61dbe9c25adbea6e27bd96b80b6f46a7db5/docs/analysis/机制分析/TraceAAD-V9.21机制首跑分析.md
- **[H5]** V9.22 完整机制设计。固定 Git 版本 `c3bc08e1dd79262b2d7826edcefbc15f0f84a5a3`。https://github.com/Fangziyang0910/TraceAAD/blob/c3bc08e1dd79262b2d7826edcefbc15f0f84a5a3/docs/methods/TraceAAD-V9.22完整机制设计.md
- **[H6]** 2026-10-04 短程改写诊断与首版门槛。https://github.com/Fangziyang0910/TraceAAD/blob/main/docs/03-现象与检验/2026-10-04-短程改写诊断与首版门槛.md
- **[H7]** 2026-10-10 V10.24 完整搜索与开发成本。https://github.com/Fangziyang0910/TraceAAD/blob/main/docs/03-现象与检验/2026-10-10-V10.24完整搜索与开发成本.md
- **[H8]** V12.0 当前实现。https://github.com/Fangziyang0910/TraceAAD/blob/main/traceaad/v12_0/traceaad.py
- **[H9]** 当前实验规范。2026-10-13 读取，作为第 14 节任务及资源默认条件的来源。https://github.com/Fangziyang0910/TraceAAD/blob/main/docs/08-实验规范与共识/README.md
- **[H10]** AAD 代表方法与机制，仓库综述索引。https://github.com/Fangziyang0910/TraceAAD/blob/main/docs/05-相关工作/AAD代表方法与机制.md

### C.2 相关工作

本轮重点核对了与表示、实验、复用最接近的原文。下列列表给出定位资料，不表示穷尽领域或已经排除全部同期机制。

- **[R1] EoH.** Evolution of Heuristics. https://arxiv.org/abs/2401.02051
- **[R2] FunSearch.** Mathematical discoveries from program search with large language models. https://www.nature.com/articles/s41586-023-06924-6
- **[R3] AlphaEvolve.** https://arxiv.org/abs/2506.13131
- **[R4] ShinkaEvolve.** https://arxiv.org/html/2509.19349v1
- **[R5] ReEvo.** https://arxiv.org/html/2402.01145v1
- **[R6] RefineEvo.** Planning-Guided Heuristic Evolution with Bidirectional Experience. https://arxiv.org/html/2607.11358v1
- **[R7] EvoMem.** https://arxiv.org/abs/2608.10795
- **[R8] DeltaEvolve.** Accelerating Scientific Discovery through Momentum-Driven Evolution. https://arxiv.org/html/2602.02919v1
- **[R9] PhyloEvolve.** https://arxiv.org/html/2601.14523v1
- **[R10] QUBE.** https://arxiv.org/abs/2412.20694
- **[R11] MCTS-AHD.** https://arxiv.org/abs/2501.08603
- **[R12] Clade-AHD.** https://arxiv.org/html/2602.00549v2
- **[R13] PAEvo.** Plan-Algorithm Evolution with LLMs for Automatic Heuristic Design. Springer 公开摘要；本轮未读取其付费全文，具体重采样算子不作为本文新颖性判断的唯一依据。https://link.springer.com/chapter/10.1007/978-3-032-36217-9_20
- **[R14] PathWise.** Planning through World Model for Automated Heuristic Design via Self-Evolving LLMs. https://arxiv.org/html/2601.20539v1
- **[R15] Back to the Beginning of Heuristic Design.** Bridging Code and Knowledge with LLMs. https://arxiv.org/html/2605.06123v1
- **[R16] LabBook.** Harnessing Experimental History for Efficient LLM-Driven Discovery. https://arxiv.org/html/2610.00675v1
- **[R17] CausalEvolve.** Towards Open-Ended Discovery with Causal Scratchpad. https://arxiv.org/html/2603.14575v2
- **[R18] HeurAgenix.** Leveraging LLMs for Solving Complex Combinatorial Optimization Challenges. https://arxiv.org/html/2506.15196v2

- **[R19] GP-GOMEA.** Improving Model-based Genetic Programming for Symbolic Regression of Small Expressions. https://arxiv.org/abs/1904.02050
