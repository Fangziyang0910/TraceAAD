# BaSE：Compute Allocation in Evolutionary Search

- 论文：Compute Allocation in Evolutionary Search: From Depth–Breadth to Multi-Armed Bandits（arXiv:2605.29268）。本地来源：main.tex、bandit.tex、bandit_appendix.tex。
- 设计对象：固定 LLM 调用预算下的计算分配，包含单条 run 内部的深度与广度划分，以及多条并行 run 之间的调用路由。模型、提示与 evaluator 在全部实验中保持不变。

## 1. 问题与设计对象

论文针对的是报告口径与部署条件之间的落差。\*Evolve 系统普遍只报告多次运行中的最好一次（FunSearch 报 4/140 命中率，CodeEvolve 只展示最好结果，AlphaEvolve 在 $n=26$ Circle Packing 上报单一数值），而单次运行成本从约 150 次调用到 204,800 个候选，跨越两个数量级。作者由此提出，这些数字刻画的不是固定预算下应当期望的结果：

> existing reports characterize what is achievable on a favorable run, not what a practitioner should expect at a finite computational cost.（`main.tex`，Introduction）

在此基础上，论文把分配确立为与模型能力、提示质量并列的第三个设计维度，并主张它与前两者正交：

> A stronger base model can generate better mutations; a more informative prompt can guide the search toward more useful edits; and, importantly, allocation determines how the evolutionary process balances exploration and exploitation. This allocation effect is orthogonal to model and prompt quality, and remains meaningful once the model, prompt, and evaluator are fixed. Better allocation can improve expected outcomes by avoiding both premature commitment to weak trajectories and excessive breadth without refinement.（`main.tex`，Introduction）

论文因此采用经典进化计算的 fixed-budget 视角，把预算 $C$ 定义为 LLM 调用次数，并依次回答两个问题：固定预算应当如何划分为深度与广度，以及这一划分能否离线预先算出。

## 2. 形式化与实验口径

在 greedy 协议下，一次运行由预算的划分完全确定：$C = T \cdot N$，其中 $T$ 为代数，$N$ 为每代并行子代数。$T=1$ 退化为 best-of-N，$T=C$ 为纯串行细化。记 $V(C,T)$ 为预算 $C$、深度 $T$ 时的期望最佳适应度，目标为 $T^{*}(C) = \arg\max_T V(C,T)$（`eq:objective`）。greedy 协议本身是单锚点爬山：始终以历史最佳程序为父代，子代只有严格超过当前最佳才被接受。这一接受规则决定了后文观察到的停滞形态。

| 项 | 设置 |
| --- | --- |
| 任务 | Circle Packing（$n=26$）、MinMaxDist（$n=16$）、Heilbronn Triangle（$n=11$），取自 AlphaEvolve，沿用 OpenEvolve 的 evaluator 与初始程序；按已发表最好构造归一化，$1.0$ 对应当前最好水平 |
| 模型 | Qwen3 1.7B/4B/8B/14B（thinking 模式）与 Llama-3.1-8B；temperature $0.6$，top-p $0.95$；vLLM fp8 服务，H100，最多 16 路并发 |
| 扫描 | greedy 协议，$C$ 取 8 至 512，$T \in \{1,2,4,\dots,C\}$，每格 10 个种子；1.7B/4B 上限为 $C=128$ |
| 对照 | island 协议的 OpenEvolve、CodeEvolve、ShinkaEvolve，均在 $C=512$，每格 10 次端到端运行 |
| 统计 | 分层 bootstrap，1000 次重采样报标准误与 95% 区间，遵循 rliable 协议 |
| 成本轴 | 每次调用记 $F = 2 P_{act} (p_{prompt} - p_{cached} + p_{out})$；固定 $C$ 时 FLOPs 随 $T$ 的变化不超过 15%（$R^2 \ge 0.94$），故模型内以调用数计价、跨模型以有效 FLOPs 计价 |

## 3. 分配是否构成独立杠杆

论文给出的第一层证据是适应度—计算包络。在每个预算上取扫描所得的最好适应度 $V_{\max}(C)$，它随计算量平滑单调上升，未出现相变或反转。实线包络与虚线 best-of-N 基线之间的垂直间距，即多代细化相对一次性并行采样的收益：$C=128$ 时 CP-8B 为 $+0.119$、MMD-8B 为 $+0.102$，约为归一化范围的十分之一；在未饱和设置上持续（MMD-14B 在 $C=512$ 仍有 $+0.115$），在饱和处收缩（MMD-8B 为 $+0.016$）。作者补充说明，无选择偏差的检验在 MMD 上确认了真实起始点，在 CP 上没有观察到明确阈值（`fig:flops_envelope`）。

第二层证据界定了这一杠杆的适用范围。按 model × task 格子做深度收益的置换检验（打乱各种子在不同 $T$ 上的适应度，20,000 次重排），深度收益 $\mathrm{penBoN} = V_{\max} - V_{T=1}$ 只在有能力的模型仍有提升空间时出现：Qwen3 1.7B 在两个任务上都不显著（$p \ge 0.16$），4B 在 MMD 上显著（$+0.213$，$p=0.002$）而 CP 上不显著（$p=0.30$），8B 在未饱和的 HT 上给出全表最干净的收益（$0.339 \to 0.672$，$+0.333$，$p=0.002$）。阈值按 (model, task) 而非按模型划分（`tab:effect_model_size`）。

Llama-3.1-8B 是最强的反面证据：CP 上 best-of-N 已达 $0.843$ 而深度毫无增益（$\mathrm{penBoN} = 0$，$p=1.00$），MMD 上名义 $+0.170$ 与最大值选择噪声不可区分（$p=0.23$），HT 上深度确实累积（$0.047 \to 0.253$，$p=0.013$）但绝对水平远低于 Qwen3-8B 在同一任务上的 $0.672$。在 Llama HT 这一格，OpenEvolve 与 CodeEvolve 的终局适应度为 $0.0$，即 512 次调用内没有任何运行产出合法构型。作者据此把能力与提示确立为分配无法逾越的上界：

> Allocation amplifies an existing signal but cannot create one. […] Capability and prompt thus jointly determine the achievable ceiling; allocation governs how efficiently a method approaches it.（`bandit.tex`，Discussion）

提示构成的是一条独立于能力的上界，作者用基线可比性给出了具体例证：ShinkaEvolve 在 CP 上的系统提示直接建议使用 `scipy.optimize`，而本文的 greedy 使用与 OpenEvolve 相同的提示，并不含此类捷径提示；ShinkaEvolve 的 HT 提示则包含一条其他基线都没有的共线三元组退化警告，恰好对应 case study 诊断出的失败模式。读取 CP 上的基线排序时需要带上这一差异。

跨模型比较还给出一项从属结论：以有效 FLOPs 重新计价后，能力排序基本消失。MMD 上 4B、8B、14B 的包络几乎重合（$R^2=0.94$），CP 上 8B 与 14B 在未触顶区间重合（$R^2=0.93$）。在该范围内，模型规模与搜索预算表现为替代关系。该塌缩同样受上述能力闸门限制。

## 4. 分配的几何及其机制解释

对未触顶格子（$V<0.97$）拟合

$$
\log(1-V) = \beta_0 + a\log T + b\log N + c\log T\log N ,
$$

仅含预算的模型（$c=0$、$a=b$）得 $R^2 \approx 0.74$–$0.78$，双线性形式提升至 $R^2 \in [0.75, 0.92]$（`eq:bi-law`、`tab:production_law`）。$a$ 与 $b$ 均为负，说明深度与广度各自都在缩小与上限的差距；决定几何形态的是交互项 $c$：$|c|$ 接近 0 时预算切片近似可分，最优点落在全深度的角上并形成宽平台；$c$ 明显为负时切片向内弯折，最优点为深广均衡的内部脊。MMD 在两个有能力的模型上都落在后一极限（8B $c=-0.106$，$p=1.2 \times 10^{-11}$；14B $c=-0.057$，$p=5.0 \times 10^{-4}$），CP 两行的 $c$ 与 0 不可区分（8B $-0.027$，$p=0.49$；14B $+0.007$，$p=0.77$）。

由该形式可代数推出平台半宽 $|\delta| \le \sqrt{\Delta/|c|}$：$|c|$ 小则错过 $T^{*}$ 的代价低，$|c|$ 大则曲率信号强、$T^{*}$ 更易从在线反馈识别。作者明确标注这一推导是双线性假设的代数结果而非独立拟合的标度律，并指出经验上 $T^{*}(C)$ 跨模型嘈杂且非单调，而同一预算轴上的 $V_{\max}(C)$ 平滑且按能力排序：

> The fitness value at the optimum is therefore stable even when its precise location is not — the surface is searchable in the sense that what online search achieves does not depend sensitively on identifying $T^{*}$ exactly.（`main.tex`，附录 `app:plateau_width`）

这一条同时说明分配不宜离线预先算定，而应在线搜索，构成后文 BaSE 的合法性论证。

任务之间几何差异的机制解释来自代码级 case study（Qwen3-8B，$T=512$、$N=1$，10 次运行）。三个任务呈现同一模式的不同实例。CP 上程序分为手写环形构造与调用 `scipy.optimize` 两族，前者因半径均匀而在角落浪费空间，内在上限约 $0.89$；所有终局超过 $0.99$ 的运行都锚定在 scipy 程序上。健康运行在第 13 代首次写出 scipy，单代由 $0.5584$ 升至 $0.8396$；停滞运行在手写族内消耗 437 代（$0.36$ 升至 $0.67$ 后停滞 233 代，再至 $0.82$ 后停滞 110 代），第 438 代出现首个 scipy 子代，单代由 $0.822$ 升至 $0.983$。MMD 上符号翻转：存在解析吸引子，即 5 点内环加 11 点外环的正多边形构造，不含任何优化器而恰好取到 $0.9603$；10 次运行中越过该吸引子的 3 次全部不含 scipy，落在 $0.58$–$0.95$ 的 6 次全部调用 `minimize` 或 `differential_evolution`。原因是 $-(d_{min}/d_{max})^2$ 高度非凸，数值优化器从通用初值收敛到局部最优，而 greedy 一旦接受它，后续子代必须一次性超过该具体的随机局部最优。两侧均值印证了符号翻转：CP 上 scipy 程序平均高 $0.35$（$0.99$ 对 $0.64$），MMD 上低 $0.14$（$0.59$ 对 $0.73$）。HT 没有解析吸引子，且初始程序返回 11 个零、得分为 $0.0$；健康运行的锚点使用 SLSQP、100 次随机重启、三条边的硬不等式约束与对 165 个三元组面积的 log-sum-exp 软最小，停滞运行则以软惩罚处理硬可行性约束，其后 466 代中有 465 个子代因越界被评为 $0.0$。

作者把这一机制命名为 asymmetric proposal mass，并用它解释 $|c|$ 的任务依赖：

> each task admits one high-fitness algorithmic family, but the LLM's base rate on it appears differ across tasks, so where the good family is rare, breadth raises the probability that at least one parallel trajectory anchors on it before depth can refine within it.（`main.tex`，`sec:production_law`）

> Across all three tasks, outcomes are gated by which algorithmic family the run anchors on in the initial generations. Breadth and depth play complementary roles here: breadth gives more parallel attempts at finding a viable family, while depth refines within the family once anchored.（`main.tex`，附录 `app:case_study`）

需要注意其证据地位：该解释由代码级个案与两侧均值支持，不是受控消融，且限于三任务、单模型。

## 5. 从固定分配到在线分配

即使在扫描出的最优 $(T,N)$ 上，固定分配仍留有一层无法消除的方差。同一配置重复 10 次，终局适应度构成分布而非单值：CP 上 greedy 稳定爬升，MMD 与 HT 上部分运行早早停滞。附录把该散布追到源头：在 $T=1$ 时所有子代共享同一父代，单次变异产生的子代适应度已铺满 $[0,1]$，run 间散布是这种单步散布沿 $T$ 代的累积（`app:distribution`）。作者据此把一次运行视为由 $(model, task, C, T)$ 决定的潜在分布的一次抽样，并说明需要第二层分配杠杆：

> Evolving may dramatically fail from time to time by converging at a low score without further improvement even under the same configuration, producing a distribution rather than a single value, a spread that within-run allocation cannot remove. A second allocation lever is therefore needed, one that operates between trajectories rather than within them.（`bandit.tex`，开篇）

BaSE 即这一杠杆：维持 $K$ 条并行轨迹，各自作为一个臂，都从同一 seed program 起步并先各调用一次 LLM 取得初始候选与适应度；此后每次调用由策略选出一条轨迹推进一步，评价结果反馈后重新选择，预算耗尽时从所有轨迹产生过的候选中取全局最好（`alg:base`）。回报随细化步数变化而非平稳，正文以 Improving Bandits 为骨架，比较 UCB、EXP3.P、Thompson 与一个随机分配基线；此处的随机指随机选择轨迹，而非随机选择父代。

将 $C=512$、$K=10$ 下的三种花法并置，动机即清楚：全部预算押在一条轨迹上时，抽到停滞轨迹会浪费全部 512 次调用；十条各分 51 次时多样性提高，但优劣轨迹获得同等预算；BaSE 在前期铺开辨认哪些轨迹仍在改善，在后期把剩余预算向其倾斜。这构成轨迹层面的第二次探索—利用权衡，与轨迹内部的那一次处于不同层级。

论文明确限定了该机制的作用范围：

> the bandit policy only decides which trajectory should reveal its next point in an online manner. It does not alter the local refinement rule or change how a selected trajectory evolves. In this sense, BaSE improves search by reallocating computation across heterogeneous trajectories rather than by modifying the trajectory dynamics themselves.（`bandit.tex`，`sec:MAB_for_trajectory_allocation`）

因此 BaSE 与父代采样协议正交：算法中的 prompt generator 可直接替换为 OpenEvolve、CodeEvolve 或 ShinkaEvolve 的采样器。作者进一步给出增益的两步分解，Greedy/Island 到 Random 隔离池效应（抽 $K$ 条独立轨迹取最好），Random 到 BaSE 隔离分配效应（自适应把预算推向仍在改善的轨迹），前者的合法性由前述 run 间异质性提供。

一处实现细节值得记录：源码中存在一段被注释掉的非平稳奖励估计器，以最近一步增益外推下一步 $\tilde r_{i,n+1} = r_{i,n} + (r_{i,n} - r_{i,n-1})$，再向全局均值收缩；被注释掉的表标题区分 cumulative 与 predictive 两种模式。现行正文只保留了 bandit 骨架的引用，未纳入该估计器。

## 6. 结果

终局适应度上，BaSE 在 8 个 (model, task) 格子上取得最好或接近最好的均值（$C=512$，`tab:fitness_scores`）。收益随任务余量变化：Qwen3-8B HT 上 Thompson 达 $0.8736$，对照 greedy $0.6780$ 与 ShinkaEvolve $0.7379$；Llama HT 上 $0.4387$，对照最好的非 BaSE 结果 $0.2538$；Qwen3-14B CP 上 $1.0003$。在已接近上限的 CP 上（8B greedy $0.9985$、ShinkaEvolve $0.9986$），改进空间本身有限。摘要将整体效果概括为较最强 island 基线提升 12.3%，并强调其性质：

> this is not a model improvement, nor a prompt improvement, but an allocation improvement.（`main.tex`，Contributions）

达阈效率是更有说服力的一组证据，因为它衡量的是有限预算下的可靠性而非偶然取得的更好结果。以 90% 样本达到阈值 $\tau$ 所需代数计（Qwen3-8B，`tab:threshold_iteration_flops`）：MMD 上 $\tau=0.95$ 时 UCB 需 92 代而 greedy 需 485 代，三个 island 基线均未达到；HT 上 $\tau=0.70$ 只有 BaSE 达到（Thompson 101 代）；CP 上 $\tau=0.999$ 只有 Thompson 达到。相对随机分配，Thompson 在七个可达格子上平均少用约 40% 的代数。这组结果与论文开篇对只报最好一次运行的批评直接呼应。

臂池规模的消融显示中等池最优（$K \in \{2,5,10,20,50\}$，`tab:bandit_ablation_arms`）。这本身是跨轨迹层的深广权衡：$K$ 增大提高轨迹多样性，同时减少每条轨迹可得的细化步数。Qwen3-8B MMD 在 $K=2$ 时降至 $0.7172$，在 $K=5$ 至 20 达到 $0.9603$，$K=50$ 回落；HT 则在 $K=2$ 至 5 最好；Llama HT 在所有 $K$ 与所有算法下均为 $0.0000$。

与父代采样协议的组合实验（`tab:pairwise_fitness`）在多数格子上改善原协议，且改善集中于余量大或轨迹方差大的设置：Qwen3-8B HT 在 CodeEvolve 提示下由 $0.5168$ 升至 $0.7164$，Llama MMD 由 $0.2315$ 升至 $0.4906$，Qwen3-14B CP 由 $0.8768$ 升至 $0.9686$。作者对适用条件的结论是：

> These results suggest that BaSE is most useful when the candidate runs contain heterogeneous trajectory quality that can be exploited by adaptive allocation.（`bandit_appendix.tex`，`sec:pairwise_fitness`）

## 7. 主张与证据

| 机制主张 | 论文证据 | 证据等级 | 判断 |
| --- | --- | --- | --- |
| 固定预算下深广分配影响适应度 | $V_{\max}$ 与 best-of-N 的包络间距；`fig:surface_all8` 的适应度面 | 直接支持 | 在所测任务、模型与预算范围内成立；同一预算下 FLOPs 随 $T$ 变化不超过 15%，差异主要来自分配而非隐藏成本 |
| 双线性形式刻画分配几何，$c$ 区分角点与内部脊 | `tab:production_law`、`tab:production_law_app`；高阶项 F 检验 | 部分支持 | MMD 的 $c<0$ 在各设定内稳定（$\pm 0.02$）；CP 与 HT 的 $c$ 与 0 不可区分且跨设定变号；平台宽度是双线性假设的代数推论 |
| 任务几何源于算法族提议质量不对称 | `app:case_study`：CP 的 scipy 门控、MMD 的解析吸引子、HT 的可行性硬边界；两族均值差 $+0.35$ 与 $-0.14$ | 部分支持 | 由代码级个案与两侧均值支持，非受控消融；三任务、单模型 |
| 分配收益受能力闸门限制 | `tab:effect_model_size` 置换检验；Llama 全部格子；`fig:model_family_split` | 直接支持 | 阈值按 (model, task) 划分；双线性拟合失效更接近模型族性质而非纯能力下界 |
| 能力排序在有效 FLOPs 轴上塌缩 | `fig:flops_envelope`，MMD $R^2=0.94$、CP $R^2=0.93$ | 部分支持 | 限于未触顶区间、Qwen3 族内与三个几何任务 |
| BaSE 整体优于 \*Evolve 基线 | `tab:fitness_scores`、`tab:threshold_iteration_flops` | 直接支持（整法有效性） | 匹配协议下的整法比较直接支持 BaSE 方案在适应度得分与达阈 FLOPs 上整体优于 \*Evolve 基线；不能顺带推出臂池、bandit 策略与 $(T,N)$ 分配各自独立有效（组件效应由后续消融分解）。 |
| 池效应与分配效应可分解 | Greedy/Island 到 Random 再到 BaSE 的两步差 | 部分支持 | 分解在达阈表上可读；终局适应度主表未逐格给出 Random 列 |
| 臂池规模存在中等最优 | `tab:bandit_ablation_arms` | 直接支持 | 对目标变量 $K$ 的消融；$K=50$ 超过每格独立种子数，见第 9 节 |
| 与父代采样协议正交可组合 | `tab:pairwise_fitness` | 反向或混合证据 | 多数格子改善，同时存在整格退化的组合 |

## 8. 机制的底层逻辑

论文各部分证据可以串成一条因果链：单步 LLM 变异的输出分布很宽（第 5 节）；一条 greedy 轨迹在最初若干代锚定到某个算法族，而族本身设定天花板（第 4 节）；两者叠加使同配置重复产生高方差的终局分布；跨轨迹分配的价值随该方差增长。这解释了收益为何集中在 HT 与 Llama 等高方差格子，而在近饱和的 CP 上接近于零。该判断与本仓库的科学主张一致，即分配的杠杆随分支间生成质量的方差增长，且本文提供的是外部独立测量的支持。

论文观察到的停滞有一部分来自 greedy 协议本身。单锚点加严格改进的接受规则意味着，第一个被接受的局部最优成为永久父代，后续所有子代必须一次性超过该具体解。保留非最优状态作为可再访问锚点的搜索结构本可缓解部分锁死，而 BaSE 选择的路径是并行开多条轨迹并路由预算。两者并不冲突，但若把 BaSE 的全部增益读作跨轨迹分配的普遍价值，会高估其中属于协议结构修补的部分。

方法层面，本文使用的是现成的 bandit 算法，其价值在于把跨轨迹的计算资源分配提炼为一个独立问题，并在模型、提示、evaluator 全部固定后测量其效应。与此相应，它对轨迹的使用停留在把轨迹当作投资对象：仅依据历史表现判断是否继续投入，不建模轨迹的内容，也不用轨迹改变生成条件。用论文自己的定位说，这是 trajectory-aware compute allocation 这一层，尚未进入 trajectory modeling。

## 9. 实验条件与分析注意事项

**BaSE 的数值很可能来自离线重放。** 现行正文未写明运行方式，而源码中多处被注释掉的表标题写明 BaSE 使用 10 条冻结的 greedy 轨迹，点估计为在该池上运行 10 次 MAB 的均值，标准误来自 1000 次重采样该池并各运行一次 MAB。臂池消融取到 $K=50$ 而每格仅有 10 个独立种子，同样只能由有放回重采样解释。对 greedy 这类给定种子后轨迹序列固定的协议，重放是合理近似；但这意味着每次拉臂取得的是已录制的续段而非新抽样，池内最好轨迹构成 BaSE 的性能上界。

**随机性层级不同。** 基线使用 seed trajectory bootstrap，BaSE 为固定池上的 MAB 重复加 bootstrap，两者不是同一种重复。

**存在选择放大。** 被注释掉的表标题说明 BaSE 每格的 $(T,N)$ 由三种 bandit 的最大点估计选出，greedy 的 $T$ 亦按均值选出；主表再按格子加粗最优算法。

**摘要的 12.3% 无法从主表复原。** 正文未给出聚合规则。按 `tab:fitness_scores` 逐格计算，BaSE 最好值相对该格最强 island 基线的相对增益从 $+0.2\%$ 到 $+190\%$（Llama HT 由 $0.1512$ 至 $0.4387$），八格平均远高于 12.3%。该数字对聚合方式高度敏感。

**主文与附录对 HT 的归类不一致。** `tab:production_law` 记 HT-8B $c=-0.012$、$R^2=0.853$ 并将 HT 归入角点区，`tab:production_law_app` 记 $c=-0.034$（$p=0.002$）、$R^2=0.75$，其标题称 MMD 与 HT 都是显著负交互即内部脊。达阈叙述亦有类似出入：正文称 UCB 在 HT 上 60 代内达到 $\tau=0.70$，而表中 60 代对应的是 $\tau=0.50$。

**BaSE 退化的条件有明确记录。** 当所有轨迹困于同一个坏吸引子时，自适应分配会把预算集中到当前改善最快的一条，即使没有一条健康：Llama HT 在 ShinkaEvolve 提示下由 $0.1512$ 退至 $0.0274$，三种 bandit 一致；同一格在 greedy $N=1$ 下由 $0.0192$ 退至 $0.0000$。原协议的深广分配已经较好时同样可能退化：Qwen3-8B 在 greedy $N=4$ 下，MMD 由 $0.9407$ 退至 $0.8930$、$0.9119$、$0.8634$，HT 由 $0.5502$ 退至 $0.5238$、$0.4633$、$0.3276$。

**口径不可直接迁移。** 本文的预算单位是 LLM 调用而非真实评价，evaluator 被视为廉价且确定；任务为三个几何构造问题而非组合优化启发式；正式比较使用 10 个种子与三个模型。

## 10. 对 TraceAAD 的意义

BaSE 的臂对应 TraceAAD 的路线层，规模亦相近（BaSE 常用 $K=10$，TraceAAD 固定 $K=8$）。两处结构差异值得记录：BaSE 的所有臂从同一 seed program 起步，多样性仅来自采样，而 TraceAAD 的每条路线独立生成互不相同的根；BaSE 在选中轨迹内部没有锚点结构，只能沿单一链条推进，而 TraceAAD 的锚点层允许回到路线内的非最优状态。

更关键的差异在轨迹的用途。BaSE 依据轨迹的历史分数决定是否继续投入，选中之后生成什么与轨迹内容无关；TraceAAD 把来时路直接写入提示，用它改变给定锚点时的候选分布。因此这篇工作与 TraceAAD 的历史机制处于不同层，可以叠加。同时它提示一处需要自查的地方：TraceAAD 的路线层目前同样只用历史最好分数 $q^{*}(r)$ 作为投资依据，在这一侧与 BaSE 同构，尚未使用轨迹语义。

| 可学习点 | 成立前提 | 主要风险 | 最小验证方式 |
| --- | --- | --- | --- |
| 分开评价路线级分配与路线内生成 | 每条路线有独立状态与调用账本 | 把调度收益归因为生成机制 | 固定生成器与历史协议，在同总预算下比较均分、随机路由与自适应路由的 best-at-budget |
| 用两步分解隔离池效应与分配效应 | 可构造"$K$ 条独立路线取最好"的中间对照 | 只报端到端差会把池效应计入分配 | 在 1000 eval 下加入随机选路线的对照臂，测量两段差 |
| 先测路线间方差，再决定分配投入 | 有跨重复的终局与过程记录 | 在低方差任务上投入调参 | 用四任务三重复估计路线间终局方差，检验方差大的任务是否正是路线层干预率高的任务 |
| 算法族锚定比树拓扑更接近路线的语义身份 | 可从代码识别算法族，例如是否调用某类求解器、是否为解析构造 | 族标签依赖人工或正则，易失真 | 在现有 lineage 上标注根程序与终局程序的族，检验早期族锚定是否预测终局质量 |
| Explore 意图应以换族为目标 | 生成意图可识别且族标签可测 | Explore 产出仍留在原族，等价于加噪 | 固定锚点，比较 Refine 与 Explore 的换族率及随后的严格改进率 |
| 调用、token 与有效 FLOPs 分列报告 | 记录 prompt、cached、output token | 只报评价次数会掩盖协议间的前缀缓存差异 | 沿用 $F = 2 P_{act}(p_{prompt} - p_{cached} + p_{out})$ 事后计价，与 1000 eval 分开呈现 |

两条与本仓库现有判断直接对话的线索：

- **路线层乐观项近似未激活**（研究认识 1.3）。BaSE 的证据表明跨轨迹路由可带来较大收益，但该收益出现在 run 间方差极大、且臂间差距来自锚定到不同算法簇的格子上。TraceAAD 的 8 条根仅要求代码互异，路线间未必存在簇级差异。顶路线集中因此至少有三种读法：异质且正确集中、同质因而分配无意义、有差异但当前分数看不见。先测量路线是否为有意义的投资单位，比继续调整乐观尺度更有信息量。当前判断见研究认识 2.1–2.5。
- **BaSE 不证明应当上 bandit。** 它迫使我们把深度—广度权衡读成“找正确的族 vs 在已找到的族内开发”，并把分配写成有限预算下兑现 transition kernel 的问题。生成与分配实验上拆开、理论上耦合；好的分配建立在提议异质性上。下一步是读出四任务的簇结构，不是把 UCB 搬进路线层。
- **延续价值的候选估计量**（研究认识 2.5 与开放问题 4）。被注释掉的 predictive mean 给出趋势型延续价值的最小形式，可在现有日志上离线检验。论文最终未在正文保留该估计器，这一取舍本身也是信息。只有在确认路线具有可利用异质性之后，才值得把这类量写入在线规则。
