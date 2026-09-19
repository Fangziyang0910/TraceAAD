# SMCEvolve：用序贯蒙特卡洛统一程序进化

- 论文：Jiachen Jiang、Huminhao Zhu、Zhihui Zhu，*SMCEvolve: Principled Scientific Discovery via Sequential Monte Carlo Evolution*，arXiv:2605.15308v1，2026-05-14，预印本。
- 本地来源：paper.pdf。
- 原始实现：[kongwanbianjinyu/SMCEvolve](https://github.com/kongwanbianjinyu/SMCEvolve)。
- 设计对象：带外部 evaluator 的 LLM 程序进化，覆盖数学构造、数值程序加速、符号回归和端到端 ML 研究程序。

## 1. 核心判断

SMCEvolve 最有价值的地方，是把程序进化中的三个经验选择——选哪个父代、怎样产生并接受子代、何时结束——放进同一个概率目标中。它先将高质量程序搜索写成对 reward-tilted 分布的采样，再用序贯蒙特卡洛（Sequential Monte Carlo，SMC）的重加权、重采样和变异去近似该分布。由此，选择压力随退火温度共同作用于父代重采样、劣化子代接受和停止过程。

论文的主实验主要证明低预算下的早期搜索效率：SMCEvolve 在 128–480 次调用内经常达到较高 best-of-3 结果。相对于本仓库正式比较使用的 1000 次真实评价预算，这些实验测量的是 short-horizon evolution，接近“初始加速度”，尚未检验长视界下持续发现新算法簇、摆脱停滞或维持改进的能力。基线固定预算而 SMCEvolve 自适应停止的设计支持 sample efficiency 与自动停止主张，但不构成严格的 matched-budget 最终质量比较。

这项工作的理论贡献需要按理想算法理解。定理保证的是：在相邻桥分布足够接近、变异核精确保持目标分布且一致遍历等假设下，终局粒子经验分布可以逼近目标分布。它不直接保证找到全局最优程序。真实实现又因为黑盒 LLM 无法给出 proposal probability，省略了完整 MH 比率中的先验项与正反 proposal 比率，因此论文的实际系统是受 SMC 启发的近似实现，尚不是拥有该定理保证的精确 SMC sampler。

对 TraceAAD 而言，SMCEvolve 的首要价值是一套清晰的机制分解：LLM kernel 决定提议分布，接受机制决定哪些提议成为可延续状态，重采样决定哪些状态获得后续预算，archive 决定哪些未入选经验仍可影响未来生成。直接移植整套 SMC 的优先级较低。它也提供一个重要反例：ESS 只测量 reward 权重退化，无法区分“全体粒子已经高质量收敛”和“全体粒子以相近低分停滞”，因此不能单独承担质量收敛或算法簇覆盖的判断。

## 2. 论文要解决什么问题

论文研究的问题是：怎样为 LLM 驱动的程序进化建立一个统一、可分析的搜索原则，使父代选择、变异、种群管理和停止不再是彼此独立的经验模块。

作者指出现有系统有两类痛点。

第一，组件设计缺少共同目标。AlphaEvolve、ShinkaEvolve、ReEvo 等系统都维护程序种群、选父代、让 LLM 变异、评价子代并更新种群，但父代采样、上下文、变异方式和种群更新通常分别凭经验设计，再靠消融确认。即使整套系统有效，也很难说明各组件为什么应当这样组合。

第二，固定迭代数无法判断搜索状态。现有系统通常提前设定预算；预算太小可能尚未到达好区域，预算太大则继续消耗调用。论文进一步声称，缺少收敛控制会使系统难以区分高质量区域收敛和局部停滞。后文的 ESS 机制确实能自适应决定退火步长与终止时刻，但它只读取当前 reward 的相对离散程度，因此并未完全解决高质量收敛与同分低质停滞的区分问题。

论文的中心问题是：能否把 LLM 程序进化建立在严格的概率框架上，由同一原则导出各组件，并获得有限样本收敛保证。

## 3. 从优化问题到目标分布

### 3.1 Reward-tilted 分布

设程序为 $x$，任务为 $q$，基础 LLM 在只看到任务时诱导的程序先验为 $p_0(x\mid q)$，evaluator 给出有界标量奖励 $R(x)$。论文将目标写成以下带 KL 正则的分布优化：

$$
\max_p\; \mathbb{E}_{x\sim p}[R(x)]-\frac{1}{\beta}D_{\mathrm{KL}}(p\Vert p_0).
$$

其唯一最优解为

$$
p^*(x\mid q)=\frac{1}{Z(q)}p_0(x\mid q)\exp(\beta R(x)).
$$

$\beta$ 控制 reward 与 LLM 先验的权衡：$\beta$ 小时接近基础 LLM 的自然程序分布，$\beta$ 大时质量差异被指数放大。这个形式化的意义在于，搜索对象成为“由 LLM 先验约束、向高 reward 倾斜的程序分布”。因此理论最终约束粒子群对 $p^*$ 的近似；best-so-far 最优性属于实验指标。

### 3.2 用桥分布逐步到达目标

直接从 $p^*$ 采样需要不可得的配分函数。SMCEvolve 构造一系列桥分布：

$$
p_t(x)\propto p_0(x\mid q)\exp(\beta_t R(x)),\qquad \beta_t=\lambda_t\beta,
$$

其中 $0=\lambda_0<\lambda_1<\cdots<\lambda_T=1$。粒子群从 $p_0$ 出发，逐步提高 reward intensity，最终目标是 $p^*$。每个阶段执行三件事：

1. 根据本阶段的增量温度重加权并重采样父代；
2. 对每个重采样父代运行 $K$ 步 LLM proposal 与接受链；
3. 根据 ESS 决定下一步 $\lambda_t$，当 $\lambda_t=1$ 时结束。

若有 $I$ 个 island、每个 island 有 $N$ 个粒子，则一次完整阶段最多产生 $I\times N\times K$ 个 LLM proposal。论文实验配置为 $I=2$、$N=8$、$K=2$。

## 4. 三个核心机制

### 4.1 自适应父代重采样

采用时间反演 backward kernel 后，SMC importance weight 化简为只依赖父代 reward 的形式：

$$
w_t(x)=\exp((\beta_t-\beta_{t-1})R(x)).
$$

归一化后，父代采样就是以 $\Delta\beta_t=\beta_t-\beta_{t-1}$ 为逆温度的 reward softmax。$\Delta\beta_t$ 小时，权重接近均匀；$\Delta\beta_t$ 大时，高分粒子获得更大的复制概率。实现使用 systematic resampling，以较低的重采样方差复制高权重粒子并删除未被抽中的粒子。

这一机制调节的是当前粒子的相对选择压力。它没有估计一条路线未来还能改善多少，也不把粒子的形成轨迹作为信用变量。高分父代被复制后，谱系可能迅速合并；是否重新扩散依赖后续 mutation kernel。

### 4.2 四种 LLM proposal kernel

论文把生成机制沿两个轴拆成 $2\times2$ 网格：

| | 无 inspiration | 有 inspiration |
| --- | --- | --- |
| 局部 diff | 对当前程序做精确局部编辑 | 参考其他程序后局部移植或改写 |
| 全量 rewrite | 从当前问题重新设计完整程序 | 综合多个参考程序进行重组 |

inspiration 由两部分组成：当前 island 的 top-$k$ 高分程序，以及从完整历史 archive 中选出的 $m$ 个、与父代文本 embedding 距离最大的程序。论文配置为 top-2 加 diverse-2。archive 保存所有被 evaluator 评价过的程序，包括随后被 MH 拒绝的 proposal。于是“是否成为搜索状态”和“是否成为未来生成信息”被明确分开：被拒绝的候选不能延续，但仍可能以 inspiration 的方式影响后续 proposal。

四个 kernel 由 Thompson sampling 自适应选择。当前实现为每个 kernel 维护 Beta 后验，以 proposal 是否严格改善当前链状态作为即时二元反馈；每次反馈前把所有臂的参数乘 $0.99$ 并下限截到 $1$，使后验偏重近期表现。这是一个固定 4 臂、即时且稠密反馈的 operator 选择问题。

### 4.3 Reward-only MH 接受

精确 MH 接受概率应包含目标密度与正反 proposal probability：

$$
\alpha_t(x,x')=\min\left\{1,
\frac{p_t(x')Q_t(x\mid x',C_t)}{p_t(x)Q_t(x'\mid x,C_t)}
\right\}.
$$

黑盒 LLM API 无法给出 $p_0$ 和 $Q_t$ 的概率比，论文将结构不对称项近似为 $1$，实际采用

$$
\alpha_t(x,x')=\min\{1,\exp(\beta_t(R(x')-R(x)))\}.
$$

改善 proposal 总是接受；退步 proposal 以随 reward gap 和 $\beta_t$ 指数下降的概率接受。早期 $\beta_t$ 小，链可以经过较差中间状态；后期 $\beta_t$ 大，链趋向保守。每个粒子连续运行 $K$ 步，已接受的状态成为下一步 proposal 的父代，拒绝则保留当前状态。

这个机制有两个有意思的含义。其一，非单调过渡构成跨越局部谷底的显式通道。其二，算法把“生成出一个候选”“接受它作为下一状态”“把它保存为未来信息”分成三个不同事件。

### 4.4 ESS 温度调度与自动停止

对候选下一温度 $\lambda$，论文计算

$$
\operatorname{ESS}(\lambda)=
\frac{\left(\sum_n u^{(n)}(\lambda)\right)^2}
{\sum_n\left(u^{(n)}(\lambda)\right)^2},
\qquad
u^{(n)}(\lambda)=\exp((\lambda-\lambda_{t-1})\beta R(x_{t-1}^{(n)})).
$$

通过二分搜索取满足 $\operatorname{ESS}(\lambda_t)\geq\kappa N$ 的最大 $\lambda_t$；若直接走到 $1$ 仍满足阈值，就令 $\lambda_t=1$ 并结束。论文使用 $\kappa=0.9$，并用 `min_iterations=3` 将单次 $\Delta\lambda$ 上限设为 $1/3$，避免 reward 过于接近时一步结束；同时保留 `max_iterations=15` 的硬上限。

ESS 在这里测量的是一次 reward 重加权后还有多少“有效权重粒子”。reward 差异大时，小幅提高 $\lambda$ 就会使权重集中，因此步长变小；reward 差异小时可以走更大步。它不直接测量代码差异、算法簇覆盖、proposal 新颖性或绝对质量。论文把“粒子仍然分散”写成较宽泛的直觉时，严格含义应限定为 reward weight dispersion。

### 4.5 Island 与迁移

完整实现运行两个相互独立的 SMC island，每三轮把各 island 最好的一个粒子发送给随机相邻 island；接收方把本地粒子与 migrant 合并后按 reward 保留 top-$N$。这一机制为并行搜索保留了短期隔离，同时允许高质量程序跨群体传播。它是额外工程机制，论文 Table 5 将 island 与迁移标为理论之外的近似扩展。

## 5. 论文如何理解三个机制的耦合

论文最有意思的整体观点，是用同一个 $\beta_t$ 协调三个阶段：

- 早期 $\beta_t$ 小，父代重采样接近均匀，较差 proposal 也有较高接受概率，搜索更容易扩散；
- 后期 $\beta_t$ 大，父代权重更偏向高分，劣化 proposal 更难被接受，搜索更集中于已发现的高质量区域；
- ESS 根据当前 reward 分布决定 $\beta_t$ 增长多快，并以到达预设终点 $\beta$ 作为停止条件。

这种耦合比独立设置固定父代温度、固定接受规则和固定迭代数更有原则。需要保留两点限定。第一，Thompson sampling 选择哪个 proposal kernel 并不由 $\beta_t$ 直接控制，它只通过近期严格改善反馈与搜索阶段间接耦合。第二，到达 $\lambda=1$ 表示退火路径到达设计的目标温度，不等于已经找到高质量程序，也不等于搜索不再可能改善。

## 6. 理论结果：证明了什么

### 6.1 定理内容

论文从已有有限样本 SMC 理论中导入结果。若 reward 在 $p_0$ 支撑集上的振幅 $\Delta R$ 有限，相邻桥满足 $\lVert p_t/p_{t-1}\rVert_{L_2(p_{t-1})}^2\leq\kappa^{-1}$，每阶段 mutation kernel 精确保持 $p_t$ 且以最坏速率 $\rho<1$ 一致遍历，则对任意固定有界统计量 $f$ 与误差 $\epsilon>0$，存在 $N,T,K$ 使

$$
\Pr\left(\left|\eta_T^N(f)-p^*(f)\right|\leq\epsilon\right)\geq\frac{3}{4},
$$

总 LLM kernel 调用预算满足

$$
B=NTK=\widetilde O\left(
\frac{\epsilon^{-2}\vee\kappa^{-1}}{1-\rho}\,\beta\Delta R
\right).
$$

这里 $\eta_T^N$ 是终局粒子的经验分布。概率 $3/4$ 可用多次独立运行和中位数放大，但这会增加调用成本。

### 6.2 三个理论项的含义

| 理论项 | 论文中的操作对应 | 实际状态 |
| --- | --- | --- |
| 相邻桥分布接近 | ESS 二分选择短温度步 | ESS 是有限粒子、数据依赖的代理；正文把桥比率界保留为显式假设 |
| mutation mixing | $K$ 步 MH 链与四 kernel mixture | 需要精确 $p_t$-invariance 和 uniform ergodicity；真实 LLM kernel 未验证 |
| 路径长度 | $\log\Gamma\leq\beta\Delta R$ | 在 reward 有界且目标分布定义成立时可推导 |
| 目标与 importance weight | $p^*\propto p_0e^{\beta R}$ 与 $e^{\Delta\beta R}$ | 形式上精确 |
| 实际 forward kernel | reward-only MH、Thompson、自适应上下文 | 近似；不继承精确 MH 保证 |
| islands 与迁移 | 并行岛及 top 粒子迁移 | 理论之外的扩展 |

### 6.3 不能从定理推出的结论

1. 定理约束固定统计量在目标分布下的期望误差，不保证 best-so-far 接近全局最优。
2. 定理是“存在足够大的 $N,T,K$”的复杂度结论，不证明论文默认的 $N=8,K=2$ 已进入保证区间；附录明确说 $N$、$K$ 各自都必须足够大。
3. 混合速率 $\rho$ 在真实程序空间中未知，uniform ergodicity 没有实证验证，因此该界不能直接变成可执行预算。
4. 真实 reward-only MH 省略 $p_0$ 与 proposal ratio 后不再保证 $p_t$-invariance；高接受率只说明 proposal 常被接受，不能验证其接近精确 MH。
5. ESS 规则面向桥分布稳定性，但论文没有证明有限粒子、数据依赖的 ESS 调度必然满足定理要求的确定性桥比率界。

因此，“有收敛保证”应准确写成：论文为理想化的 SMCEvolve kernel 给出有限样本 SMC 分布近似界，并用实际机制近似这些条件。保证的适用范围不覆盖当前黑盒 LLM 实现或全局程序最优性。

## 7. 实验设置与结果

### 7.1 主实验

除 AutoResearch 外，各方法共享 `gpt-5-mini + gemini-3-flash` 的 50/50 模型组合；AutoResearch 使用 `gpt-5.4 + gemini-3-pro`。比较方法为 ReEvo、OpenEvolve 与 ShinkaEvolve。每个单元格报告 3 个 seed 中的最好分数。基线使用固定调用上限：数学 200、AlgoTune 500、符号回归 400、AutoResearch 200；SMCEvolve 按 ESS 自动结束。

| 领域 | 任务数 | SMCEvolve 取得表中最好值 | SMCEvolve 调用数 | 固定基线预算 |
| --- | ---: | ---: | ---: | ---: |
| 数学构造 | 10 | 9/10 | 128–192 | 200 |
| AlgoTune 程序加速 | 8 | 7/8 | 288–480 | 500 |
| 符号回归 | 16 | 13/16 | 128–384 | 400 |
| AutoResearch | 1 | 终局 reward 最高 | 图中约 190 次自动停止 | 200 |

按 Tables 1–3 逐项计数，SMCEvolve 在 34 个表格任务中有 29 个最好。例外也值得保留：Heilbronn Triangle、`polynomial_real`、CRK3、PO0 和 PO2 分别由其他方法更好，其中 `polynomial_real` 上 ShinkaEvolve 的 speedup 为 33.8776，明显高于 SMCEvolve 的 2.4000。论文主结果支持的是完整配方在这些 benchmark 和模型设置下具有较强的 best-of-3 表现及较低调用数，不支持每个内部组件都单独带来跨域收益。

### 7.2 预算公平性与搜索视界

主实验的预算协议是“固定预算基线对自适应停止方法”：ReEvo、OpenEvolve 和 ShinkaEvolve 分别跑到各领域的固定调用上限；SMCEvolve 固定每轮的粒子数与链深度，由 ESS 决定轮数 $T$，到达 $\lambda=1$ 时停止，同时以 `max_iterations=15` 截断。论文配置下两个 island 每轮最多使用 $2\times8\times2=32$ 次 proposal。因此这里的 budget control 主要自适应搜索长度，不在线改变每轮的宽度 $N$ 或深度 $K$。

这种比较能回答：一套方法能否自行停止，并以少于基线上限的调用获得较高结果。它不能直接回答：所有方法使用完全相同调用数时谁的终局质量更高。严格的 matched-budget 对照需要让 SMCEvolve 也运行到相同调用数；完整效率比较还应同时报告 best reward—calls 曲线、达到固定质量阈值的调用数，以及相同长视界预算下的最终结果。

更重要的边界是搜索视界。数学任务只用 128–192 次调用，AlgoTune 最多 480 次，整体硬上限也只有 15 个 SMC stage。这些运行已经包含多轮重采样、连续变异、接受与 kernel 更新，因而不能称为“尚未发生进化”；但它们主要比较谁能在前几百次 proposal 内更快进入高 reward 区域。强局部改进和 exploitation 机制在这种协议下更容易显效，依赖长期轨迹积累、跨算法簇迁移或后期重新扩散的机制可能尚未进入发挥区间。现有结果因此支持 short-horizon sample efficiency，不支持 SMCEvolve 在 1000 次及以上长视界 AAD 搜索中仍保持优势。

### 7.3 Circle Packing 消融

Table 4 只在 Circle Packing $N=21$ 上做机制消融，每项报告 3 个 seed 的最好 reward。论文称各变体保持 LLM 调用预算不变；宽度—深度两端进一步固定每轮 $N\times K=16$。

| 设置 | Best reward |
| --- | ---: |
| 默认：$N=8,K=2$，自适应四-kernel mixture | **0.9993** |
| uniform 父代重采样 | 0.9514 |
| greedy 父代重采样 | 0.9760 |
| 仅 diff、无 inspiration | 0.9379 |
| 仅 diff、有 inspiration | 0.9868 |
| 仅 rewrite、无 inspiration | 0.9379 |
| 仅 rewrite、有 inspiration | 0.9514 |
| uniform 四-kernel mixture | 0.9929 |
| $N=4,K=4$ | 0.9379 |
| $N=16,K=1$ | 0.9379 |

这组结果直接支持三条局部结论：在该任务上，自适应重采样优于 uniform 和 greedy 两端；多 kernel 优于任一单 kernel；Thompson 选择略高于均匀四-kernel mixture；固定每轮调用数时，中等粒子宽度和两步链深度优于两个极端。

它没有单独消融 reward-only MH 接受，也没有在匹配调用数下比较 ESS 自动停止与固定停止。因此接受规则和停止规则改善最终质量的独立效应仍未识别。

## 8. 主张与证据审计

| 论文主张 | 主要证据 | 证据强度 | 判断 |
| --- | --- | --- | --- |
| SMC 可统一父代选择、变异与停止 | §2、Algorithm 1、Appendices B–D | 形式化支持 | 对理想 SMC 构造成立；实际 kernel 有近似项 |
| 完整系统跨域且调用更少 | Tables 1–3、Figure 4 | 部分支持（口径受限） | 整系统比较直接支持在所测 29/34 表格任务及 AutoResearch 上取得较少调用与较好峰值，但口径为 best-of-3、缺少均值/方差与统计检验；整法胜出亦不能顺带推出每个 SMC 组件分别有效。 |
| 低预算下具有较高早期搜索效率 | Tables 1–3、Figure 4 | 间接支持 | 自适应停止时调用少于基线上限且多数任务结果更好；完整系统与停止规则共同作用 |
| 长视界进化能力优于现有方法 | 无对应实验 | 不支持 | 最大预算 500 次调用，未展示 1000 次及以上搜索中的持续改进、算法簇发现或停滞恢复 |
| 自适应重采样优于固定两端 | Table 4 | 直接但窄 | 单任务、3 seeds 的最好值支持，不足以推出跨任务普遍性 |
| kernel diversity 有益 | Table 4 | 直接但窄 | 四-kernel mixture 高于四个单 kernel |
| Thompson 自适应选择有益 | Table 4 | 直接但弱 | 0.9993 对 uniform mixture 0.9929；单任务、best-of-3 |
| reward-only MH 接受提高搜索 | Figure 3 接受率与完整系统 | 未单独识别 | 没有 unconditional/greedy acceptance 对照；高接受率不等于质量收益或理论校准 |
| ESS 自动停止兼顾质量与成本 | 主表调用数、Figure 4 | 间接支持 | 完整系统以较少调用获得较好 best-of-3；缺少同系统固定停止与质量—成本前沿对照 |
| 实际系统具有有限样本保证 | Theorem 3.1、Appendices E–G、Table 5 | 部分支持 | 定理要求精确不变核、桥正则与一致遍历；真实 forward kernel、TS、islands 均为近似或扩展 |
| 平衡宽度与深度普遍更好 | Table 4 | 探索性 | 只支持 Circle Packing 的 $N\times K=16$ 设置 |

## 9. 有意思的结论与观点

### 9.1 程序进化可以被理解成近似推断

将进化写成对 $p_0(x\mid q)e^{\beta R(x)}$ 的近似采样，比“维护一个种群并不断变异”多给出了一层解释：父代重采样、mutation mixing 和停止都服务于同一目标分布。这个视角使组件间的相互作用可以被分析，也迫使方法说明自己最终希望保持什么分布，而不仅是怎样保留最高分个体。

### 9.2 Proposal、接受和预算分配是三种不同能力

同一个 LLM proposal kernel，在不同接受率和重采样权重下，会形成不同的实际搜索过程。评估一个生成算子时，仅看被保留子代会把生成能力和选择机制混在一起。更完整的过程证据应分别报告 proposal 有效率、即时改善率、接受率、成为后续父代的概率、被重采样复制的概率以及单位评价预算收益。

### 9.3 非单调状态可以有两种命运

较差 proposal 可以被温度化接受，作为短链中的下一状态；即使被拒绝，它也进入 archive，成为以后生成的信息。论文因此同时保留“状态连续性”和“失败信息价值”，比把所有退步候选立即删除更细致。它仍然只用即时 reward gap 决定状态接受，没有估计一个退步状态的有限续段价值。

### 9.4 宽度与深度需要在同一预算内共同考虑

$N$ 决定一次重采样保留多少并行粒子，$K$ 决定每个粒子有多少连续 mutation 机会。只增大 $N$ 会使单条链来不及混合，只增大 $K$ 会使重采样缺少足够的群体宽度。Circle Packing 的结果显示平衡点有价值，但它是任务相关的经验结果，不能解释成固定的 $N=8,K=2$ 普遍最优。

### 9.5 “自动停止”必须明确停止的对象

SMCEvolve 自动结束的是退火路径：$\lambda$ 已到 $1$。当所有粒子具有相近的低 reward 时，任意温度下的权重仍接近均匀，ESS 很高，算法会以允许的最大步长推进并在最少轮数后停止。这一情况在数学上满足“重加权不退化”，却可能对应语义同质、低质量或无改善的种群。因此 ESS 适合做权重退化与选择压力诊断，不能单独证明高质量收敛、非停滞或算法簇充分覆盖。

### 9.6 算子多样性与动态算子选择是两个问题

四种 kernel 同时存在，扩大了局部/全局、单父代/跨程序四类 proposal 的可达范围；Thompson sampling 则根据近期严格改善反馈改变四类 proposal 的实际调用比例。Figure 3 在 Circle Packing 上表明 inspiration kernel 具有更高接受率，随后被选择得更多，支持“机制运行且改变选择”。Table 4 中自适应 mixture 的 0.9993 高于 uniform mixture 的 0.9929，提供了“终局质量更高”的单任务证据，但差距较小且仍是 best-of-3。因而可以带走的研究问题是“何时使用哪个算子”，不能据此认定 Thompson sampling 跨任务普遍改善质量。

### 9.7 用四个维度理解现有进化框架

论文将 AlphaEvolve、ShinkaEvolve 等写成 SMCEvolve 的特例：固定选择压力、$K=1$、无条件接受和固定迭代数。这个视角的价值，是把程序进化放到 selection、proposal、acceptance、budget control 四个维度上比较，明确每种方法在哪一层改变了搜索行为。它是一种机制坐标系，不证明这些系统的全部行为都被 SMC 理论解释。

## 10. 实验条件与复现注意

1. **主表只报 best-of-3。** 论文以高方差和收敛控制为动机，却没有在主表报告三次运行的均值、标准差、置信区间或成功率。最好一次能说明能力可达性，不能说明稳定性。
2. **组件消融集中在单任务。** 所有核心消融只覆盖 Circle Packing $N=21$，且仍取 3 seeds 中最好值。跨域完整系统结果不能替代跨域机制识别。
3. **主比较没有匹配预算。** 基线跑到固定上限，SMCEvolve 自适应停止；现有结果适合解释为质量—调用效率，不是相同预算下的终局能力排序。
4. **搜索视界较短。** 主实验上限为 200–500 次调用，尚未检验 1000 次及以上预算中的持续演化行为。低预算协议可能偏向快速局部提升。
5. **停止机制没有独立消融。** 主实验同时改变父代选择、kernel mixture、接受规则、上下文、islands 和停止；较少调用与较高质量是整套配方的联合结果。
6. **理论假设没有在代码空间验证。** uniform ergodicity、混合速率 $\rho$、精确 $p_t$-invariance 和数据依赖 ESS 对桥比率的控制均未获得真实 LLM 程序空间证据。
7. **reward 尺度影响温度含义。** 固定 $\beta=20$ 在不同 reward 尺度上产生不同的目标倾斜强度。ESS 会调节路径步数，但不会消除目标分布本身对 reward 标定的依赖；硬 `max_iterations` 还可能在到达 $\lambda=1$ 前截断。
8. **代码与技术说明存在版本不一致。** 2026-08-19 核对的 `smcevolve/island.py` 实现 reward-only MH，并使已接受状态成为下一 proposal 的父代；同仓库 `docs/SMCEvolve_Technical_Document.md` 的若干段落仍写成旧的 best-of-K 逻辑，文件索引也仍指向 `_best_of_k()`。论文正文与当前核心代码在 MH 上一致，复现时应以实际代码和运行工件为准。
9. **论文配置与仓库默认值需要区分。** 论文 Table 10 使用 `max_iterations=15`，公开技术说明列出的当前默认值为 30。公开仓库仍在变化，复现论文数字应固定具体版本和配置。

## 11. 与 TraceAAD 的关系

### 11.1 两项工作的研究对象不同

| 维度 | SMCEvolve | TraceAAD |
| --- | --- | --- |
| 核心对象 | 粒子群如何逼近 reward-tilted 程序分布 | 改进来时路如何改变下一步生成与有限预算分配 |
| 生成条件 | 当前程序、任务、top/diverse inspiration | 当前算法、匹配的 parent improvement path、生成意图 |
| 主要分配单位 | 当前粒子及其 reward | route、hypothesis 或 anchor 及其历史证据 |
| operator 分配 | 4 个全局 kernel 的即时 Thompson 选择 | operator 与锚点条件下的生成及分配问题 |
| 负向状态处理 | 早期允许 reward-tempered 接受；全部评价程序进入 archive | 保留非单调形成事实，并测量短续段发展价值 |
| 停止口径 | ESS 推进到目标温度，另有硬上限 | 正式比较固定 1000 次真实评价预算 |
| 理论目标 | 终局粒子经验分布逼近 $p^*$ | 有限预算下改进轨迹的生成与投资决策 |

SMCEvolve 的历史主要以全局候选 archive 和 inspiration 进入提示；生成条件不含父代改进来时路，分配也不估计路线的 continuation value。它提供的是围绕当前粒子群的 proposal—acceptance—resampling 框架，与 TraceAAD 的轨迹条件生成研究对象可以并列比较。

### 11.2 对 TraceAAD 最有价值的五点

#### 价值一：把“生成了什么”和“最终投资了什么”彻底分开

SMCEvolve 的 SMC 分解与 TraceAAD 的两个核心对象高度兼容：proposal kernel 对应 $P(x_{t+1}\mid x_t,h_t,o_t)$，接受与重采样共同构成有限预算下实际兑现 proposal 的选择过程。TraceAAD 的过程诊断可以据此固定一条分解链：

$$
\text{proposed}\rightarrow\text{valid}\rightarrow\text{accepted}\rightarrow
\text{revisited}\rightarrow\text{produced future gain}.
$$

各阶段应分别报告，不用终局 held-out 反推某一个生成或分配组件有效。

#### 价值二：SMCEvolve 的 Thompson 成功不能外推到 V9.10 式联合分配

SMCEvolve 只有 4 个全局 kernel 臂，每个 proposal 立即得到“是否严格改善当前状态”的完整二元反馈，并以 $0.99$ 衰减。这是低维、稠密、即时反馈问题。TraceAAD V9.10 的联合臂随新锚点增长，反馈包含有限窗口等待与右删失，约半数锚点没有再次选择，终局仍有大量 pending 动作。两者虽然都叫 Thompson sampling，统计问题完全不同。

这篇论文反而强化了一个诊断：bandit 是否有效，首先取决于臂是否稳定、反馈是否及时、每个臂是否能积累代表性样本。不能因为四个全局 operator 上的 Thompson 有益，就认为对持续扩张的 anchor × intent 臂也应有益。

#### 价值三：ESS 适合作为分配形状诊断，不适合直接成为停止或调参目标

TraceAAD 可以借用 effective sample size 的思想，报告路线或锚点分配质量对应的有效投资单位数：

$$
N_{\mathrm{eff}}=\frac{1}{\sum_i w_i^2}.
$$

它能把“权重集中在少数对象”压缩成可比较数字，并与 unique lineage、实际选择频数和 proposal 成功率并列。它不能证明路线语义异质、选择正确或搜索已经收敛，也不应反向调参数去追求某个 ESS。TraceAAD 的正式比较仍应保持 1000 次真实 evaluator 预算；若研究自适应停止，应作为单独的质量—成本实验。

#### 价值四：将状态池与经验池分离

SMCEvolve 让被拒绝 proposal 离开粒子状态，却保留在 archive 中供未来 inspiration。TraceAAD 已经保存 direct attempts 和失败事实，但默认 parent-path prompt 不常驻这些信息。论文提供了一个可检验方向：只在 Explore 的固定锚点实验中，对比 parent path、top+diverse rejected archive、二者组合，测量有效率、静态机制代理切换、即时质量和强制短续段恢复。代码 embedding 距离不能直接代表算法 family，因此只能作为候选检索器，不能当作多样性结论。

#### 价值五：把生成意图拆成“改动尺度 × 信息来源”

TraceAAD 的 Refine/Explore 主要表达族内发展与替代方向提议；SMCEvolve 进一步把 edit scope（diff/rewrite）和 information source（single/inspiration）正交化。这个拆分有助于回答：Explore 的效果来自更大改动，还是来自看到了其他程序；Refine 的稳定性来自局部 patch，还是来自单父代锚定。论文只证明四-kernel mixture 整体优于单 kernel，没有识别两个轴各自的因果贡献，因此应先做固定锚点的 $2\times2$ 生成实验，再决定是否改变在线机制。

### 11.3 一个值得谨慎检验的机制：温度化跨谷接受

TraceAAD 已有证据表明即时退步状态可能在有限续段内恢复。SMCEvolve 给出一种最小在线实现：按搜索阶段和 reward gap 决定是否让退步 child 成为下一状态。它的优点是把早期探索和后期保守统一起来；风险是即时 reward 标度主导接受率，且未估计状态特定的 continuation value。

最小验证不应直接重写正式搜索。可以固定 Explore child 和后续 proposal seed，在相同评价预算下比较严格拒绝、固定两步强制续段和温度化接受，观察 parent recovery、best gain、无效链比例与不同任务的方向。只有该受控实验显示温度化规则比固定短续段更有效，才有理由进入在线版本。

## 12. 对 TraceAAD 的行动优先级

| 优先级 | 可吸收内容 | 最小验证 | 不能直接外推的部分 |
| --- | --- | --- | --- |
| 高 | proposal—acceptance—revisit—future gain 分层日志与报告 | 直接复用现有事件和 checkpoint 做离线漏斗统计 | 不能由访问集中推断生成更好 |
| 高 | 用 effective size 描述分配集中度 | 对 route、anchor 选择权重与实际频数并列计算 | ESS 不代表算法簇覆盖或高质量收敛 |
| 高 | 识别 4 臂即时 TS 与扩张联合臂延迟 TS 的差异 | 对 V9.10 报每臂反馈数、等待时间、pending 和后验有效样本 | SMCEvolve 的正结果不支持直接调 V9.10 后验参数 |
| 中 | rejected archive 作为 Explore 专用全局信息 | 固定锚点比较 parent path、archive、hybrid | embedding 远不等于语义新颖 |
| 中 | edit scope × information source 因子化 | $2\times2$ 固定锚点生成实验 | 单任务 mixture 消融不能证明哪个轴有效 |
| 中 | 温度化接受退步 child | 与严格拒绝、固定续段做匹配预算对照 | reward-only MH 不继承精确 SMC 保证 |
| 低 | ESS 自动停止 | 另做 matched-quality / matched-budget 效率实验 | 不改变当前 1000-eval 正式比较口径 |

## 13. 总体评价

SMCEvolve 是一篇理论视角强、机制组合有启发、实证边界需要谨慎读取的工作。它最成功的部分，是将父代重采样、LLM mutation、接受与终止放进 reward-tilted SMC 的同一语言中，并以 29/34 个表格任务最好和较少调用展示了完整系统在低预算下的早期搜索效率。它最薄弱的部分，是理论保证依赖真实实现未满足或未验证的条件，主实验只报 best-of-3，预算未严格匹配且搜索视界较短，接受与自动停止缺少独立消融，ESS 也不能完成论文动机中“高质量收敛与低质量停滞”的区分。当前证据可以支持“初始加速度较强”，尚不能支持“长时进化能力更强”。

对 TraceAAD 的直接结论是：继续坚持生成与分配实验上拆开、理论上耦合；将每个 proposal 的生成、接受、回访和后续价值分层观测；把有效样本量作为分配退化诊断；优先研究低维、反馈充分的 operator 选择和 rejected archive 的信息价值。SMCEvolve 没有证明应把 TraceAAD 改造成粒子滤波器，也没有证明 reward-only 重采样能识别有意义的路线。它提供的是一套更严格地提问和设计对照实验的语言。
