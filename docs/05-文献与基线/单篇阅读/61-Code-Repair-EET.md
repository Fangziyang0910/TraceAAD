# Code Repair EET（REx）

- 论文：*Code Repair with LLMs Gives an Exploration-Exploitation Tradeoff*；本地来源：`main.tex`；研究对象：LLM 迭代修复代码的策略选择理论（arm-acquiring bandit），而非一套 AAD 系统。

## 1. 核心问题与方法

给定可分解为合取约束 $\Phi=\phi_1\wedge\cdots\wedge\phi_K$ 的规范（测试用例、验证条件），refinement 分布定义为以随机反例为条件的提示生成 $P_{refine}(\rho'\mid\rho,\Phi)=\mathbb{E}_{\phi\sim U(CE)}[P_{LLM}(\rho'\mid prompt(\rho,\phi,\Phi))]$，启发式进度 $h(\rho)$ 为已满足约束比例。论文的核心论证（建模推导，非编号定理）：每轮必须选择**精炼哪个程序**——精炼通过测试最多的程序即利用，精炼被考虑较少的程序即探索；每次 refine 产生全新程序、动作集随时间增长，构成 **arm-acquiring bandit**（arm=程序，pull=refine 一次，奖励 $r\sim Bernoulli(\theta_\rho)$，$\theta_\rho$ 为该程序被修复成功的概率）。标准 MCTS 不适用：分支因子无限、转移随机、rollout 需过长调用链。

## 2. 论文宣称的机制贡献（逐项）

- REx 选择规则：先验把启发式知识注入 $P(\theta_\rho)=Beta(1+C\cdot h(\rho),\,1+C\cdot(1-h(\rho)))$；被 refine $N_\rho$ 次均失败后后验为 $Beta(1+C\cdot h(\rho),\,1+C\cdot(1-h(\rho))+N_\rho)$，期望 $\mathbb{E}[\theta_\rho\mid N_\rho]=(1+C\cdot h)/(2+2C+N_\rho)$ 随失败次数衰减、方差按 $O(N_\rho^{-3})$ 收缩；每轮从各程序后验采样 $\hat\theta$、精炼采样值最大者；新程序作为新 arm 进入，空程序也是一个 arm（精炼它=从头生成）。$C=20$ 全域使用。
- 失败计数是比显式 $\varepsilon$-greedy 更有原则的探索机制：期望随失败下移，但每个程序始终保有非零被选概率（Thompson 的天然软探索）。
- Greedy/BFS/Fixed-Width 各锁定一个固定深宽比，最优深宽比随数据集变化；REx 自适应决定何时搜宽、何时沿有希望分支深挖。

## 3. 实验究竟支持了什么

|主张|论文证据|证据等级|判断|
|---|---|---|---|
|REx 跨域最鲁棒|APPS 竞赛/入门、ARC 40 题、38 个循环不变量；GPT-4 temp=1|直接支持|全部领域解题数最多，但最终数只适度领先；主要优势在成本（APPS-Comp 约 2×、不变量 2–5×）、超参鲁棒性与跨数据集稳定性；容易题（APPS-Intro）无加速。|
|固定深宽策略无跨任务最优解|Greedy 在最易档第一、难题差；Fixed-Width 在 ARC/不变量接近 REx、在 APPS-Intro 最差；各策略最优超参逐数据集漂移|直接支持|REx 树形态"整体宽、有希望分支上深精炼"；自适应的收益更多体现在鲁棒性而非峰值。|
|失败计数后验是有效的探索信号|Beta 后验公式与过程行为|部分支持|数学结构清晰但无 regret 界等正式保证；任务目标是离散"通过测试"，与连续 AHD fitness 之间隔一层。|

循环不变量上 REx 28/38（73.7%）对专用 SOTA G-CLN 24/38。作者自述局限：大计算极限下 REx 只"适度多解题"，攻克最难题更依赖基础模型改进。

## 4. 机制的底层逻辑（阅读分析，不是作者已证明结论）

修复式 refinement 是被理论化的**利用型算子**（每步试图严格改进父代），其探索不来自算子而来自"选哪个父代继续修"的分配层——把探索-利用从生成层挪到分配层的干净示范。同一父代的重复投入存在边际递减，失败历史可直接进入分配决策。REx 迁移到 AAD 属于合理外推而非已验证事实：AHD fitness 连续且含噪，Bernoulli 化需要额外的二值化设计（Clade-AHD 的 Beta-Bernoulli 更新是一例）。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 可学习点：对每个锚点维护"继续修的成功率信念 + 已投入次数"，失败越多权重越低但非零，用采样而非硬规则选下一个精炼对象。前提：失败可归因到父代。风险：评价噪声被计为失败。最小验证：与固定秩选择同预算对照，报告后验校准（预测成功率 vs 实际改进率）。
- 可学习点：把"自适应分配"的价值预期放在超参鲁棒性与跨任务稳定性，而非单任务峰值。最小验证：多任务上比较固定深宽谱系与自适应规则的排名方差。
