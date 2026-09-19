# Memento 2

- 论文：*Memento 2: Learning by Stateful Reflective Memory*；本地来源：LaTeX 源码目录 `Memento_2_Learning_by_Stateful_Reflective_Memory/`；研究对象：冻结参数、靠情景记忆 + 反思持续学习的 LLM 智能体的 RL 理论（纯理论，无新实验）。

## 1. 核心问题与方法

为"记忆驱动的免训练自改进"建立收敛理论。设定 SRDP 元组 $\langle\mathcal S,\mathcal A,\mathcal P,\mathcal R,\gamma,\mathfrak M,p_{LLM}\rangle$：每步两段动作——检索 $c_t\sim\mu(\cdot\mid s_t,M_t)$ 再行动 $a_t\sim p_{LLM}(\cdot\mid s_t,c_t)$，复合策略 $\pi^\mu(a\mid s,M)=\sum_c\mu(c\mid s,M)p_{LLM}(a\mid s,c)$。**Reflected MDP**：增广状态 $x=(s,M)$ 恢复马尔可夫性，动作空间即记忆 $\mathcal C(M)=M$，LLM 被吸收进环境（转移 $\mathcal P_{LLM}$、奖励 $\mathcal R_{LLM}(x,c)=\sum_a p_{LLM}(a\mid s,c)\mathcal R(s,a)$），唯一可控决策是检索策略 $\mu$。Read=策略改进、Write=策略评价：Parzen 窗先验 $\mu_0$ + 空案例 $c_\varnothing$（混合系数 $\lambda(x)$ 实现"检索 vs 由 LLM 内部知识发现"），KL 正则软策略迭代有闭式 $\mu^+(c\mid x)\propto\mu_0(c\mid x)e^{Q(x,c)/\alpha}$。

## 2. 论文宣称的机制贡献（逐项）

- 定理 1：固定记忆下软策略迭代收敛到 KL 正则最优。
- 定理 2：双时间尺度（$\rho_t/\eta_t\to 0$、Robbins-Monro、鞅差噪声、记忆紧吸引集）下 $(Q_t,\mu_t,M_t)$ 联合收敛。
- 定理 3 与推论：值差 $\|V^{\pi^\star}-V^{\pi_M}\|_\infty\le\frac{2R_{\max}}{(1-\gamma)^2}\Delta_M$，且 $\mathrm{TV}(\pi_M,\pi^\star)\le\varepsilon_{LLM}(r_M)+\delta_M$——记忆覆盖半径 $r_M\to 0$、检索误差 $\delta_M\to 0$ 则渐近最优。核心假设"LLM 局部一致性"：在 $d(s,s(c))\le r$ 内 $\mathrm{TV}(p_{LLM}(\cdot\mid s,c),\pi^\star(\cdot\mid s))\le\varepsilon_{LLM}(r)$，即 LLM 对参考案例的胜任半径。

## 3. 实验究竟支持了什么

|主张|论文证据|证据等级|判断|
|---|---|---|---|
|收敛与渐近最优保证|定理 1–3 及证明|直接支持（理论）|假设清单明确（双时间尺度、局部一致性、紧吸引集），均为渐近性、无样本复杂度。|
|记忆系统实践有效|引用 Memento（zhou2025）、CBR-LLM（数据科学/软件测试）、Agent K 的既有实验|间接支持|本文无新实验；实证全部来自同作者序列前置工作。|

## 4. 机制的底层逻辑与理论适用边界

Memento 2 将值差形式化分解为“LLM 局部胜任误差 $\varepsilon_{LLM}$ + 记忆覆盖误差 $\delta_M$”——记忆的作用在于缩小生成器需要泛化的局部邻域。

**绝不能将 Memento 2 直接作为 TraceAAD 的理论证明（保留为状态、动作与信用分解的启发，无法对应处严禁使用“同构”略过）**：
1. **渐近收敛 vs 有限预算选父**：Memento 2 的检索策略、局部一致性和渐近收敛条件，建立在双时间尺度随机逼近（$t \to \infty$、$\rho_t/\eta_t \to 0$）的无穷步极限上，不给出有限样本复杂度；它不自动等同于有限真实评价预算（如 1000 eval）下的父代选择与路线调度问题；
2. **“局部一致性”假设与代码空间脱节**：理论核心假设 LLM 在参考案例的局部邻域 $r$ 内策略全变差距离有界（$\mathrm{TV} \le \varepsilon$）。但在高度非凸、微小语法修改即引起语义剧变的算法代码空间中，这一平滑连续性假设缺乏实证基础；
3. **“慢记忆更新”是证明收敛的数学工具，绝非工程操作准则**：理论中要求记忆更新慢于策略评估（双时间尺度），是证明鞅差噪声消解的技术手法，**绝不能用来要求 TraceAAD 延迟或按批次保存每步客观搜索事实**。TraceAAD 的每一次真实评估产生的结果、代码、报错与耗时事实必须即时、无损入库；
4. **拒绝虚假的“同构”包装**：增广状态 $x=(s,M)$ 和检索动作 $c$ 提供了一种把历史检索纳入 MDP 的概念框架，但其转移核与奖励函数定义均依赖强马尔可夫性假设，与黑盒 LLM+evaluator 的序贯搜索有本质区别，不能用“同构”略过状态、动作与反馈机制的根本不同。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- **分析视角的启发，非数学定理的继承**：把搜索性能拆解为“生成器在给定上下文下的局部胜任能力”与“检索/选父机制的历史覆盖程度”，有助于我们在诊断搜索停滞时，分清是上下文信息不足还是模型能力墙。
- **坚守事实即时记录底线**：明确理论中的“慢更新”与系统工程中的“实时事实落库”毫无关系，不将论文中的理论约束错误转化为降低工程数据保真度的借口。
