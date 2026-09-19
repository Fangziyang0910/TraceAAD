# Evolutionary Discovery of RL Algorithms via LLMs

- 论文：*Evolutionary Discovery of RL Algorithms via LLMs*；本地来源：`main.tex`；设计对象：固定网络与优化器下的 RL 更新规则/损失代码。

## 1. 核心问题与方法

论文搜索的不是策略网络结构，而是学习更新逻辑。GPT-5.2 与 Claude 4.5 Opus 依 prompt 生成候选算法代码；候选在五个 Gymnasium 训练环境上训练，将经环境归一化后的回报汇总为 fitness。

**变异与交叉算子机制**：
- **宏观变异（Macro Mutation）**：单步重写更新规则中一个具有完整语义的模块（概率 $p_{\mathrm{macro}}=0.65$），进行较大幅度探索；
- **多样性感知的交叉（Diversity-Aware Crossover）**：概率 $p_{\mathrm{cross}}=0.35$。第一父代 $f_1$ 按适应度 softmax 采样；第二父代 $f_2$ 依组合分采样：
  $$S(f_2 \mid f_1) = \alpha F(f_2) + (1 - \alpha)\, d_{\mathrm{lev}}(f_1, f_2)$$
  其中 $d_{\mathrm{lev}}$ 为两父代代码间的归一化 Levenshtein 距离。**该机制按公式用于交叉父代选择中的结构相异性加权（防止与近期变异产生的近似重复个体近亲交叉导致多样性崩溃），不直接等同于对子代变异幅度施加惩罚**。变异算子采用单模块宏观重写，属于大步长结构探索。最终选出 CG-FPD 与 DF-CWP-CP，并在十个环境比较 PPO、A2C、DQN、SAC。

## 2. 论文宣称的机制贡献（逐项）

- 语言模型可在更新规则空间提出超越预设 actor-critic/TD 结构的算法。
- 交叉父代选择中的结构相异性评分（Levenshtein 距离加权）：平衡第二父代质量与遗传相异性，避免近亲配对，不约束变异算子步长。
- 固定 256×256 MLP、Adam 和 action head，使比较聚焦学习逻辑。

## 3. 实验究竟支持了什么

|机制主张|论文证据（具体表/图/消融/章节）|证据等级|判断|
|---|---|---|---|
|演化 fitness 随代改善|Fig. `evolution_curves`（§Results）|直接支持|曲线为两 evolutionary seeds 的均值±标准差，支持该设置下的搜索进展。|
|发现算法在十环境与基线竞争|Table `results`、Fig. `env_comparison`、§Results|直接支持（整法有效性）|匹配协议下的整法比较直接支持演化生成的损失函数与策略在十个连续控制环境中展现出与基线竞争的整法表现；不能顺带证明是 LLM、演化或任一发现结构单独造成。|
|中等配对权重 $\alpha=0.5$ 优于极端配置|Fig. `alpha_ablation`、§Ablation Studies|部分支持|正文消融比较了 $\alpha=0$（纯相异度配对）与 $\alpha=1$（纯适应度配对，极易近亲重复），主实验 $\alpha=0.5$ 达到更高适应度区间；证明交叉配对中兼顾质量与多样性有效。|
|CG-FPD 不需要 terminal value bootstrap|Table `ablation_value_bootstrap`|部分支持|添加 TD(0) terminal bonus 降峰值、降方差；只检验一个具体改动。|

## 4. 机制的底层逻辑

固定表示和优化器减少了“把大网络容量当成算法创新”的混杂。**机制关键纠偏**：Levenshtein 距离只在 diversity-aware crossover 中用于挑选互补的第二父代，并不约束变异算子能改多少行或改多大。宏观变异本身是单模块整块替换的大步长算子。如果将 Levenshtein 误读为“突变幅度正则”，就会错误推导出“该论文主张限制 LLM 变异步长”。此外，fitness 由五个训练环境汇总，存在对训练环境过拟合的风险；后演化的超参数微调也使最终测试分数不能完全归因于纯搜索阶段。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- **结构距离用于配对而非压制变异**：用代码距离作为交叉配对或多父代融合（Fuse）的相异性约束是成立的，能防止同源代码自我繁衍；但不要用文本距离强行限制变异幅度。
- **实验隔离控制**：固定非目标组件（网络、优化器）使评价完全聚焦于算法逻辑，有助于清晰归因。
