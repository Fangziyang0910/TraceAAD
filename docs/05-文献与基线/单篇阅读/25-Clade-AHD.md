# Clade-AHD

- 论文：Clade-AHD: Clade-level Selection for MCTS in Automatic Heuristic Design；本地来源：main.tex 与 appendix.tex；设计对象为 MCTS-AHD 的谱系（clade）级选择。

## 1. 核心问题与方法

Clade-AHD 指出逐节点 UCT 在稀疏访问下"结构性过度开发"：点估计被评价噪声支配（特定随机种子使次优启发式瞬态高分），节点级贝叶斯方法又假设兄弟独立、系统性低估"单独表现差但派生优秀后代"的垫脚石节点。修正为**clade 级 Beta 信念 + 温度化 Thompson 采样**：叶节点评价归一化后更新 Beta 参数，自底向上按深度衰减聚合后代证据（$\alpha_{\mathcal T}(v)=1+\sum_u\lambda^{dist}(\alpha_u-1)$，$\lambda=0.8$；$\lambda=0$ 即节点级近视基线），加伪计数 $n_{pseudo}=10$ 稳定，采样 $\hat\theta_v\sim\mathrm{Beta}(\tilde\alpha\tau(p),\tilde\beta\tau(p))$ 取最大；预算退火 $\tau(p)=(1/(1-p))^{1.0}$ 使探索随预算耗尽过渡为确定性利用；动态冻结（访问 $\geq$10 且均值低于全局 $0.1\times$）硬剪枝。六动作 prompt 与 MCTS-AHD 完全相同（附录声明 without any modification），以隔离搜索策略贡献。

## 2. 论文宣称的机制贡献（逐项）

1. clade-level selection 降低节点级噪声和早熟利用。
2. 家族级多样性保留不同改进方向。
3. 与 MCTS-AHD 相比改善 AHD 搜索效果。

## 3. 实验究竟支持了什么

|机制主张|论文证据（具体表/图/消融/章节）|证据等级|判断|
|---|---|---|---|
|整体比较|§Experiment，表 `tab:step-by-step`、`tab:step-bpp-online`、`tab:aco`；收敛图 `fig:convergence_curves`|直接支持（整法有效性）|匹配协议下的整法比较直接支持 Clade-AHD 方案在所测任务上优于基线；不能顺带证明 clade 选择单独造成全部领先（组件贡献需看逐项消融）。|
|clade 选择|附录组件消融表 `tab:ablation_components` 与 $\lambda$ 敏感性（TSP50，3 次）|部分支持|去动态冻结退化最大（+7.751%）、去温度退火 +6.657%、去深度衰减 +4.950%、去伪评价 +1.869%、去自适应归一化 +1.870%；$\lambda=0$（节点级）明显最差。缺 UCT↔Thompson 的显式选择规则交换消融（prompt 已相同，主表接近受控）。|
|优势并非全域|`tab:step-by-step`、`tab:aco`|部分支持|TSP50 构造与 ACO MKP 上 MCTS-AHD 仍反超（9.69 vs 10.39%）；KP100 上各组件影响 ≤0.862%。|
|谱系覆盖|`flow.pdf`、`comparison.pdf`|间接支持|流程/可视化不是性能因果证据。|

## 4. 机制的底层逻辑

阅读分析：clade 是对“同一早期设计思想的多个后续实现”的共享信用。它能降低单一 child 的评价方差，但也可能让弱祖先的家族持续占预算；clade 切分规则决定统计单元，必须明确。

## 5. 对 LLM4AD / TraceAAD 可学习之处

|可学习点|成立前提|主要风险|最小验证方式|
|---|---|---|---|
|按共同早期边聚合路线信用|谱系准确|家族定义任意|固定两种切分，比较预算分布和后继增益。|
|把家族探索与节点扩展分层|每层都有独立日志|双层 UCB 难审计|记录 clade 选择、节点选择、生成、回传四事件。|
