# ShinkaEvolve

- 论文：ShinkaEvolve；本地来源：`main.tex` 与 `sections/`；设计对象：可执行优化程序、MoE load-balancing loss、竞赛启发式等。

## 1. 核心问题与方法

ShinkaEvolve 是带档案的 LLM 演化 harness：多模型提出程序修改，系统依质量选父、以自适应 bandit 分配模型调用，并以嵌入拒绝采样过滤过于相似的提案；任务 evaluator 执行候选。论文展示圆打包、MoE 负载均衡损失和 ALE-Bench 等不同对象，并讨论同步/异步吞吐的取舍。

## 2. 论文宣称的机制贡献（逐项）

- 自适应 bandit 按历史 fitness 增益分配不同模型。
- embedding novelty filter 减少近重复代码、节省评估。
- 档案与父代选择支持长期累积的程序改进。

## 3. 实验究竟支持了什么

|机制主张|论文证据（具体表/图/消融/章节）|证据等级|判断|
|---|---|---|---|
|不同模型分配策略的结果不同|`sections/05_analysis.tex` 的 model-selection analysis|直接支持|直接比较文中列出的分配策略，但结论限于其任务/模型池。|
|embedding rejection 改变重复提案处理|`05_analysis.tex` novelty-filter 分析（阈值 0.95）|直接支持|支持过滤规则的操作效果；不自动等于最终质量提升。|
|过滤提升最终搜索|`05_analysis.tex` 的 No Rejection / Embedding / LLM-judge 对照|部分支持|对照可支持该设置的总体效果，不能证明嵌入相似度准确刻画语义新颖性。|
|跨域发现普适有效|`sections/04_results.tex` 的圆打包、MoE、ALE-Bench 案例|间接支持|异质案例和不同预算不构成统一机制试验。|

## 4. 机制的底层逻辑

阅读分析：bandit 将模型调用看作资源分配，novelty filter 将档案变成反重复约束；两者都在“生成前”塑造搜索，而 evaluator 仍定义最终成功。嵌入阈值可减少同文异义/异文同义的误判之一却未必同时解决二者；bandit 又可能过早偏向短期高收益模型，牺牲后续互补探索。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 可学习点：以贡献而非固定比例分配生成来源。前提：贡献归因窗口稳定。风险：短期噪声造成饥饿。最小验证：记录每模型调用、有效率、后续 best 增益，并与均匀分配等预算比较。
- 可学习点：对历史轨迹做去重。前提：相似度与行为冗余相关。风险：过滤掉有价值的小改动。最小验证：人工/行为抽样估计误拒率，再比较 held-out 改进。
