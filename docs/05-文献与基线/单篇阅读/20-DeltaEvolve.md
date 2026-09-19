# DeltaEvolve

- 论文：DeltaEvolve: Accelerating Scientific Discovery through Momentum-Driven Evolution；本地来源：main.tex，分文件 `3_framework.tex`、`4_methods.tex`、`5_experiments.tex`、`appendix.tex`；设计对象为科学代码/候选的迭代进化。

## 1. 核心问题与方法

DeltaEvolve 不直接保存历史完整代码，而让 LLM 为每条父子边生成 `semantic delta`：改了什么、为何改变、表现是 Improved/Degraded。`§DeltaEvolve` 将其放入 Level-1/Level-2 多层数据库；Progressive Disclosure Sampler 取 top-$k$ 与 diverse-$m$ 的 delta，再配当前父代完整代码生成下一候选。作者将连续 semantic delta 类比优化中的动量。

## 2. 论文宣称的机制贡献（逐项）

1. 从连续改进中抽取动量方向以引导变异。
2. 以动量加速有限预算下的科学发现。
3. 在多任务中取得更快/更高的结果。

## 3. 实验究竟支持了什么

|机制主张|论文证据（具体表/图/消融/章节）|证据等级|判断|
|---|---|---|---|
|整体质量与 token 消耗|§Experiments/§Main Results，表 `tab:main_exp`；五域、三 seeds（11/42/100）|直接支持（整法有效性）|匹配协议下的整法比较直接支持 DeltaEvolve 方案在五域、三 seeds 下优于对比基线；不能顺带推出 semantic delta、多层库或 sampler 各自独立有效（需看逐项消融）。|
|“保留代码上下文比标量分数重要”|§Evolutionary Framework/§Context Selection Dominates Scalar Feedback，表 `tab:ablation_raw`：Standard、Blind-Elite、Random-Context|直接支持|这是目标问题的受控上下文/分数对照；它支持选择的代码上下文价值，不等于直接证明 delta 动量。|
|semantic delta 的独立因果|正文未给出仅移除 delta、保留多层库/采样器其余不变的消融|未验证|主表不能拆解为动量本身有效。|
|过程轨迹|Appendix Task Details 的 `fig:bbob_evolution` 等为固定 seed 42；Case Study 的 `fig:case_study`|间接支持|代表运行与案例，不替代重复统计。|

## 4. 机制的底层逻辑

阅读分析：delta 相当于把父子差分编码为局部方向信息，适合渐进可组合的改进；当评价地形高度非平稳或一次改动跨越机制边界时，动量可能把搜索惯性锁在错误方向。

## 5. 对 LLM4AD / TraceAAD 可学习之处

|可学习点|成立前提|主要风险|最小验证方式|
|---|---|---|---|
|提示中呈现连续边的“改了什么”|差分可读且 lineage 可靠|把语法差分当功能差分|在少量边上人工/静态核验功能变化，再做有无 delta 对照。|
|依据连续增益调节延续或分叉|多步趋势超过评价噪声|追逐偶然连胜|要求跨两个独立评价窗口一致。|
