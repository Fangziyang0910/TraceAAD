# HiFo-Prompt

- 论文：HiFo-Prompt: Prompting with Hindsight and Foresight for LLM-based AHD；本地来源：templateArxiv.tex；设计对象为组合优化与 BO 的启发式/策略提示生成。

## 1. 核心问题与方法

HiFo-Prompt 将指导拆为 hindsight 与 foresight。前者从精英代码蒸馏“insight”，经文本 Jaccard 新颖性筛选、效用/新近性/使用惩罚维护，并检索入下一轮 prompt；后者由 Evolutionary Navigator 读取进展、停滞和语义多样性，选择更偏探索或利用的指导。两者和基础操作提示组成 Guided Prompt Synthesis。

## 2. 论文宣称的机制贡献（逐项）

1. Insight Pool 使知识脱离被丢弃代码而可累计。
2. Navigator 用种群态势做主动搜索调控。
3. 二者合用提升质量、收敛速度和 query efficiency。

## 3. 实验究竟支持了什么

|机制主张|论文证据（具体表/图/消融/章节）|证据等级|判断|
|---|---|---|---|
|主结果优于基线|§Experiments 的 TSP/BPP/FSSP/BO 表；同为 Qwen2.5-Max 的 AHD 比较|直接支持（整法有效性）|匹配协议下的整法比较直接支持 HiFo-Prompt 方案在所测任务上优于基线；不能顺带分别证明 Insight Pool 或 Navigator 各自独立有效（需看逐项消融）。|
|Insight 与 Navigator 分别有用|表 `tab:ablation`：TSP、Online BPP，去 Hindsight、去 Foresight、都去|部分支持|支持两个模块在该两任务的联合系统中有增益；不足以证明每项内部规则。|
|更快收敛|附录 comparative 的 EoH 比较|部分支持|曲线可支持该对照的达到分数速度；需与相同有效调用和 evaluator 成本核对。|
|参数稳健|附录 parameter analysis|间接支持|敏感性不是机制有效性的独立对照。|

## 4. 机制的底层逻辑

阅读分析：hindsight 解决“代码淘汰即丢失设计理由”，foresight 解决“只看父代分数而不看种群状态”。它们分别对生成上下文和搜索策略施加先验；但 insight 的文本相似度与真实算法语义不等价，Navigator 阈值也可能把随机波动误判为停滞。

## 5. 对 LLM4AD / TraceAAD 可学习之处

|可学习点|成立前提|主要风险|最小验证方式|
|---|---|---|---|
|从有效轨迹抽取可检索的“改进思想”|轨迹有可靠父子与评价|摘要脱离代码语义|比较原始轨迹、摘要轨迹、无记忆三种提示。|
|仅在可观测停滞时改变探索力度|状态指标与预算对齐|阈值调参掩盖机制|固定阈值、相同种子，记录触发次数及后续增益。|
