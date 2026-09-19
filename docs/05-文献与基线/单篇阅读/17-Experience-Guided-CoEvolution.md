# Experience-Guided Reflective Co-Evolution

- 论文：Experience-Guided Reflective Co-Evolution of Prompts and Heuristics；本地来源：iclr2026_conference.tex；设计对象是 AHD 中 prompt 与启发式代码的协同演化。

## 1. 核心问题与方法

论文不把 prompt 当作固定外壳：由启发式评价及反思经验同时更新 prompt 与代码。选择机制是 5 岛 MAP-Elites 式档案（按行为特征占格，空格或更优才替换，定期迁移让精英跨岛竞争）；变异侧是 5 档策略谱系（参数修改→冗余消除→结构修改→启发式规则重写→完全重写），按经验匹配采样，从保守到破坏性构成探索梯度。探索/利用模式的表述存在（探索=随机采样父代，利用=跨描述符持续高质量的启发式优先），但切换规则原文未给出。演化的 prompt 呈层级结构：代码无效时限制任务为错误修复，有效后才转向性能优化（Case Study 支持）。

## 2. 论文宣称的机制贡献（逐项）

1. prompt 与启发式共同进化，缓解固定提示的表达瓶颈。
2. 以正负经验引导反思和下一代提示。
3. 经验驱动提高搜索效率与代码质量。

## 3. 实验究竟支持了什么

|机制主张|论文证据（具体表/图/消融/章节）|证据等级|判断|
|---|---|---|---|
|联合框架主效果|§Experiment/§Experiment Result/§Main Result，表 `main_result`（TGB、BOB）|直接支持（整法有效性）|匹配协议下的整法比较直接支持联合框架在所测任务上的整体优越性；不能顺带证明 prompt 进化、经验或岛屿选择各自独立有效（组件贡献需看消融表）。|
|提示共同进化的贡献|§消融表 `ablation`（w/o Strategy Sampling / w/o Prompt Evolution / w/o Island-Elites × 6 个初始启发式）|部分支持|三变体消融存在：w/o Prompt Evolution 显著（Christofides 5.17→9.24%），w/o Island-Based Elites Selection 最大（nearest-insertion 4.41→9.70%）；但无重复次数与方差报告。|
|经验质量改善生成|`TSPPrompt.pdf`、`BPPPrompt.pdf`、`Case.pdf` 案例|间接支持|案例解释可读，不是重复的因果检验。|

## 4. 机制的底层逻辑

阅读分析：代码空间和提示空间相互条件化，等价于同时搜索“候选”和“产生候选的局部语言”。它可能扩大可到达区域，但 prompt 改变也改变了每次调用的信息量；若不把调用、tokens 与评价成本计入，效率归因会混杂。

## 5. 对 LLM4AD / TraceAAD 可学习之处

|可学习点|成立前提|主要风险|最小验证方式|
|---|---|---|---|
|把轨迹经验写成可审计的生成约束|经验确实来自已评价边|提示自我强化错误规律|固定一份经验，做有/无经验的配对下一代试验。|
|把提示版本与程序谱系共同记录|每次调用可追溯|状态空间膨胀|只记录关键指导差分和生效边。|
