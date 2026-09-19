# Evolving Code with a Large Language Model

- 论文：*Evolving Code with a Large Language Model*（Hemberg, Moskal, O'Reilly，MIT CSAIL，期刊版）；本地来源：`papers/Evolving_Code_with_A_Large_Language_Model/ELM_GP_jrnl_2023.tex`；设计对象：符号回归 Python 表达式（框架化论文，非新机制论文）。

## 1. 核心问题与方法

该文把"LLM 进化代码"抽象为 **GP+LLM 算子代数**：初始化 $i_{LLM}$、执行 $e_{LLM}$、适应度 $\phi_{LLM}$、选择 $s_{LLM}$、交叉 $v_{xoLLM}$、变异 $v_{muLLM}$、替换 $r_{LLM}$、选优 $b_{LLM}$ 八个算子全部形式化为三步协议（**Formulate** prompt → **Interface** LLM → **Check** 响应），并指出 Koza 五步准备工作中有两步被"为每个算子设计 prompt-function"取代——人的努力从设计原语/算子转移到设计提示。EA 循环不变，每个算子都可以（但不必）外包给 LLM。演示实验是**简化符号回归**（ALFAECLLM 包，pop=10、30 代、每 run 300 次适应度评价、30 runs、GPT-3.5-turbo few-shot 2-shot 样例取自当前种群；GP 基线用锦标赛 size=2 + 精英 1）。注意：该文不使用 ELM 的 diff 模型、Sodaracer 或 MAP-Elites。

## 2. 论文宣称的机制贡献（逐项）

- 算子代数与三步协议的形式化；prompt 模板语法 `<ρ> ::= <EXAMPLES><QUERY><PRIMITIVES><RESPONSE_FORMAT>`。
- 相关工作分类学：LLM 行为引导两类（升温增加多样性、微调）；prompt 工程技术清单（Template、变温、Chaining、Few-shot、Summarization 等）。
- 成本与错误率的实证刻画：每 run 均值 LLM-only 837s/\$2.63、GP+LLM 1664s/\$3.90、GP+someLLM 743s/\$1.87、GP 0.1s；选择/替换算子 token 消耗最大、错误最多，变异/初始化最便宜稳健。

## 3. 实验究竟支持了什么

|机制主张|论文证据（具体表/图/消融/章节）|证据等级|判断|
|---|---|---|---|
|各算子可 LLM 化并有可测代价|Experiments 的成本/时长/错误率统计|直接支持|对演示级符号回归成立。|
|哪些算子适合 LLM 化|错误率与 token 分析（选择/替换最差，变异/初始化最稳）|部分支持|单一任务、GPT-3.5；归因未与任务难度分离。|
|LLM 化优于经典 GP|主对比表|未验证|演示级实现（无多样性维护，种群上限受 4096 token 上下文约束），无跨任务结论。|
|无探索-利用相关消融|—|—|该文不机制化 exploration/exploitation；温度作为"更多样解"的旋钮只出现在相关工作转述中。|

## 4. 机制的底层逻辑

阅读分析：该文的真正贡献是把"LLM 该接管 EA 的哪些算子、代价几何"问题显性化，并给出第一个系统的成本/错误 profile。GP+LLM 的计算代价度量须在 FE 之外加 token 数与调用数；种群规模本身是 LLM 上下文窗口的函数——这两个工程事实后来成为所有框架的隐含约束。

## 5. 对 LLM4AD / TraceAAD 可学习之处

- 可学习点：把搜索循环的每个决策点显式列出并问"谁来做、代价多少"。前提：决策点可枚举。风险：形式化掩盖各决策点间的耦合。最小验证：对现有搜索日志按决策点统计调用量与错误率。
- 可学习点：token/调用数与评价数分开核算（与仓库"评价预算与生成成本必须分开"的口径一致）。
