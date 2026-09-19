# PathWise

- 论文：PathWise；本地来源：example_paper.tex；设计对象为以改进路径为中心的 LLM 启发式搜索。

## 1. 核心问题与方法

PathWise 把启发式发现建模为**蕴含图上的序贯决策 + 多智能体规划**：节点 $v=(h,\kappa,d,P(h),\mathrm{PM})$（代码、自然语言推导理由、描述、性能、父代元数据），有向边记录"如何从父代集推导而来"，允许多父。策略 agent 产出动作 $(S,\kappa)$（父代集 + 动态发明的推导指令，算子不再取自固定集合）；world model agent 每动作 rollout $N_w=2$ 个候选程序，$N_a=2$ 个动作共 4 个 rollout 全部评价、只最优者入图；policy/world-model 两个 critic 用动作平均回报排序与最优-最差对比产出语言反思。结构是**图-种群双时间尺度**：外环维护 $N_p=6$ 个根节点种群，内环在每个种群上建蕴含图（上限 $I_{max}=3$ 步）；状态更新加新节点、删被用掉的父代（全局最优保留）。探索在 prompt 层而非统计层：时间衰减扰动率 $\varepsilon$（0.5→0.25）从探索短语库采样注入 + 状态洗牌消除位置偏置。预算按评价次数统一 $n_e=500$。

## 2. 论文宣称的机制贡献（逐项）

1. 蕴含图（多父 DAG）与父代元数据压缩摘要保留推导历史，替代访问统计做选择依据。
2. 策略/世界模型/critic 多智能体分工：critic 的语言反馈替代 UCT 型数值回传。
3. prompt 级探索（$\varepsilon$ 扰动注入 + 状态洗牌）作用于 LLM 采样分布的形状，SDR（选择多样性率）为过程指标（完整版 75.79%）。

## 3. 实验究竟支持了什么

|机制主张|论文证据（具体表/图/消融/章节）|证据等级|判断|
|---|---|---|---|
|整体结果|§Overall Results，`tab:tsp_kp_step_by_step`、`tab:aco_general_framework_merged`；曲线 `fig:evolution_same_ne_exp`|直接支持（整法有效性）|匹配协议下的整法比较直接支持 PathWise 方案在所测 TSP/KP/ACO 任务上优于基线；不能顺带证明图状态、agent 分工或提示内容各自的独立贡献（需看逐项消融）。|
|critic 反馈|§Ablation Study，`tab:ablation-ours`：固定其余组件，移除 policy/world-model critics|直接支持|目标 critic 机制的受控消融，限 TSP 构造、GPT-5-nano(low)、5 次平均。|
|prompt 多样性|§Ablation Study，`tab:ablation-diversity-v2`：固定其余组件，移除 prompt perturbation/state shuffling|直接支持|支持该任务上两个多样性处理的局部贡献。|
|路径可解释性|方法示例/路径图|间接支持|仅说明记录形式。|

## 4. 机制的底层逻辑

阅读分析：蕴含图把候选价值从静态 $f(program)$ 扩展为"该路线过去如何响应修改"；状态更新规则（保留全局最优=利用、剪除已用父代=复杂度控制、否定性探索短语=探索）承担了 UCT 在树方法里的分工。若父子边缺失或不同路线的评价条件不同，图价值会被错误归因；world model 是预训练知识的 prompt 使用而非学习的动力学模型。

## 5. 对 LLM4AD / TraceAAD 可学习之处

|可学习点|成立前提|主要风险|最小验证方式|
|---|---|---|---|
|以 trajectory 为最小记忆单元|完整 lineage 与操作记录|把长路径全部塞入上下文|固定 token 上限，比较最近边、压缩路径、无路径。|
|把路线价值与终点分数分开|有可验证的后继收益|后见之明挑选成功路径|预先定义路线评分，按未来增益验证。|
