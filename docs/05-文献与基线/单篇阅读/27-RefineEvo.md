# RefineEvo

- 论文：RefineEvo: Planning-Guided Heuristic Evolution with Bidirectional Experience；本地来源：main.tex，分文件 `03_method.tex`、`04_experiment.tex`、`06_appendix.tex`；设计对象为组合优化的启发式进化。

## 1. 核心问题与方法

RefineEvo 保持种群进化骨架（N=10、精英截断、1000 评价），把"静态试错"改造为规划引导 + 经验驱动：Planner 感知种群状态 $\mathcal S_t=(\Phi_{pop},\Psi_{ops})$——含种群多样性 $D_t$（分数标准差归一化）与改进率 $R_t$（窗口 $w=1$ 的最优值增量）、各算子近效（代码有效率、子代超父率）——输出 exploration/exploitation 模式开关与算子优先级（早熟收敛→Exploration 模式优先结构新奇算子；稳步提升→Exploitation 模式局部精炼）。经验分正负两库，各条含修改摘要、建议与适用条件，语义检索 top-3 注入 prompt（"Insights to Follow" + "Pitfalls to Avoid"）；经验效用分 ±1 更新、负分删除、矛盾条目互相惩罚。算子描述本身可被事件触发的 incremental/radical 重写（连续无改进时）。

## 2. 论文宣称的机制贡献（逐项）

1. 规划指导降低盲目代码变异。
2. 正负双向经验共同引导改进。
3. 在 ACO 等任务上提升质量、收敛和存活率。

## 3. 实验究竟支持了什么

|机制主张|论文证据（具体表/图/消融/章节）|证据等级|判断|
|---|---|---|---|
|整体效果|`04_experiment.tex` 主结果；`curves.pdf`、`bar_chart.pdf`、`aco_results_*`|直接支持（整法有效性）|匹配协议下的整法比较直接支持 RefineEvo 方案在所测任务上优于基线；不能顺带证明规划与双向经验各自独立有效（组件贡献需看逐项消融）。|
|双向经验池|Appendix 的 `tab:ablation_bep`：w/o Experience 15.91%、w/o Negative 13.94%、w/o Positive 12.92%（完整 11.54%，TSPLIB）|直接支持|移除对照齐全且方向一致；**负经验比正经验更重要**（去负退化大于去正），负经验承担"不重复失败"的探索侧保护。|
|规划/操作精炼|Appendix `tab:full_k_ablation` 比较 Random/Planner Selection 与 Fixed-Interval Refinement 的 $k$|部分支持|参数/策略对照支持所测设置，不证明“规划”全部语义内容。|
|经验避免失败|`survival_rate_heatmap1.pdf`、`survival_rate_heatmap2.pdf`|间接支持|过程统计可描述现象，不能确认经验是唯一原因。|

## 4. 机制的底层逻辑

阅读分析：负经验的作用是建立“不要重复什么”的约束，正经验给出可延续的方向；规划使两者不只作为附注而参与操作选择。关键风险是失败常依赖父代、实例和 evaluator，去上下文的负规则可能错误排斥新组合。

## 5. 对 LLM4AD / TraceAAD 可学习之处

|可学习点|成立前提|主要风险|最小验证方式|
|---|---|---|---|
|成对保存成功边与失败边的条件|失败能定位到实际改动|把噪声/timeout写成机制失败|仅纳入重复评价一致的失败。|
|规划建议必须回链到实际操作|操作与提示均有日志|“规划”停留在叙述层|统计每类计划的执行率和一步收益。|
