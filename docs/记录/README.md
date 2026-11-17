# 当前实验与已验证边界台账

本页只保留当前有效对照批次与经代码或真实树数据核实的客观事实。早期迭代记录与废案讨论已移入本地封存目录 `archive/documents_20261116_history/`，讨论当前机制设计时不引用旧阶段推论。

## 1. 当前主对照批次

| 方法 | 批次标识 | 范围与预算 | server4 冻结源码 | 结果目录与状态 |
| --- | --- | --- | --- | --- |
| EvoTrace V3 · 定型版 | `20261117_181830_evotrace_v3_clean_co4` | 四任务 × 3 种子，1000 评/路 | `/home/fzy/code/LLM4AD/TraceAAD_runs/20261117_181830_evotrace_v3_clean_co4` | `experiments_result/evotrace_v3_clean`（北京时间 11-17 18:19 启动，运行中；当前默认看板实验） |
| EvoTrace V3 · 事实反思版 | `20261117_153134_evotrace_v3_factual_co4` | 四任务 × 3 种子，1000 评/路 | `/home/fzy/code/LLM4AD/TraceAAD_runs/20261117_153134_evotrace_v3_factual_co4` | `experiments_result/evotrace_v3_factual`（11-17 18:17 停止，每路 253–321 评、共 3567 评） |
| EvoTrace V3 · 代表覆盖版 | `20261116_165500_evotrace_v3_rep_cover_co4` | 四任务 × 3 种子，1000 评/路 | `/home/fzy/code/LLM4AD/TraceAAD_runs/20261116_165500_evotrace_v3_rep_cover_co4` | `experiments_result/evotrace_v3_rep_cover`（11-17 停止：FSSP/OP 跑满 1000 评，TSP/CVRP 停于 797–874 评；论文第 3 节与附录 F 当前固定前缀数据来源） |
| EvoTrace V3 · 精简版 | `20261117_132027_evotrace_v3_lean_co4` | 四任务 × 3 种子，1000 评/路 | `/home/fzy/code/LLM4AD/TraceAAD_runs/20261117_132027_evotrace_v3_lean_co4` | `experiments_result/evotrace_v3_lean`（11-17 停止并作废，每路 91–137 评；机制为提交 `1e6552b3`，论文第 2 节与附录 E 当前按此描述） |
| EvoTrace V3（早期） | `20261116_000118_evotrace_v3_full_co4` | 四任务 × 3 种子，1000 评/路 | `/home/fzy/code/LLM4AD/TraceAAD_runs/20261116_000118_evotrace_v3_full_co4` | `experiments_result/evotrace_v3_full`（已停止；提交 `a28b1ed7` 早期 40 评窗口快照） |
| EvoTrace V1 | `20261112_165634_evotrace_v1_co4` | 四任务 × 3 种子，1000 评/路 | `/home/fzy/code/LLM4AD/TraceAAD_runs/20261112_165634_evotrace_v1_co4` | `experiments_result/evotrace_v1`（已完成搜索与独立测试） |
| TraceAAD Base | `20261109_base_boltzmann_co6` | 六任务（看板取四任务）× 3 种子，1000 评/路 | `/home/fzy/code/LLM4AD/TraceAAD_runs/20261109_traceaad_base_v23_2` | `experiments_result/traceaad_base_boltzmann_20261109`（纯 Boltzmann 基线，已完成四任务搜索与独立测试；默认看板参照） |

注：当前主线源码（提交 `190fe372`）与运行中的事实反思版 `evotrace_v3_factual` 机制一致，并进一步剥离了 V1–V3 内部工程包装（独立包不共享 `traceaad_base`）。双语论文稿 `docs/paper*.md` 第 3 节与附录 F 当前使用 `evotrace_v3_rep_cover` 固定前缀快照，第 2 节与附录 E 对应减法版（`1e6552b3`），待 `evotrace_v3_factual` 推进后统一更新正文与主实验表。

## 2. 已核实的机制边界与证伪记录（防重复试错）

以下条目仅记录经数学构造或真实运行快照重算确认的客观现象：

1. **局部宽深裁剪不等于全局有界：** 仅限制单父代保留子数与无分叉链长时，完全分叉树仍随深度指数增长；在简化规则中混入父子适应度淘汰会与全局抽样职责重叠，并在移除叶节点后连锁暴露冷父代、破坏单轮简化幂等性。
2. **代码相似度硬过滤的固有折中：** AST 结构二元组特征忽略数值常数变化（常数调优的距离为 `0.0`），硬阈值去重会拦截有效参数改进；不设相似度过滤时，离散或平台型任务（如 FSSP）上同分改写会在活跃窗口内并存并挤占个体排名。
3. **跨路径同分合并与峰值分级排名的副作用：**
   - 直接将“同标量分数”当作同一个体去重，会在 FSSP 等离散目标任务上合并删除结构显著不同的实现（真实轨迹同分组内最大 AST 距离中位数约 `0.40–0.43`）。
   - 仅按路径峰值分数集合计名次并跨路径合并精确同分投入，在真实树扰动下对 `1e-16` 浮点末位差异存在超过 `50%` 的总变差跳变；峰值层数较少时会将后半段差节点采样份额从约 `6%` 抬升至 `26%–32%`。
4. **分层或分组分配对深挖锐度的影响：** 在真实树快照上，先按 `lineage_peak` 分组再组内分配或从根沿树逐层递归选择，均大幅压低连续改进任务（如 TSP）前排个体的抽样份额。
5. **Explore 全局概览不能仅展示路径峰值：** 节点经验可挂在非峰值分叉节点上，边经验可进入非峰值保留节点，且绝大多数保留的 `Explore` 尝试并非路径峰值；仅展示峰值在真实快照上会遗漏近半数经验载体（如 FSSP seed1 中 `13/25`）及 `98%` 已尝试的非峰值探索草稿。
6. **路径收缩与峰值计费的耦合边界：** 路径收缩保持存活节点的 `subtree_evaluations` 原值不变；但当非全局最优的路径峰值因超出活跃窗口且非分叉点而被收缩移除后，其未改进后代会在当前保留树上重新计算路径最优祖先。

## 3. 设计与筛选记录

- [对比任务名单](2026-11-16-EvoTrace对比任务筛选.md)（2026-11-16）：固定主实验四项经典 AHD 任务（`tsp_construct`、`cvrp_aco`、`op_aco`、`fssp_gls`），记录其余任务排除依据。
- [对比基线名单](2026-11-16-EvoTrace基线系统筛选.md)（2026-11-16）：固定五个外部基线（`EoH`、`ReEvo`、`MCTS-AHD`、`RefineEvo`、`PathWise`）与内部参照体系。
- [演化轨迹上下文的设计](2026-11-17-EvoTrace演化轨迹上下文设计.md)（2026-11-17）：以经验和 design 为主体，代码只给被改写或被组合的算法，不给 diff；Explore 改为代表个体视图（提交 `1e6552b3`）。
- [V3减法](2026-11-17-EvoTrace-V3减法.md)（2026-11-17）：边、节点反思合为一个模板，每个算法给出来源与结果、design、失败、已有经验与代码，删去代码差分与比较账本；经验写明起始计算、修改与结果，不把单次结果写成规则或禁令（提交 `1e6552b3`；`evotrace_v3_context` 批次作废）。
- [事实反思](2026-11-17-EvoTrace-V3事实反思.md)（2026-11-17）：反思只记录计算与测得结果，不写规则或编号；Explore 挂根开始新谱系，根经验只给 Explore；上下文改用设计语言；根区域按适应度从好到差排列（`evotrace_v3_lean` 批次作废，`evotrace_v3_factual` 启动；源码并入 `190fe372`，同时移除 V1–V3 内部工程包装）。
- [三代融合动态树设计构想](2026-11-17-EvoTrace三代融合动态树设计.md)（2026-11-17）：结合 V1 统一分差竞争与边际差异代表选择、V2 动态活跃保留核（$Q \cup A$）与自由生长、V3 挂根拓扑与分块纯事实反思/精简上下文的设计构想与待评审边界（讨论稿，不改动运行中批次）。
