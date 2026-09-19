# TraceAAD V10.7R 机制设计

## 1. 核心判断

V10.7 已经改善了单次生成的格式稳定性和调用成本，但它把原研究问题替换成了另一个较弱的问题：原本要利用历史证据提高下一次设计实验的价值，实际主要实现成了按质量分层展示若干程序，再让模型自行比较组合。

一次搜索机会应被理解为：

\[
a_t=(\text{设计底座},\ \text{设计任务},\ \text{参考证据})。
\]

真正关心的是该机会在剩余预算内带来的最好算法增益，而不是材料覆盖率本身：

\[
Q_h^\pi(a\mid H_t)=
\mathbb E_\pi\!\left[
\max\{F_t,f(x_{t+1}),\ldots,f(x_{t+h})\}-F_t
\mid H_t,a
\right].
\]

V10.7R 不拟合复杂的价值模型。它先做更小、可归因的修复：先确定 Refine、Pivot 或 Fuse 需要什么证据，再从真实档案中选择承担对应角色的完整程序。

## 2. 修复目标与边界

本轮改变：

- 将默认上下文从统一分层采样改为 `task_evidence_v1`。
- Refine、Pivot、Fuse 使用不同的材料角色和提示目标。
- 把直接生成边编码为有方向的设计实验，避免只展示程序后让模型猜测关系。
- 删除 AST 结构签名弱偏好：它既无价值证据，又引入一个代理、一个超参数和一堆代码；采样只保留任务权重。
- 程序块按固定语义角色组织，不再使用 `Algorithm N` 编号与 fitness 排序；输入视图称历史 Idea 为 `Design note` 并声明以代码为准。
- Refine 无证据时使用不提 contrast 的裸模板；Fuse 默认只用底座加迁移源两份程序。
- 输出协议改为严格的两段契约（一段 Idea 加一个代码块，不允许其它内容）。
- Idea 必须是脱离临时编号仍可理解的算法陈述；旧编号污染只在提示视图中省略。
- 统一记录超父代、超上下文和推进前沿三种响应，并按（代码，算子）统计开发次数。

本轮保持：

- 单次调用同时生成简短 Idea 与完整 Code，不恢复第二次校准或摘要调用。
- 8 个有效根、真实评价预算口径、父代联合分布和 Refine/Pivot/Fuse 请求比例 `0.50/0.15/0.35` 不变。
- 所有有限有效 fitness 的程序入档；退步、持平和重复代码不因结果被删除。
- 不加入成功接力、祖先信用、固定多步 rollout、行为探针、新颖奖励或自动修复。

因此，首轮实验可以主要归因于“生成条件是否更适合具体设计任务”，而不是调度与上下文同时变化。

## 3. 档案候选与两阶段质量抽样

参考只来自本次请求前已经完成真实评价的全档案，并满足：fitness 有限、代码与当前底座不完全相同。Idea、Code 和 fitness 始终来自同一个真实节点，不拼接记录。

去重必须晚于本槽位真正关心的关系判定。Refine 先分别找“直接父节点”和“直接子节点”，子节点再按生成算子是否与本次 Refine 一致分成两池，然后在各自池内对完全相同代码的记录均匀保留一条；因此，无关分支上的同代码记录不会顶掉真实形成边。Pivot 和 Fuse 不依赖直接亲缘关系，才在全体合格候选上按代码去重。

两个档案算子都分两阶段抽样：先按任务语义给质量组分配概率质量，再在组内均匀选一个实现。节点数量再多的 fitness 平台也无法靠数量吞掉质量信号：

\[
P(\text{node}_i)=\frac{P(\text{group}(f_i))}{N_{\text{group}(f_i)}}.
\]

Pivot 的组是现存质量层：候选 fitness 的线性插值 `1/3`、`2/3` 分位数划出 low/middle/high，每个现存层等总概率、层内均分。Fuse 的组是不同 fitness 水平：水平按 fitness 从低到高赋正整数秩，水平质量正比于其秩，水平内均分。Refine 不使用质量层，其日志也不记录质量边界，避免误读为经过质量分层。不再有 AST 签名、结构加成系数与相关日志字段。

## 4. 面向设计任务的证据组织

### 4.1 Refine：一个主要假设与一个有方向的设计实验

上下文包含当前底座和至多一份参考，不为凑满更多程序加入额外材料。

Refine 按固定优先级尝试三池证据，上一池没有可用或可容纳材料时才进入下一池：

1. `formation_evidence`：直接父节点 → 当前底座，回答“当前算法当时是怎样形成的”。边上的算子和 fitness 变化来自当前底座。
2. 同算子 `development_evidence`：当前底座 → 本次同样由 Refine 产生的直接子节点，回答“从这个底座继续 Refine，以前发生过什么”。
3. 异算子 `development_evidence`：由 Pivot 或 Fuse 产生的直接子节点； experiment type 与本次任务不完全匹配，只作为最后备选。

形成边永远先于任何子代获得容量检查，不会被子代池耗尽尝试次数。成功、持平和退步边一视同仁；历史 Fuse 边只声明“另有程序参与但未展示”，不引用未展示 donor 的 fitness（无实现的分数对下一步代码设计不可操作）。

没有可用直接关系或关系池都无法完整放入上下文时，只展示底座，且 Prompt 不再提及 contrast。提示要求围绕一个主要改进假设进行设计，改动为验证该假设所需的最小连贯计算集合；无关部分默认不动。措辞刻意不用 effective/successful/useful：fitness 评价的是整个程序，不能把“父代中已存在的模块”说成“已证明有效的模块”。

这不是单行变更约束，也不把代码距离当作验收标准。它只使相邻候选更接近可解释的设计实验。

### 4.2 Pivot：明确比较对象，但不强制继承

上下文包含当前程序作为 `comparison_baseline`，另提供至多一份 `alternative_reference`。

Pivot 的 `p_task` 让每个当前存在的质量层获得相同总概率，再在层内均分，无放回抽取。因此低、中、高质量候选都保留正概率。提示明确要求相对底座尝试不同的主要决策方法，但不要求继承底座，也不设代码改动比例或强制新颖性门槛。Alternative Reference 只作为一种不同做法的已实现例子：仅当它提示出有竞争力的决策机制时才使用，不是必须复制的模板。archive reference 不配关系说明段：它的角色标题已说明用途，无需声明亲缘有无。

### 4.3 Fuse：只选迁移主参考

Fuse 不再先抽两份覆盖材料、再被动把其中较好者解释为 donor。系统只选择 `transfer_source`：

1. 在全部去重候选上按不同 fitness 值从低到高赋正整数秩；同分候选同属一个水平。
2. 水平质量正比于其秩并归一化，使较强水平获得更高总概率，但任何水平都不被清零；水平内均匀选择，使平台节点数不影响水平总质量。
3. 无放回尝试能与底座完整放入上下文的候选。

因此，donor 选择是纯粹的质量倾向：先选质量水平，再从该水平挑一个实现。它不是互补性的语义估计器，也不保证 donor 强于父代。

默认上下文即底座加迁移源两份程序，不再补第三份跨质量对照：那份材料不是父代、donor、形成证据或已知反例，只是“另一质量区的另一份程序”，覆盖不等于价值，且会稀释本已认知负荷最高的 Fuse 算子的注意力。第三份对照如需研究，应独立做消融。提示把迁移源定位为机制的候选来源：只改编其中看起来与改善底座相关的计算，不为凑齐两份输入而强制拼接；目标是超过较好的输入，但入档不要求超双亲，也不规定两亲代码贡献比例。措辞刻意不预设 donor 拥有已被验证为 useful 的部件：donor 只是按 fitness 水平选出的候选迁移源，这与 Refine 段不用 effective 是同一种认识论约束。

没有可容纳的不同代码 donor 时，执行算子回退为 Refine，父代和已消耗的父代选择次数不变。

## 5. Prompt 语义与历史编号修复

每份完整程序只用固定语义角色小节呈现，没有 `Algorithm N` 编号，也不再按 fitness 排序：

```text
# Design Base
Fitness: ...
Design note: ...
Code:
...
```

固定角色顺序为底座在前、证据在后（Refine：Design Base、Evidence、Observed Transition、Task；Pivot：Comparison Baseline、Alternative Reference、Task；Fuse：Design Base、Transfer Source、Task）。Fitness 只是每个小节的属性，不再形成隐式序列语义。这一次性消除编号引用污染、排序变化导致的 Prompt 抖动与大量正则清理需求。

输入视图把历史 Idea 改称 `Design note`，并在证据区只声明一句：代码是权威实现，设计陈述未经验证。以往已出现 Idea 描述的机制根本未被 Code 实现的情况（如宣称有效实则因 NaN 未生效的计算），模型必须以代码为准，而不是复用高分程序的自我陈述。不恢复第二个摘要器。

只有真实直接边才配 `Observed Transition` 段：

- `Formation Evidence --Refine--> Design Base`，或 `Design Base --<算子>--> Development Evidence`，写出前后 fitness 与差值；
- 历史 Fuse donor 未展示时，只写“这是多输入 Fuse 转移，另有程序参与但未展示”，不引用其 fitness；
- 段末只有一句可操作警告：把 fitness 变化视为整次转移的结果，复用任一改动前先检查代码。

Pivot 的 alternative 与 Fuse 的 transfer source 没有该段：本来就没声称直接生成关系，亲缘声明与因果 caveat 只会把注意力引向系谱与认识论而非算法本身。引用关系仍由 `reference_roles` 与 `context_program_roles` 记录，`evidence_relations` 只保留真实直接边，不再为 archive reference 配非亲缘对象。

输出协议是严格的两段契约，只返回一段 Idea 加一个代码块，不允许标题、分析、解释或额外代码块（解析器只接受以 `Idea:` 开头后接单个 python 块的响应，“先描述再实现”的措辞已被删除）。Idea 在 100 words 内只描述最终算法本身：说明其主要决策规则与关键计算，不得引用输入程序、提示角色、来源或任何临时展示标签。违反协议的候选仍按原规则评价和入档，但依赖提示局部上下文（临时编号或临时角色名）的 Idea 在未来提示中会被安全省略（`temporary_prompt_reference`），避免 Algorithm N 错指以角色名形式复活。

旧档案只改变提示视图，不覆盖归档内容：

- Idea 依赖提示局部上下文（明确临时算法编号，或 Design Base、Transfer Source 等六个临时提示角色名）时，整段 Idea 省略并记录 `temporary_prompt_reference`；Design Base 代码里命中同一模式的注释同样只从提示视图剥离，记同一原因。归档原文一律不动。
- 历史 Idea 进入提示前先压成单段纯文本：去掉换行与行首 markdown 标记，旧文本里的 `# Output` 等伪小节不再改变 Prompt 结构；内容本身保留在同一行内。
- 参考证据代码的注释 token 在提示视图中全部删除（docstring 保留），Design Base 暂保留原样以便单独消融；含临时编号的注释移除仍单独记录。字符串字面量和其它可执行内容保持不变。
- 原始 Code 与 Idea、真实评价结果均不修改。
- 超过 256 tokens 的 Idea 仍整段省略；完整组合超限时可进一步省略 Idea，但不截断代码，也不增加模型压缩调用。

## 6. 容量与有界选择

默认模型窗口、输出预留和安全余量为 `32768/16384/256`，完整输入上限 16128 tokens。最终使用服务端 chat token 计数。

每个参考槽位最多尝试 32 个无放回候选。容量不足时减少材料数量，不扫描并预先 tokenize 全档案。Refine 的三池按固定优先级依次尝试，形成边永远先获得检查机会；没有直接关系时不产生无意义的容量尝试。每种算子至多使用一份参考证据：Refine 和 Pivot 最多两份完整程序（底座加一份参考），Fuse 最多两份（底座加迁移源）。`max_context_programs` 默认 2，只允许 1 或 2：采样器至多提供一份参考，3 没有不同行为，不提前成为配置维度；未来真有第三证据时再扩展。`max_context_programs=1` 时不选择参考：Refine/Pivot 只看底座，Fuse 回退 Refine。

2026-09-08 对 38 个 `tree_state`、26296 个节点的注释审计：注释普遍存在（47% 节点注释 ≥200 词，最高约 3500 词，常占代码 90% 以上），但最大原始底座块仅 4281 词（约 6246 tokens 量级），远低于 16128 上限，注释决定的资格翻转为 0。因此暂不改变 eligibility 规则（底座注释仍参与容量判定）；审计脚本留 `experiments/traceaad_v10_7/analysis/audit_base_comment_eligibility.py`，窗口变化时重审。

## 7. 选择、评价与预算

父代选择继续使用 V10.6 的质量 ESS 校准、按 node ID 的选择次数修正和父代—算子联合分布。参考曝光不增加父代选择次数。重复代码产生的新节点仍保留独立 node ID，但新增的实现级统计会把相同代码的开发尝试合并观察。

单次生成失败不占真实评价预算；进入正式评价后的失败或超时占预算。标量 fitness 评价整个程序，不被解释为 Idea 中每个模块均有效。V10.7R 不增加通用 NaN 拒绝规则或自动行为判定；疑似无效模块应通过正常 Refine 预算形成受控删除、替换或简化候选。

## 8. 日志与恢复

除原有选择、成本和恢复字段外，每次事件新增：

- `context_program_roles`、`reference_roles`：展示程序和参考槽位的真实角色。
- `evidence_relations`：只记录真实直接边的关系类型、方向、生成算子、fitness 变化与历史 donor 身份；archive reference 不进入该字段（角色标题与 `reference_roles` 已足够，不重复表示同一事实）。
- `reference_attempts[].selection_weight`：Pivot/Fuse 被尝试候选的归一化后任务权重。
- `quality_boundaries`、`reference_layers`：只在 Pivot 记录，因为只有它按低/中/高层分配质量；Fuse 按精确 fitness 水平选材，不记录三分层（需要时离线计算）。
- `context_view_omissions`：因提示局部引用、Idea 长度、注释剥离、设计陈述压平或容量产生的视图省略。
- `code_hash`：候选自身的稳定 SHA-256 实现身份，保留。`parent_code_hash`、`donor_code_hash`、`context_code_hashes` 已删除：可由 `context_node_ids`/`parent_id`/`donor_id` 回到 tree state 离线计算。`context_program_count`、`context_parent_index`、`context_donor_index` 已删除：Base 固定第一份、参考至多一份，无信息增量。
- `context_best_fitness`、`context_improved`、`context_delta`：相对所有完整程序输入中最好者的响应。

原有 `parent_delta` 与 `frontier_delta` 继续分别表示超父代和推进全局前沿。对 Fuse 继续记录超父代与 donor 中较好者的 `both_delta`。这样可以区分追上强参考、产生局部改善和真正推进前沿。

在线实现尝试计数与 Prompt 重复计数已删除：它们不参与父代、算子、证据选择或 Prompt，只是在线缓存统计。`prompt_repeat` 可由 `events.prompt_hash` 离线累加，实现尝试可由 `parent_id + requested_operator + tree_state.code` 离线恢复；“以后可能分析的统计量”不进入在线算法 state。

选定材料、角色、Prompt、RNG 和父代计数前值在请求前持久化；恢复时复用原 Prompt，不重新选择材料或重复正式评价。源码、生成协议或上下文配置变化时拒绝恢复旧检查点。

## 9. 配置与对照

当前实现只有 `task_evidence_v1` 一种上下文机制：按 Refine/Pivot/Fuse 的证据角色组织材料。`--context-policy`、`--donor-topk`、`--traj-gens`、`--history-tokens` 等旧入口已删除；检查点、事件与 manifest 以固定常量记录 `context_policy=task_evidence_v1`。`donor_topk`/`traj_gens`/`history_tokens` 只为满足继承构造器而传入固定值，在机制身份与 `run_config` 中都改记 `inherited_unused`，不再是存活参数（V10.7R 的 trajectory/fits 路径不读取 `history_tokens`）。

方法身份已与旧 V10.7 区分：`METHOD=v107r`、检查点 `version=1073`、运行名 `..._v107r_repN`。1073 相对 1072 是纯减法 cleanup，不改证据选择与 Prompt 语义：删除在线实现尝试/Prompt 重复计数及其恢复链、删除可推导的 context telemetry（程序计数、父代/donor 位置、三处代码哈希）、Fuse 不再计算三分层日志（只 Pivot 记录）、删除 `TRAJECTORY_OUTPUT` 别名并合并 Refine/RefineBare 为单模板加条件句、删除 `evidence_reference` 静默 fallback（缺角色直接 fail-fast）、审计脚本移出生产入口。1072 相对 1071 的变化是删除 AST 结构加成及其日志字段、Fuse 改为 fitness 水平分质量、临时编号与临时角色合并为单一省略原因；旧正式实验的 `v107` 身份保持冻结，两类数据不会混入同一分析口径。

`sampled_trajectory_v1`、`uniform_trajectory_v1`、`ancestor_history` 已正式退役，不再有同源码条件下的对照能力。需要复现冻结批次 `20260907_bounded_formal`（`sampled_trajectory_v1`）或旧对照时，只能使用提交 `ac6f4b9c` 的冻结代码续跑，禁止用新代码恢复旧检查点（指纹校验会直接拒绝）。

## 10. 验证与实验问题

代码验收至少覆盖：三种算子的角色化选材且参考至多一份、缺角色直接 fail-fast、Refine 三池优先级与同算子优先、无关系时不补随机档案且不提 contrast、Pivot 层等分层内均匀且是唯一记录三分层的算子、Fuse 水平分质量且大平台吞不掉质量信号、机制身份与检查点不含 structure_preference 与在线计数器、关系段只出现在直接边且隐藏未展示 donor 分数、archive reference 不进 `evidence_relations`、角色小节无编号无排序、严格输出契约可解析、依赖提示局部上下文的 Idea 入档但不再展示、设计陈述单段压平防伪造小节、Fuse 不预设 donor 有 useful 部件、Pivot 限定 Alternative Reference 用法、参考注释剥离不碰归档、`max_context_programs=3` 被拒绝、容量退化、单次调用、评价预算和断点恢复。

固定状态实验应分别检查编号移除、Refine 有向设计实验、Fuse 主参考和 Pivot 比较目标是否改变了预期生成行为。完整搜索只跑 `task_evidence_v1`，并报告 250/500/1000 次真实评价下的 best、最终独立测试、生成次数、tokens 与时间；与旧策略的比较只能引用冻结代码产出的历史批次数字，不得在新代码下重跑旧策略。若需要对关系块做因果消融，应使用独立实验构造器或冻结分支，不把退役模式重新塞回生产 CLI。

正式大实验前还有三个待定消融，不进入默认配置：Pivot 是否保留 archive reference（`Pivot(base)` vs `Pivot(base+alternative)` 的固定锚点对比，看灾难尾部与有效新主决策率）；参考注释剥离是否扩展到 Design Base；Fuse 第三份对照是否独立成项。其中 Pivot 与 Base 注释两项是正式搜索前的最后门控：若无明显负效应即停止机制设计，直接跑正式搜索，不再继续优化 Prompt。默认配置保持最小充分证据。

判断是否形成“滚雪球”时，核心观察首次有竞争力突破出现的时间，以及突破后固定预算内兑现的额外收益。树深、节点数和质量层覆盖率不是成功指标。

成功子代的一次性优先接力属于下一项独立实验。只有生成端修复先证明能更稳定地产生可用响应后，才考虑加入；本设计不为祖先回传信用，也不锁定固定多步开发。
