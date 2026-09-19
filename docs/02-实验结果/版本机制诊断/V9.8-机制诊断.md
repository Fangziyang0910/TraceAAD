# TraceAAD-V9.8机制诊断

## TraceAAD V9.8 机制识别与提议核异质性实验分析

### 1. 核心判断

V9.8 用固定锚点配对实验看 Refine/Explore 与来时路的单步行为，并用强制续段看 Explore 子代短短几步能否继续变好。

1. **两种提示的改写行为可以分开。** 在 code-only 与 parent-path 两种上下文下，Explore 的离线静态机制标签切换率（37%–54% vs 2%–12%）与代码修改比例显著高于 Refine，即时有向质量显著更低（36%–87% 出生即退步）。
2. **来时路对 Refine 更稳。** 四任务中 `parent_path - code_only` 的锚点级配对即时质量均值全部为正（72 个锚点中 58 好、14 差）。
3. **来时路对 Explore 方向不稳定。** CVRP 7正/7负，OBP 7正/9负/2平。
4. **短短几步续改随任务而变。** 五步强制续段里，OBP 与 OP 部分能回到来源父代（恢复率 38%–61%），CVRP 几乎不能（1/15）。完整搜索里的边界宽限把额外机会给了即时质量低的 Explore 子代；CVRP 三档同向退化 2.2%–4.4%。联合版本不能归因到单个宽限项。

### 2. 实验协议与数据事实

#### 2.1 P1/P2：History × Intent 受控配对设计

- 抽自 V9.7 冻结事实层，四任务按有向质量分成 low / middle / high 三层，每层 6 个代码互异锚点；
- $2\times2$ 配对设计（History：code-only / parent-path × Intent：Refine / Explore），同一 `anchor × replicate` block 共享采样 seed；
- 72 个固定锚点，每个锚点 3 次独立重复、4 个条件，共 864 个原始调用；
- 746 个有效响应，850 次真实 evaluator 调用；主统计先在同一锚点内平均重复观测，再以源锚点为独立单位。

#### 2.2 P3：Explore child 强制续段

- 179 个有效且非 no-op 的 `parent_path × Explore` 观测中，按预注册规则为每个源锚点选择第一个可用 child；
- 69 个源锚点进入 P3：CVRP 15 个，其余三任务各 18 个；
- 每个 child 配对运行 child-chain 与 hypothesis-level 两种协议，每种协议五步，共 690 个响应；
- 138 条 continuation 全部包含恰好 step 1–5，690 个唯一 `(continuation_id, step)` 与 690 个原始调用一一对应；
- 545 个有效响应，145 个无效响应，659 次真实 evaluator 调用。

两阶段都采用同进程逐响应流水：一次生成落盘后立即解析和评价，再进入下一次生成。上述完成数来自冻结后的事实层验收，不来自控制台近似计数。

### 3. P1：Operator 是否改变单步生成

下表的即时差为锚点内配对的 `Explore - Refine` 有向质量变化。正 / 负表示锚点级配对均值的方向；不同任务的 $q$ 尺度不同，不能横向比较差值幅度。宏簇切换率和修改比例展示 parent-path 条件下的锚点级均值。

| 任务 | Code-only 即时差，正/负 | Parent-path 即时差，正/负 | Parent-path 宏簇切换率 R / E | Parent-path 修改比例 R / E |
| --- | ---: | ---: | ---: | ---: |
| TSP | -1.527，4/12 | -1.345，4/12 | 0.222 / 0.435 | 0.736 / 0.917 |
| CVRP | -4.171，2/12 | -7.063，1/13 | 0.204 / 0.583 | 0.375 / 0.814 |
| OP | -1.238，3/15 | -1.374，1/17 | 0.185 / 0.685 | 0.652 / 0.843 |
| OBP | -804.083，2/16 | -749.567，1/17 | 0.148 / 0.444 | 0.476 / 0.801 |

Explore 在所有任务和两种上下文下都更常产生即时退步，同时在 parent-path 条件下都更常切换静态机制代理并形成更大代码变化。Code-only 条件也保持相同总体结构。由此可确认 operator 指令改变了生成分布；不能把 Explore 的更高切换率解释为真实 family discovery，也不能因即时质量更低判定其后续价值为零。

### 4. P2：Parent path 分别怎样作用于两种 Operator

下表报告锚点级配对的 `parent_path - code_only` 即时有向质量，形式为均值 ± 样本标准差；方向计数为正 / 负 / 平。有效率是三次重复先在锚点内求比例后的均值。

| 任务 | Refine 配对差，方向 | Refine 有效率 C / H | Explore 配对差，方向 | Explore 有效率 C / H |
| --- | ---: | ---: | ---: | ---: |
| TSP | +0.574 ± 1.047，9/5/2 | 0.870 / 0.852 | +0.756 ± 4.567，11/5/0 | 0.870 / 0.852 |
| CVRP | +3.080 ± 4.241，12/2/0 | 0.667 / 0.778 | +0.188 ± 3.898，7/7/0 | 0.574 / 0.556 |
| OP | +0.515 ± 0.686，13/5/0 | 0.981 / 0.981 | +0.379 ± 1.980，12/6/0 | 0.963 / 0.944 |
| OBP | +36.590 ± 103.804，14/3/1 | 1.000 / 0.981 | +91.106 ± 648.913，7/9/2 | 0.963 / 0.981 |

其中 C 表示 code-only，H 表示 parent-path。Refine 的四任务均值与方向计数一致支持 parent path；CVRP 的有效率也明显增加。Explore 的条件均值虽都为正，但 CVRP 与 OBP 的方向计数不稳定，标准差远大于均值，有效率也有升有降。因此，第一版完整 V9.8 可以把 parent path 保留为共同 prompt baseline，但关于 Explore 的独立科学主张应保持未验证；后续可以单独比较 Explore 的局部来时路、机制摘要与全局已探索区域提示。

### 5. P3：短续段能否发展或救回 Explore Child

下表给出 H5 时相对 Explore 入口的平均 internal gain，以及相对原父代的 recovery rate 从 H0 到 H5 的变化。H0 已包含 Explore child 自身，所以其 recovery rate 可以非零。

| 任务 | $n$ | Child-chain H5 internal gain | Child-chain recovery H0 → H5 | Hypothesis-level H5 internal gain | Hypothesis-level recovery H0 → H5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TSP | 18 | 0.884 | 0.389 → 0.444 | 0.767 | 0.389 → 0.389 |
| CVRP | 15 | 1.543 | 0.067 → 0.067 | 1.355 | 0.067 → 0.067 |
| OP | 18 | 0.587 | 0.222 → 0.389 | 0.372 | 0.222 → 0.278 |
| OBP | 18 | 585.347 | 0.111 → 0.611 | 584.722 | 0.111 → 0.500 |

到 H5 时，child-chain 观察到 internal gain 的 child 数为 TSP 10/18、CVRP 10/15、OP 14/18、OBP 14/18；hypothesis-level 分别为 8/18、6/15、16/18、15/18。发展发生不等于越过原父代：CVRP 两协议都只有 1/15 最终 recovery，而 OBP 分别达到 11/18 与 9/18。

Hypothesis-level 减去 child-chain 的 H5 internal gain 配对均值为：TSP -0.117 ± 0.843、CVRP -0.188 ± 1.946、OP -0.215 ± 0.937、OBP -0.625 ± 6.324。H1/H3 的部分任务方向相反，因此不能形成“区域重选优于单链”或相反的统一结论。P3 支持短续段机会具有任务相关价值。

### 6. 回到原子搜索循环的解释

V9.8 的基本操作仍然是“选一个锚点和 operator，生成一个 child，评价后更新状态”。Stage P 没有评价整个调度器，它只识别这个原子循环中的三个问题。

#### 6.1 Operator 决定要采样的 transition 类型

记给定锚点、上下文和 operator 时的生成分布为：
$$
K_o(x'\mid a,h)=P(x'\mid a,h,o).
$$

P1 表明 $K_R$ 与 $K_E$ 在即时质量、静态机制代理切换和修改规模上有一致的可观测差异。这是后续区分 development 与 departure 的必要条件：若 operator 不改变 proposal behavior，则 operator-specific 上下文与分配都没有可利用的差异。当前数据不识别分布全貌，也不识别 Explore 是否进入更好的真实算法族。

#### 6.2 历史不具有脱离 Operator 的统一价值

P2 识别的是 interaction：parent path 对 Refine 的即时质量呈跨任务正向，对 Explore 则不稳定。同时观察到 Refine 修改更集中与质量更高，但本实验没有对“理解当前方向 → 修改集中 → 质量改善”做中介分析，因此该链条只是候选解释。对 Explore 而言，parent path 也可能同时提供问题信息并形成 trajectory anchoring。这同样尚未识别。

因此更基本的设计对象是 operator-conditioned context policy：
$$
h_t=\operatorname{Ctx}(a_t,o_t,\mathcal H_t).
$$

第一版仍可使用共享 parent path 作为可比 baseline；是否应让 Explore 读取更弱的局部路径、机制摘要或已探索区域，需要新的单因素实验。

#### 6.3 即时质量不能直接等同于有限续段价值

记对入口 child $x'$ 强制运行 $H$ 步的协议为 $\pi$，其有限续段价值为：
$$
V_H^{\pi}(x')=\max_{0\leq k\leq H}q(x_k)-q(\operatorname{parent}(x')).
$$

P3 观察到一部分 $q(x')<q(\operatorname{parent}(x'))$ 的 child 在五步内恢复，因此将跨边界的低即时质量直接映射为零后续机会会产生 false negative。但 $V_H^{\pi}$ 同时依赖任务、horizon、proposal kernel 和续段协议，它不是 child 的内在不变潜力。CVRP 的低 parent recovery 只表明该样本与五步协议下的固定投入可能低效，不足以把全部内部发展定义为浪费。

这导向一个分配原则：**development opportunity 不等于 guaranteed development budget**。Stage P 说明预算分配不应把即时回撤当作必然死刑，但没有告诉系统应为哪个 child 支付多少观测成本。这一问题仍必须由每步重新选择的 allocation policy 回答。

#### 6.4 Hypothesis 表示与局部 Development Policy 必须分开识别

Hypothesis 至少包含三个可分问题：Explore boundary 如何划分 trajectory segment，哪些统计在 segment 内聚合，以及选中 segment 后如何选锚点。P3 只比较了第三个问题的两种强制续段，因此 hypothesis-level 未胜出不能推出 hypothesis abstraction 无价值。

H1/H3 与 H5 方向的变化可以产生一个新的探索性假设：早期在 segment 内回访多个锚点，出现清晰领先谱系后转向 child-chain commitment。当前样本量、嵌套 horizon 和协议差异都不支持将它写成已观察 dynamics；它只是后续 adaptive local policy 的预注册候选。

### 7. 对 V9.8 的证据分层

#### 7.1 已有直接行为证据

- Refine / Explore 的角色划分可以保留。两种指令稳定改变了静态机制代理切换、修改规模和即时质量；这支持两种 proposal role，不识别真实 algorithm family。
- Parent path 可以作为 Refine 的默认 development context。对 Explore 则只能作为当前可比 baseline，其价值需要在 operator-specific context 实验中单独识别。

#### 7.2 只支持问题动机

- Explore 更常触发静态机制代理切换，因而将其作为新 trajectory segment 的操作边界有行为依据；该边界是否比 root route 更适合聚合分配状态，Stage P 没有回答。
- P3 证明“跨边界的低即时质量不能一律当作零续段价值”，因而支持延迟容忍问题的存在。它不支持 $c_R$ 的差额形式、平方根衰减或任何固定续段深度。简言之，P3 支持 $C$ 要解决的问题，不支持 $C$ 这个解法。
- 五步强制续段在 OBP 观察到的 parent recovery 增量远高于 CVRP，因此不恢复 protected probe。V9.8 用有限、衰减且每步重新竞争的宽限作为更小的待验证回答。

#### 7.3 尚无对应证据

- $M$ 仍是启发式设计。P3 中“child 能继续改善”不等于“历史平均 gain 能预测下一份计算的价值”。
- Hypothesis 聚合是否优于 route 聚合，仍需要 Route-$Q+U$ 与 Hypothesis-$Q+U$ 对照。Hypothesis-level 局部重选未胜出只限于锚点 policy，不能外推到 boundary 或聚合统计。
- 固定 `0.7/0.3` 只是控制条件，Stage P 没有识别 operator 频率。

#### 7.4 任务差异应拆成两个可测属性

Stage P 显示，“任务是否适合 trajectory”过于粗糙。至少需要分开：

1. **跨边界提议几何**：Explore 改变机制代理、修改规模和入口质量的联合分布；
2. **有限视野可恢复性**：在给定 horizon 和续段 policy 下，初始退步的新段能否恢复到来源父代。

P3 中 internal gain 与 parent recovery 的分离说明，“新方向内部还能改”不等于“它能在有限预算内追上已有方向”。这两个属性共同决定跨边界宽限的可能收益，但当前只有固定锚点与五步干预下的局部估计，不能升格为任务的不变属性。

### 8. 证据层级与后续识别

- 本分析属于 **mechanism identification / behavioral validation**：识别指令、上下文和强制续段是否改变所测行为。Stage A 才是 **search-policy effectiveness / causal ablation**；1000-eval 完整批次是 **end-to-end search performance**，held-out 则评价最终程序的泛化。这四种证据不互相替代。
- P1/P2 是固定锚点重复观测，P3 是条件于有效、非 no-op 的 parent-path Explore child 的强制干预；两者都不是独立完整搜索重复。P3 recovery rate 只适用于该选入 cohort 与强制协议，不是全部 Explore attempt 的无条件概率。
- 静态宏簇是代码规则代理，不是真实算法 family；更高切换率只说明 proposal 的机制代理分布改变。
- P3 的 horizon 是同一生成前缀的嵌套读取，H1、H3、H5 不是独立样本。
- 下一层证据来自 Hypothesis-Uniform、Route-$Q+U$、Hypothesis-$Q+U$、$Q+U+C$、$Q+U+C+M$ 的 Stage A 对照；完整正式 V9.8 只能评价联合协议。
- “区域内早期回访，出现领先谱系后承诺”只是由 H1/H3/H5 方向变化产生的探索性假设。若继续检验，必须预注册切换信号、切换时点和总 horizon，不能在当前结果上事后择时。

### 9. 事实工件

- P1/P2 分析：`experiments/generation_probe/20260815_221500_v98_p1_p2/analysis/summary.json`；
- P3 分析：`experiments/generation_probe/20260816_001100_v98_p3/analysis/summary.json`；
- 正式 V9.8 批次：`20260815_225000`（四任务 × 三重复）。

## TraceAAD V9.8 完整版本复盘与机制有效性分析

正式搜索为四个任务各三次、每次 1000 次真实评价；held-out 工件为各任务 `eval_best_20260817_v98_complete/results.json`。比较对象是 V9.7 与完整 V9.8 `q_u_c_m`。

### 1. 核心判断

V9.8 针对了 V9.7 中已经被证据定位的两个接口问题，但没有把这些设计回答转化为跨任务的最终性能优势。

第一，Refine / Explore 已经不是名称上的两种算子：固定锚点实验显示它们产生了可区分的 proposal 分布，Refine 更偏向即时质量和集中修改，Explore 更偏向大修改与静态机制代理切换。第二，V9.8 的 hypothesis、边界宽限和历史发展项确实进入了在线状态，并改变了预算路由的几何形态。然而，V9.8 的最终结果相对 V9.7 是任务依赖的：OP 三个规模同向改善，CVRP 三个规模同向退化，TSP 与 OBP 没有统一改善。跨任务描述性平均名次从 V9.7 的 3.367 变为 V9.8 的 4.333。

因此当前最准确的结论是：

- **生成意图机制已经 work 到“改变 proposal behavior”这一层。**
- **parent path 对 Refine 的作用已经 work 到“固定锚点单步行为”这一层；对 Explore 不能外推。**
- **hypothesis、$C$、$M$ 和联合 `q_u_c_m` 机制确实运行，但尚未证明改善有限预算搜索。**
- **完整 V9.8 作为统一版本升级没有 work。** OP 的正向结果只能作为任务内联合结果，不能归因于某个单项机制。

### 2. 设计意图：V9.8 想从 V9.7 改进什么

V9.7 的原子循环本身没有被推翻：选择一个锚点，提供该锚点的父代来时路，生成一个候选，真实评价，再更新状态。问题在于循环前的两层分配和生成条件没有对齐搜索几何。

| V9.7 暴露的问题 | V9.7 的具体证据 | V9.8 的设计回答 | 当前证据能支持到哪里 |
| --- | --- | --- | --- |
| root route 只是共同来源，不是稳定算法簇 | 12 次运行中有 9 次最终程序改变所在 root 的静态宏簇；96 条路线有 64 条 bootstrap 后没有正式后续段 | 用有效 Explore child 开启新的 hypothesis，Refine 在 hypothesis 内继承 | hypothesis 边界确实被大量创建；它是否比 route 更适合作为投资单位，尚未识别 |
| 质量加一次尺度的分配会使路线或 child 早夭 | Explore child 出生即淘汰率为 36%–87%；当前路线分数存在不可恢复区 | 用来源基线加衰减的边界宽限 $c_R$，每次响应后重新竞争 | Stage P 证明延迟发展问题存在；不支持 $c_R$ 的具体差额形式或平方根衰减 |
| Refine 与 Explore 只是“大改/小改”的描述 | Explore 的静态宏簇切换率为 37%–54%，Refine 为 2%–12%；12 个终局程序全部由 Refine 生成 | 在 prompt 与状态更新中明确“发展当前方向”与“提出替代方向” | P1 直接支持两种 proposal role；不等于 Explore 发现真实新 family |
| 已投预算和历史发展没有进入 hypothesis 级路由 | V9.7 路线分数只读取当前前沿和累计访问数 | 在 hypothesis 分数中加入 operator-specific $U$、边界宽限 $C$、历史平均发展收益 $M$ | $C/M$ 在正式事件中被计算并多次非零；无固定生成协议的 Stage A，不能判断其净价值 |
| 宏观 scheduler 容易把实现假设误写成科学机制 | 上一版草案包含 protected probe、ticket、block 和固定通道 | 收缩为“一次选择—一次生成—一次评价—一次更新” | 原子循环和日志契约清楚，在线没有未来预算承诺；这是设计质量改进，不是性能证据 |

V9.8 的核心变化可以写成：

$$
\text{route} \rightarrow \text{hypothesis segment},
\qquad
q^*+U \rightarrow q^*+U+C+M,
$$

同时保留固定的 `0.7/0.3` Refine / Explore prior、最多 8 条父代来时路和单步原子循环。这里的 hypothesis 只是 Explore 开启的 trajectory segment，不是真实算法 family，也不自动拥有固定预算。

### 3. 证据分层与实验完整性

机制按四层证据判断，上一层不能替代下一层：

1. **机制规范**：代码和设计稿规定系统怎样运行。
2. **过程激活**：日志证明组件实际计算、创建状态或改变访问分布。
3. **行为/搜索结果**：固定锚点或完整搜索显示 proposal、best-at-budget 是否改变。
4. **最终质量**：独立 held-out 是否在三次运行上同向改善。

正式 V9.8 为四任务三重复，每次 1000 次真实评价加 held-out。Stage P 的 864 个 History × Intent 响应和 690 个 Explore child 强制续段响应是机制识别工件，不是完整搜索重复。预注册的 Hypothesis-Uniform、Route-$Q+U$、Hypothesis-$Q+U$、$Q+U+C$、$Q+U+C+M$ 逐项 Stage A 消融未运行，$C$、$M$ 与 hypothesis 聚合没有独立因果功劳可分配。

### 4. 正式 V9.8 的运行行为

#### 4.1 Hypothesis 边界确实运行，但产生了很多短段

每次完整运行从 8 个 root 开始，正式搜索中新增约 227–303 个 hypothesis；最终 `n_hypotheses` 为 235–311。这个数量说明 Explore-defined boundary 并非名义字段，而是把大量有效 Explore child 作为新的轨迹段写入在线状态。

但新段是否继续获得计算高度依赖任务：

| 任务 | 最终 hypothesis 数 | 被选中的不同 hypothesis 数 | 单个 hypothesis 最大选择份额 | 新 hypothesis 后续获得至少一次 Refine 的比例 |
| --- | ---: | ---: | ---: | ---: |
| TSP | 235–240 | 219–228 | 4.0%–7.1% | 91%–97% |
| CVRP | 244–266 | 232–257 | 3.6%–15.5% | 96%–98% |
| OP | 254–297 | 232–271 | 4.4%–8.0% | 86%–91% |
| OBP | 277–311 | 20–115 | 13.2%–90.4% | 0%–27% |

这组数据同时包含两个方向。对 TSP、CVRP 和 OP，hypothesis 分配比 V9.7 的 route 层更分散，且大多数新段至少得到过一次 Refine；对 OBP，hypothesis 数量很多，但实际被选中的对象很少，rep3 有 290 个新 hypothesis 且没有一个得到后续 Refine，选择份额最高的一个 hypothesis 占 90.4%。因此“创建了 hypothesis”不能等同于“获得了可比较的探索机会”。

#### 4.2 $C$ 与 $M$ 被激活，但没有反事实证明它们改变了正确选择

从 `events.jsonl` 的每次搜索响应读取选择时的分数组件：

- $C$ 在 TSP、CVRP、OP 的约四成选择中非零；OBP 为 0%–39%，不同重复差异很大。
- $M$ 在 TSP、CVRP、OP 的约三至四成选择中非零；OBP 三次约为 62%–100%。

日志保存了被选 hypothesis 的 $q,U,C,M$，但没有在同一状态下计算去掉某一项时会选择谁的反事实，也没有与固定 prompt、同一 operator seed schedule 的无 $C$ 或无 $M$ 运行配对，非零次数推不出 $C$ 或 $M$ 提高了 best-at-budget。

OBP 的行为尤其值得警惕：$M$ 的激活很高，但 hypothesis 选择仍可能塌缩到一个对象；这与“历史平均 realized gain 能预测下一份计算价值”的设计假设并不一致，至少说明该信号在该任务上没有自动带来稳定的覆盖。

#### 4.3 最终全局最好仍主要由 Refine 产生

12 次正式运行的最终 best 程序中，11 次由 Refine 生成，1 次由 Explore 生成（CVRP rep1）。这与 V9.7 的 12/12 Refine 终局事实相近。它支持以下运行图景：Explore 负责低命中、结构迁移的提议，Refine 负责把少数可用方向发展到最终前沿；但它不证明 V9.8 的 hypothesis 或宽限使这种发展更有效。

Stage P（第一部分）与正式运行一致：Explore 更常退步、更大修改、更多静态宏簇切换，parent path 对 Refine 为正、对 Explore 不稳定，五步续段的 parent recovery 存在且任务异质。

因此，V9.8 的“生成角色”有行为依据；“给所有新段一份统一边界宽限”没有同等强度的依据。

### 5. Held-out 结果：完整 V9.8 是否改善了 V9.7

下表报告 V9.8 相对 V9.7 的三次运行均值差。最小化任务中正值表示箱数或路径长度变差，OP 最大化任务中正值表示收益变好。差值是 `(V9.8 - V9.7) / V9.7`，仅作描述性比较，不是显著性检验。各任务页的 `±` 是三次运行的样本标准差，不是置信区间。

| 任务 | V9.8 三次均值 | 相对 V9.7 的方向与幅度 | 结论 |
| --- | --- | --- | --- |
| TSP50 / 100 / 200 | `6.197989 / 8.526220 / 11.949251` | `+3.46% / +1.63% / -0.63%` | 两档退化，一档小幅改善；无统一方向 |
| CVRP50 / 100 / 200 | `9.220167 / 15.748477 / 28.461341` | `+2.21% / +3.88% / +4.39%` | 三个规模同向退化，幅度明显 |
| OP50 / 100 / 200 | `15.070311 / 30.308873 / 53.801198` | `+0.45% / +1.35% / +2.59%` | 三个规模同向改善，但仍是联合版本事实 |
| OBP 六档 | `413.800 / 2022.467 / 4032.533 / 81.000 / 402.333 / 804.067` | `+0.19% / +0.16% / +0.12% / -0.25% / -0.02% / -0.02%` | 规模与容量方向混合，整体接近持平 |

跨任务代表性同场的描述性平均名次为：V9.7 `3.367`，V9.8 `4.333`。单独只放一个 TraceAAD 版本时，V9.7 为 `2.200`，V9.8 为 `2.667`。V9.8 仍有竞争力，但它没有达到“V9.7 的统一升级”这一设计目标。

性能结果应按任务解释：

- **CVRP** 是最强的负面信号。V9.8 的新段和边界机制运行充分，但三个规模都比 V9.7 差，说明“更多 hypothesis / 更宽的跨边界容忍”没有兑现为有效的 CVRP 前沿推进。
- **OP** 是正面信号。三档均值同向改善，收益属于 proposal 与 allocation 的联合变更。
- **TSP** 说明跨规模泛化不能从单档改善推断。200 规模略好，但 50、100 变差。
- **OBP** 的差异处于约千分之几到四分之一个百分点，且方向混合；这更接近“机制没有形成可见净收益”，而不是明确成功或明确伤害。

### 6. 机制逐项判定

这里的 `work` 指证据达到的层次，不把“运行”与“改善搜索”混为一谈。

| 机制 | 证据状态 | 当前判定 | 原因与边界 |
| --- | --- | --- | --- |
| Refine / Explore 语义分工 | 固定锚点 P1；正式运行中的最终谱系 | **work：proposal behavior** | 两种指令稳定改变即时质量、修改规模和静态机制切换；不证明 Explore 发现真实新 family，也不证明 30% 比例最优 |
| Parent path | P2 History × Intent | **Refine 上 work；Explore 未验证** | Refine 四任务配对差为正；Explore 方向和有效率不稳定，不能共享同一强主张 |
| Hypothesis boundary | 正式日志创建 227–303 个新段，改变选择覆盖 | **运行并改变状态；收益未识别** | hypothesis 不是 route 的简单重命名，但它是否更接近可利用算法簇、是否提高 best-at-budget，缺少 Route/Hypothesis 消融 |
| 边界宽限 $C$ | 事件中多次非零，任务间激活差异大 | **运行但未证明有效** | P3 只证明延迟价值问题存在，不支持当前差额与衰减形式；OBP 仍出现严重选择塌缩 |
| 历史发展项 $M$ | 事件中多次非零，OBP 激活最高 | **运行但未证明有效，可能过度偏置** | $M$ 是 realized gain，不是未来边际价值；没有 `w/o M` 对照，不能判定是无效还是有害 |
| Operator-specific $U$ | 每步按 operator 计数并重选 | **实现完成，效果未识别** | 只是欠观察启发式；没有与相同 proposal 的无 $U$ 对照 |
| 每步重新选择、无固定 probe/ticket | 事件和 checkpoint 满足原子循环 | **设计上 work** | 删除了未来预算承诺，因果链更可审查；但它也可能让需要连续发展的 child 仍得不到足够视野 |
| 固定 `0.7/0.3` prior | 响应份额接近预设，任务间实际探索行为不同 | **控制变量，不是已验证机制** | 该比例未由 Stage P 或正式结果识别；任务几何不同，统一 prior 的合理性仍是待验证假设 |
| 完整 `q_u_c_m` 联合协议 | 四任务三重复 held-out | **作为通用升级未 work** | CVRP 明显退化、TSP/OBP 无统一改善、OP 改善 |

### 7. 结合研究认识得到的解释

#### 7.1 生成侧已经制造了可利用的异质性，但异质性不是自动价值

研究认识要求先确认 proposal 是否存在可分配的差异。V9.8 的 P1 已经满足了这一前提：Explore 更常切换机制代理，Refine 更常保持当前方向并产生即时改善。这说明分配不是在完全同质的响应上工作。

但异质性有两个维度：跨边界提议几何和有限视野可恢复性。P3 显示 OBP 的 child recovery 很高，CVRP 很低；同一个“Explore 后退步”在两个任务上不具有相同的后续价值。V9.8 把这两个维度压到同一条通用 $C$ 公式中，因而没有理由预期所有任务同向受益。

#### 7.2 路线不是 family，hypothesis 也还不是 family

V9.7 的 9/12 根到终局宏簇变化说明 route 只代表 provenance。V9.8 用 Explore boundary 改善了状态表示的语义方向，但“Explore 发生了”仍只是生成事件，不保证新段进入不同且更好的算法盆地。正式运行中新 hypothesis 数量达到数百，且没有全局 hypothesis bank；代码不同可能只是既有机制区域的重访。

因此当前应把 hypothesis 视为更细的轨迹分段，而不是已经验证的投资臂。它的科学价值要靠覆盖后的有限预算 ceiling、行为簇或语义簇证据来确认。

#### 7.3 分配信号描述过去，不等于预测未来

$C$ 使用来源基线与当前前沿的差，$M$ 使用历史平均前沿增益。二者都是已经发生的事实：一个描述回撤，一个描述已实现发展。它们没有直接估计下一份计算的边际价值 $V_H$。

这解释了正式结果中的两个现象：一方面，新 hypothesis 在 TSP/CVRP/OP 多数能得到一次后续 Refine，但这不保证发展到更好前沿；另一方面，OBP 的 $M$ 高激活没有阻止部分运行的 hypothesis 选择塌缩。把 hindsight success 直接写进在线信用，可能只是在奖励已经被选择过的容易区域。

#### 7.4 轨迹信息的价值是 operator-conditioned 的

parent path 对 Refine 的跨任务正向结果支持“来时路帮助当前方向发展”。Explore 的不稳定结果说明它需要的可能是更弱的局部路径、机制摘要或全局已探索区域提示，而不是把 Refine 的上下文原样复用。V9.8 共享 parent path 是合理的可比 baseline。

#### 7.5 泛化来自读出任务几何

四个任务给出不同结果：CVRP 退化、OP 改善、TSP 混合、OBP 接近持平。这与研究认识中“簇稀有度、族内可改进性和簇间质量方差不同”的方向一致，但还不能把 V9.8 的差异写成已验证的泛化机制。当前只能说：固定 `0.7/0.3` 与统一 $C/M$ 在不同任务上产生了不同的搜索行为，任务特性如何被在线读出仍未解决。

### 8. V9.8 的优点与缺点

#### 优点

1. **问题定位准确。** V9.8 直接针对 V9.7 的 route 语义错位、Explore child 早夭和意图混合，不是继续调一个乐观项系数。
2. **机制边界收缩得更好。** 删除 protected probe、ticket、block 和固定 lane 后，在线过程恢复为单步闭环；hypothesis 不再自动拥有预算权利。
3. **生成与分配被明确拆开。** P1/P2/P3 先识别 proposal，再让正式搜索回答联合路由问题，符合“实验上拆开、理论上耦合”的研究认识。
4. **过程证据可审查。** 正式事件保存 hypothesis、$q/U/C/M$、锚点、operator、父链和响应 ID；Stage P 也有逐响应落盘和配对结构。
5. **识别出了真正的任务异质性。** OBP 的短期 recovery 和 hypothesis 选择集中与 CVRP 的低 recovery、较强退化形成对照，说明统一机制不能直接当成跨任务规律。

#### 缺点

1. **假设堆叠超过了消融进度。** hypothesis boundary、$C$、$M$、operator-specific $U$ 同时进入正式版本，最终差异无法拆分。
2. **hypothesis 产生过快，且没有全局新颖性约束。** 数百个新段被创建，但 OBP 中大量段没有后续 Refine；这增加了状态复杂度，却没有保证有效覆盖。
3. **$C$ 的先验没有被任务几何校准。** P3 只证明“低即时质量不等于零延迟价值”，没有证明来源基线差额和平方根衰减是正确尺度。
4. **$M$ 可能把已实现收益误当作未来价值。** 它受选择性预算影响，容易偏向已经被投入且本来容易改进的 hypothesis。
5. **Explore 的上下文仍未专门设计。** parent path 对 Refine 有证据，对 Explore 不稳定；共享 prompt 可能一边提供问题信息，一边把替代方向锚定在原路径。
6. **固定 operator prior 没有任务适配。** 四个任务的跨边界质量分布和恢复性明显不同，但 `0.7/0.3` 不读取这些差异。
7. **最终性能没有达到升级目标。** V9.8 的完整联合协议在当前四任务、1000-eval、三重复口径下没有超过 V9.7 的综合表现，CVRP 还出现三档同向退化。

### 9. 经验教训与下一步

#### 9.1 先证明“可利用异质性”，再设计分配公式

P1 已经证明 Refine/Explore 提议分布不同，但还需要分别测量每个任务的跨边界质量尾部、恢复性和有限预算 ceiling。不能因为 Explore 能换簇，就默认新 hypothesis 值得得到统一宽限。

#### 9.2 每个新分配信号都必须有单因素反事实

下一轮应按已经写入 V9.8 协议的阶梯完成：

1. `Single` 与 `Hypothesis-Uniform`，确认维护多个起点是否有 pool value；
2. `Route-Q+U` 与 `Hypothesis-Q+U`，确认动态分段是否优于 root provenance；
3. 加入 $C$ 的单因素对照，判断边界宽限是否改善 best-at-budget；
4. 在固定 $C$ 后加入 $M$，判断历史 realized gain 是否有额外路由价值；
5. 最后才与 V9.7 做联合比较，并把 held-out 作为整体系统结果。

每一臂固定 root、prompt、operator seed schedule、evaluator 和 1000 次预算，三次独立重复。没有这些对照时，$C$ 与 $M$ 应继续被称为待验证先验。

#### 9.3 把“发展机会”与“保证预算”分开

强制五步续段适合测量 $V_H$，不应直接变成在线承诺。在线版本可以继续逐步重选，但需要记录新 hypothesis 的首次 Refine 等待时间、获得的后续步数和有限视野 recovery；否则无法判断是 proposal 价值低，还是 allocation 没有给它观察机会。

#### 9.4 Explore 需要 operator-conditioned context 实验

至少应在固定锚点上比较：

- Refine：parent path；
- Explore：完整 parent path、局部路径、机制摘要、已探索区域摘要。

该实验必须固定同一锚点、采样 seed 和输出契约，避免把上下文差异与锚点选择差异混在一起。

#### 9.5 控制 hypothesis 数量与语义重访成本

在没有可靠语义簇识别前，不应加入昂贵的在线 judge 或 embedding 作为默认机制；但应离线报告新 hypothesis 对既有静态/行为 proxy region 的重访比例。若同一任务持续出现“很多 hypothesis、很少被继续”，应先修正边界粒度或覆盖协议，再继续增加信用项。

#### 9.6 把任务异质性写成可检验属性

下一版报告至少应分开：

- **跨边界提议几何**：Explore 的修改规模、机制切换率、即时质量分布与正尾；
- **有限视野可恢复性**：在预先声明的 $H$ 和 continuation policy 下回到父代或超过父代的概率；
- **路线/段的可利用差异**：在统一覆盖后的 finite-budget ceiling 差异。

只有这些属性能够预测分配收益时，才能把“在线读出任务特性”从研究认识中的待验证主张提升为机制结论。

### 10. 结论

V9.8 是一次有价值的机制识别版本，但不是已经证明有效的完整方法版本。它最重要的成果是把研究问题收敛到可检验的接口：Refine/Explore 确实产生不同的 proposal，parent path 的价值依赖 operator，跨边界延迟发展依赖任务，hypothesis 分配会改变覆盖但可能产生状态膨胀。

当前可以写入研究认识的结论是：**轨迹条件生成已经显示出可利用的角色差异；轨迹感知分配的价值取决于任务特定的跨边界恢复性与段间异质性，不能由统一的 $q+U+C+M$ 公式预先保证。**

当前不能写入的结论是：hypothesis 一定优于 route、边界宽限一定有效、历史平均发展收益能够预测未来边际价值，或 V9.8 的 OP 改善由其中某一机制单独造成。
