# TraceAAD V9.19 完整机制设计（历史版本）

V9.19 将自动算法设计建模为**行为落地形成轨迹引导的搜索**。候选的求解轨迹提供执行行为坐标；候选的形成轨迹保存算法改进的来时路。每一条有效形成边再由实际求解行为落地，记录这次改写是否提高质量，以及它相对既有档案到达了多新的行为位置。

$$
\boxed{
\text{Behavior Landscape}
+
\text{Behavior-Grounded Formation Trajectory}
+
\text{Trajectory-Conditioned Generation}
}
$$

在线控制器只保留三个概念状态：

$$
\boxed{
P_t(a)\;\text{(Promise)},
\qquad
U_t(a)\;\text{(Opportunity)},
\qquad
T(a)\;\text{(Trajectory Response)}
}
$$

- $P$ 表示当前节点及其行为邻域已经兑现的质量；
- $U$ 表示该行为区域相对还获得了多少开发机会；
- $T$ 表示当前形成路径最近的改写是否仍值得继续开发。

三者共同决定哪个节点获得下一次评价机会；同一个 $T$ 再决定本轮的 `Develop`、`Explore` 或 `Crossover` 倾向。生成端接收当前代码、与它匹配的行为落地形成来时路，以及交叉时的第二父代。

## 1. 搜索认识

自动算法设计的每一步回答两个问题：给哪一个节点一次决策机会，以及怎样帮助它完成这次决策。当前 fitness 说明候选已经达到的质量；继续投资还读取它所在行为区域的质量前景与开发量，以及当前形成路径最近是在持续改进、重复既有行为，还是进入了尚未兑现的新方向。

TraceAAD 使用两条不同时间轴上的轨迹描述这些事实。求解轨迹记录一个固定算法如何逐步构造解，用于建立已访问算法之间的经验行为几何；形成轨迹记录算法如何经过多轮 `Idea + Code`、评价和修正到达当前状态，用于表示具体的发展历史并组织下一次生成上下文。

V9.19 的连接点是**行为落地形成边**。每个有效子节点不仅记录父子质量关系，还记录它相对生成前档案的行为新颖度。由此，一次声明的算法思想能够落到两个可观察结果上：它是否改善了目标质量，以及它是否到达了档案中较少见的实际求解行为。

质量提升承担完整信用。对于没有提升的形成边，行为新颖度区分无进展的行为重访与尚未兑现的行为变化。执行行为由此成为形成轨迹的解释变量，远距离本身只获得有限的中性信用。

## 2. 原子决策协议

每一个 primary evaluator slot 执行一次完整决策：

$$
\boxed{
\text{Select Parent}
\rightarrow
\text{Select Develop, Explore or Crossover}
\rightarrow
\text{Build Trajectory Context}
\rightarrow
\text{Generate One Algorithm}
\rightarrow
\text{Evaluate}
\rightarrow
\text{Recompete}
}
$$

每次生成只创建一个 `Idea + Code`。一次父代选择对应一个 primary slot；真实评价写回后，全部节点在下一个 slot 重新竞争。

## 3. 搜索状态

每个有效 Algorithm 节点 $a$ 保存：

- 完整代码与方向化 fitness $q(a)$，其中越大越好；
- 唯一父节点、形成时的 `Idea`、动作和 primary slot；
- 形成边的质量结果、行为新颖度和行为标签；
- 作为 parent 获得的 primary generation opportunities；
- BehaveSim 训练轨迹；
- 从初始根到当前节点的唯一形成路径。

每次 generation attempt 记录父节点、动作、结果类型、父子 fitness、错误与耗时。无效、超时和重复 attempt 不创建 Algorithm 节点，也不伪造行为新颖度；它们均经过真实 evaluator，消耗一个 primary 评价机会，并进入 parent opportunity coverage。

### 3.1 BehaveSim-v4 训练轨迹

$$
\boxed{d_B(a,b)=\text{BehaveSim-v4 train-trajectory distance}}.
$$

行为轨迹取自训练评价本身。V9.19 使用 tracked evaluation，逐位复刻基准 evaluator 的求解循环，包括实例、随机数消耗顺序和结果聚合。一次执行同时返回基准 fitness 与每个训练实例上的逐步选择序列。

| 任务 | 轨迹来源 | PSTraj 状态 |
| --- | --- | --- |
| TSP Construct | 训练集全部实例 | 逐步选点的部分路线，均匀保留 12 个状态 |
| VRPTW Construct | 训练集全部实例 | 含 depot 折返的路线前缀，均匀保留 12 个状态 |
| Online BPP | 训练集全部实例 | 累计箱号选择序列，均匀保留 12 个状态 |
| OP-ACO | 训练集全部实例 | 每轮迭代后的 incumbent 路线，保留 5 个状态 |
| CVRP-ACO | 训练集全部实例 | 每轮迭代后的 incumbent 路线，保留 5 个状态 |

对两个中间解状态 $u,v$：

$$
d_{\mathrm{state}}(u,v)=
\frac{\operatorname{Levenshtein}(u,v)}{\max(|u|,|v|)}.
$$

以 $d_{\mathrm{state}}$ 为局部代价执行 DTW，并用较短 PSTraj 的长度归一化。实例按索引配对，距离取全部训练实例的均值：

$$
d_B(a,b)=\operatorname{mean}_i d_{\mathrm{traj}}(a_i,b_i).
$$

轨迹来源、状态产生规则、状态保留数、DTW normalization 和实例聚合方式共同构成本版本的距离协议。行为区域只表示该协议与训练分布下的经验执行行为邻域。

### 3.2 增量缓存

程序第一次成为有效非重复节点后缓存其训练轨迹，之后永久缓存。新节点 $c$ 加入时只计算 $d_B(c,x)$，其中 $x\in\mathcal A_t$。旧节点之间的距离不重算。checkpoint 保存全部轨迹、distance matrix 和协议元数据；轨迹缺失时运行报错。

轨迹记录发生在计价的训练评价内部，不产生第二次执行，不额外消耗 primary fitness budget。训练分布上的违规候选按普通 invalid 处理。

## 4. 动态行为景观

在时刻 $t$，$\mathcal A_t$ 为具有有效 behavior profile 的节点集，$M_t=|\mathcal A_t|$。分配统计只使用本轮决策前已经完成的事件。

### 4.1 重叠行为区域

外部近邻数为

$$
k_t=\min\left(
M_t-1,
\max\left(2,\left\lceil0.05(M_t-1)\right\rceil\right)
\right).
$$

$\mathcal N_t(a)$ 是除 $a$ 之外 BehaveSim 距离最近的 $k_t$ 个节点。距离并列时按节点 ID 排序。定义

$$
\mathcal R_t(a)=\mathcal N_t(a)\cup\{a\}.
$$

新有效节点加入后，在下一次决策前重建全部邻域。区域随档案增长而变化，彼此可以重叠。

### 4.2 区域前景 $P$

先将当前 fitness 转为从低到高的中秩 percentile：

$$
Q_t(a)=\frac{r_t^{\mathrm{mid}}(a)-1}{M_t-1}.
$$

最优节点趋近 1，最差节点趋近 0；全部并列时取 $0.5$。区域前景同时保留当前节点已经达到的质量，以及相近执行行为是否形成了局部质量支持：

$$
\boxed{
P_t(a)=
\frac{
2Q_t(a)+
\operatorname{median}\{Q_t(x):x\in\mathcal N_t(a)\}
}{3}
}.
$$

当前节点质量占三分之二，外部邻域质量占三分之一。$P$ 是预算分配中的唯一质量概念量。

### 4.3 区域机会 $U$

$c_t(x)$ 为节点 $x$ 被选作 parent 后获得的 primary generation opportunities 数。每个 primary slot 计一次，包括 improve、plateau、regress、invalid、duplicate 和 timeout。初始化根的 $c_t$ 为 0。

$$
B_t(a)=\sum_{x\in\mathcal R_t(a)}c_t(x),
$$

$$
C_t(a)=\operatorname{RankPercentile}\left(\log(1+B_t(a))\right),
\qquad
\boxed{U_t(a)=1-C_t(a)}.
$$

全部 $B$ 并列时，$C$ 与 $U$ 取 $0.5$。$U$ 表示该行为区域相对欠开发的程度。无效、重复和超时虽然不形成新节点，仍会降低该区域之后的 $U$。

## 5. 行为落地形成轨迹

### 5.1 档案相对行为新颖度

设有效候选 $c$ 即将由父节点 $p$ 创建，$\mathcal A_t$ 是插入 $c$ 之前的档案。先计算档案中每个节点的最近邻半径：

$$
m_t(x)=\min_{z\in\mathcal A_t\setminus\{x\}}d_B(x,z).
$$

新候选到既有档案的最近距离为

$$
\rho_t(c)=\min_{x\in\mathcal A_t}d_B(c,x).
$$

将 $\rho_t(c)$ 放入当前最近邻半径分布，得到中秩 percentile：

$$
\boxed{
\nu_t(c)=
\operatorname{RankPercentile}
\left(
\rho_t(c);
\{m_t(x):x\in\mathcal A_t\}\cup\{\rho_t(c)\}
\right)
}.
$$

$\nu_t(c)\in[0,1]$ 是创建时刻的档案相对行为新颖度。该值随形成边一起冻结，尺度全部来自当前运行内的相对秩。

形成边记录为

$$
e_i=
\left(
\operatorname{Idea}_i,
o_i,
q(p_i),
q(c_i),
\nu_i
\right).
$$

为生成上下文提供的定性标签由 $\nu_i$ 的三等分位得到：

| $\nu_i$ | Behavior 标签 |
| --- | --- |
| $0\le\nu_i<1/3$ | `near-known` |
| $1/3\le\nu_i<2/3$ | `intermediate` |
| $2/3\le\nu_i\le1$ | `far-from-archive` |

标签描述候选在创建时相对固定训练轨迹档案的位置。

### 5.2 四类形成响应

对有效形成边 $e_i=(p_i,c_i)$，定义严格质量改善指示量

$$
y_i=\mathbf 1[q(c_i)>q(p_i)].
$$

$y_i$ 与 $\nu_i$ 共同区分四类过程状态：

| 质量结果 | 行为新颖度 | 过程含义 | 后续倾向 |
| --- | --- | --- | --- |
| improve | 低 | 有效局部开发 | 继续 Develop |
| improve | 高 | 有效行为发现 | 在新位置继续 Develop |
| no improve | 低 | 无进展行为重访 | 提高 Explore 倾向 |
| no improve | 高 | 尚未兑现的行为变化 | 保留有限继续开发机会 |

行为新颖度对质量信用的作用限制在未改善边。单边轨迹价值定义为

$$
\boxed{
v_i=y_i+\frac{1}{2}(1-y_i)\nu_i
}.
$$

因此，严格改善边始终取 $v_i=1$；未改善边的 $v_i\in[0,0.5]$，其信用上限严格低于质量改善。

### 5.3 统一轨迹响应 $T$

读取当前节点唯一形成路径上最近 $H_T=4$ 条有效形成边：

$$
h=\min(4,\operatorname{depth}(a)).
$$

使用对称先验平滑后，定义

$$
\boxed{
T(a)=\frac{1+\sum_{i=1}^{h}v_i}{2+h}
}.
$$

根节点的 $T=0.5$。$T$ 越高，最近形成路径越多地表现为真实改善；未改善但行为较新的边保持中性信用；反复落入既有行为且没有改善会降低 $T$。节点创建后，形成路径响应保持不变，但节点获得机会后的直接成败会形成平滑更新的有效响应：

$$
T_t^{\mathrm{eff}}(a)=
\frac{1}{2}T(a)+
\frac{1}{2}\frac{1+s_t(a)}{2+s_t(a)+f_t(a)},
$$

其中 $s_t$ 与 $f_t$ 分别是该节点已结算直接机会中的改善与未改善次数。分配和动作控制使用 $T_t^{\mathrm{eff}}$。

## 6. 预算分配

节点分数只使用 $P$、$U$ 与 $T$：

$$
\boxed{
S_t(a)=0.75P_t(a)+0.10U_t(a)+0.15T(a)
}.
$$

| 状态 | 权重 | 回答的问题 |
| --- | ---: | --- |
| $P$ | 0.75 | 当前状态及其行为区域是否已经显示质量前景 |
| $U$ | 0.10 | 该行为区域是否相对欠开发 |
| $T$ | 0.15 | 当前形成路径最近是否仍值得继续 |

父节点使用 Boltzmann 分布采样：

$$
\pi_{\mathrm{alloc}}(a)=
\frac{
\exp\left(\beta_t(S_t(a)-S_t^{\max})\right)
}{
\sum_b\exp\left(\beta_t(S_t(b)-S_t^{\max})\right)
}.
$$

$\beta_t\ge0$ 通过一维二分求解，使目标有效样本量为

$$
\operatorname{ESS}_t=\max(0.1M_t,2).
$$

目标不可达时使用非负 $\beta_t$ 上最近的可达分布；全部分数相同时使用均匀分布。Target ESS 只校准全局采样集中度。

## 7. 轨迹条件生成

### 7.1 Develop、Explore 与 Crossover

生成策略保留三个互补动作：

| 动作 | 生成目标 | 预期搜索行为 |
| --- | --- | --- |
| `Develop` | 保留当前算法的主要框架，选择一个最有价值的局部规则或实质机制继续开发 | 深化当前形成方向 |
| `Explore` | 提出具有明显不同主要决策逻辑的完整算法 | 打开新的行为与思想入口 |
| `Crossover` | 将当前父代与行为距离和质量兼顾的参考父代融合 | 迁移参考算法的有效结构并重入可行行为区域 |

选中父节点后，同一个轨迹响应 $T(a)$ 决定 Explore 概率；固定保留 $p_C=0.25$ 的交叉预算，其余概率用于 Develop：

$$
\boxed{
p_E(a)=
\operatorname{clip}
\left(
0.60-0.60T(a),
0.10,
0.60
\right)
}.
$$

当 $T=0.5$ 时，$p_E=0.30$、$p_C=0.25$、$p_D=0.45$。持续改善使生成偏向 Develop；无改善的行为重访降低 $T$，使生成偏向 Explore；Crossover 保留行为迁移和结构融合通道。节点被反复尝试后，直接机会的成功/失败率与形成路径响应共同更新有效 $T_t$，避免历史高响应节点在连续失败后长期占据预算。

### 7.2 形成轨迹上下文

生成上下文为

$$
\boxed{
\text{Current Code}
+
\text{Behavior-Grounded Formation Path}
+
\text{Action Instruction}
}.
$$

三个动作共享：

- 任务定义、可编辑接口与约束；
- 当前节点的完整代码；
- 与当前节点严格匹配的父代形成来时路，最近至多 $H=8$ 条有效形成边；
- 本轮 action-specific instruction。

`Crossover` 额外接收第二父代的完整代码、fitness、行为标签和与当前父代的 BehaveSim 距离。第二父代从质量分位和距离分位都不低于中位数的候选中按联合前沿采样；候选不足时退回联合得分最高者。

每条形成事件只提供：

```text
Idea: ...
Result: improve | plateau | regress
Fitness: parent -> child
Behavior: near-known | intermediate | far-from-archive
```

提示保留形成边的定性行为标签。原始 BehaveSim 距离、$\nu$ 数值、$P/U/T$、节点分数和区域统计留在控制器状态中；默认上下文只保留有效主父链事件。

**Develop**

> Continue developing the current algorithm. Preserve its main framework and make one coherent modification with a clear performance rationale. Use the formation path to identify what has already worked, what has been revisited, and which recent direction deserves refinement. You may improve a local rule or one substantive mechanism, but avoid redesigning unrelated parts.

**Explore**

> Propose a materially different algorithmic direction for the task. Change the main decision logic rather than making a cosmetic variation. Use the formation path to avoid repeating behavior that has already been revisited without improvement, while keeping the new design coherent and executable.

**Crossover**

> Combine the current algorithm with the provided reference algorithm. Identify one useful mechanism or decision rule in the reference and integrate it coherently into the current framework. Preserve the strong parts of the current algorithm, avoid copying code blindly, and return one executable hybrid design.

输出协议：

````text
Idea: <one concise design idea, <= 500 chars>

Code:
```python
<complete program>
```
````

程序只含 import、目标函数签名和可执行语句，不写模块或函数 docstring，也不写注释。设计理由放在 `Idea` 行。

## 8. 评价、修复与状态更新

候选生成后立即执行真实 evaluator。有效非重复候选 $c$ 的唯一父节点为本轮选中节点 $a$。Tracked evaluation 同时返回 $q(c)$ 与训练求解轨迹；随后计算 $d_B(c,x)$、$\nu(c)$、行为标签、形成边与 $T(c)$。新节点从下一个 primary slot 开始与全部已有节点共同竞争。

$$
\operatorname{Result}=
\begin{cases}
\operatorname{improve}, & q_c>q_p,\\
\operatorname{plateau}, & q_c=q_p,\\
\operatorname{regress}, & q_c<q_p.
\end{cases}
$$

Duplicate、invalid 与 timeout 不创建节点。本轮 primary slot 已消耗，父节点的 $c_t$ 加一，当前行为区域的 opportunity coverage 随之更新。

每个 primary candidate 最多执行 2 次有界修复。修复只提供当前运行错误，保持原 parent 与 action。初始候选消耗一个 primary slot；repair LLM calls 和 repair evaluator calls 单独记录。一个 primary attempt 在 opportunity coverage 中只计一次。

每次原子决策另存一条训练接口记录 $D_t$：task、parent id、current code、behavior-grounded formation path、action、LLM output、$q_p$、$q_c$、result、$\nu$、behavior tag、$P/U/T$，以及 `exact_prompt`、`exact_response`、`model_id`、`sampling_temperature` 和 `seed`。该记录供后续轨迹条件 RL 复原决策状态，见轨迹条件 RL。

## 9. 完整伪代码

```text
Generate 8 valid roots under one virtual root.
Evaluate each root; tracked evaluation returns fitness and training trajectories.
Set T = 0.5 and parent opportunity count = 0 for every root.

while primary evaluator budget remains:
    use only events settled before this slot

    compute fitness percentile Q(a) for every valid node
    rebuild the overlapping 5% kNN behavior region R(a)
    compute region promise P(a)
    compute opportunity U(a) from all spent parent opportunities

    S(a) = 0.75 P(a) + 0.10 U(a) + 0.15 T(a)
    sample one parent a with target-ESS Boltzmann probabilities

    p_explore = clip(0.60 - 0.60 T(a), 0.10, 0.60)
    sample Develop, Explore or Crossover
    if Crossover: select a strong, behavior-different reference parent and add its code to the prompt

    build current code + matched behavior-grounded formation path + action instruction
    generate exactly one Idea + Code
    charge one primary evaluator slot and evaluate
    run at most two bounded repairs when needed
    increment the parent's opportunity count once

    if valid and non-duplicate:
        obtain the training trajectory from tracked evaluation
        compute distances only from the child to existing nodes
        compute archive-relative behavior novelty nu before insertion
        create one behavior-grounded formation edge
        compute T(child) from the latest four formation edges
        cache the child and its trajectory

    return to global parent selection
```

## 10. 冻结参数

| 参数 | 值 |
| --- | ---: |
| 初始有效根 | 8 |
| 父代形成来时路长度 $H$ | 8 |
| 轨迹响应窗口 $H_T$ | 4 |
| 行为距离协议 | BehaveSim-v4 训练轨迹 |
| Behavior neighborhood | 最近 $5\%$，至少 2 个外部近邻 |
| 区域前景 | $P=(2Q+\operatorname{median}(Q_{\mathrm{neighbor}}))/3$ |
| 节点分数 | $0.75P+0.10U+0.15T$ |
| Target ESS | $\max(0.1M,2)$ |
| Behavior 标签 | $\nu$ 的三等分位 |
| 中性 Explore 概率 | $0.30$ |
| Explore 概率范围 | $0.10$--$0.60$ |
| 生成动作 | `Develop`、`Explore`、`Crossover` |
| Crossover 固定预算 | 0.25 |
| Bounded repair | 最多 2 次 |
| Primary evaluator slots | 1000 |

核心在线状态为

$$
\boxed{P,\quad U,\quad T}.
$$

## 11. 主实验

按主实验配置执行：任务为 `tsp_construct`、`cvrp_aco`、`op_aco`、`online_bin_packing`、`vrptw_construct`；每任务 3 次独立搜索；每次 1000 个 primary evaluator slots；训练集优化，held-out 评估不同规模新实例。正式结果同时报告 BehaveSim 额外计算成本。全部重复与测试完成后更新结果页。
