# TraceAAD V10 完整机制设计

V10 实现 Trajectory-aware Joint Design Opportunity Allocation。当前科学认识见研究认识。本文给出可复现的机制定义。树结构、形成路径、有界执行修复与评价口径沿用既有 TraceAAD 平台。

V10 是逐 evaluator-slot 重规划的 controller。不变量为

$$
a_t\in\Omega(\mathcal{H}_t),
\qquad
\text{one primary evaluation per decision},
\qquad
\mathcal{H}_{t+1}\text{ immediately re-enters allocation}.

$$

## 1. 总览与搜索协议

在评价预算 $B$ 下，时刻 $t$ 的历史 $\mathcal{H}_t$ 是用于分配的全局搜索状态，剩余预算 $b=B-t$。每一步选择一次设计决策

$$
a=(s,o,r).

$$

$s$ 是作为起点的已评价程序节点，Restart 时为空。$o$ 是设计干预。$r$ 是参照状态，仅 Transfer 非空。

分配分两阶段。估价构造竞争集合 $\mathcal{C}_t$，排除当前证据下缺乏投资理由的设计决策；覆盖在 $\mathcal{C}_t$ 内部按暴露次数抑制对同类 state–action 区域的反复消费。每一次正式评价之后立即重新进入分配。初始化生成 $N_{\mathrm{root}}=8$ 个有效根，各建一条线程。搜索循环为

$$
\mathcal{H}_t\rightarrow S_t\rightarrow\Omega_t\rightarrow
\text{valuation}\rightarrow\text{allocation}\rightarrow
\text{realization}\rightarrow E\rightarrow\mathcal{H}_{t+1}.

$$

```text
Algorithm 1  TraceAAD V10
initialize N_root valid roots, each with a thread
for each primary slot:
    screen candidate states
    construct design opportunities
    evaluate opportunities with critic
    form competitive set
    allocate by coverage
    realize selected opportunity
    evaluate candidate
    update state and thread statistics
return argmax_P q_train(P)
```

## 2. 节点描述与线程统计

$s$ 是已评价程序节点的局部描述；$\mathcal{H}_t$ 才是用于分配的全局搜索状态。

$$
s=(P,q,\tau,A,\Gamma)

$$

四类字段分别提供当前实现 $(P,q)$、形成证据 $\tau$、局部干预证据 $A$，以及线程级延迟证据 $\Gamma$。$P$ 与 $q$ 给出当前代码和已兑现的训练质量，并附带档案 mid-rank、创建时刻、父节点与所属线程。$q$ 不解释为潜力。

### 2.1 形成路径与尝试账本

形成路径 $\tau$ 取最近至多 $H_\tau=8$ 步，每步为 Idea、结果（improve / plateau / regress）与 fitness 变化。估价与生成共用这一表示，使用方式不同。

直接尝试账本 $A$ 按算子记录从该节点出发的每一次分配：Idea、结果、父子质量、槽位、子节点；结果包括 invalid、timeout、duplicate。另记参照对次数 $n_{\mathrm{pair}}(s,r)$，供日志与估价查看。账本默认不进入生成提示。

### 2.2 实验线程

实验线程 $\Gamma$ 是按形成谱系记账的单位，记录声明与预算消耗。打开类动作仅在得到有效子代时新建线程，并写下当时的 `origin_idea`。延续类动作的有效子代加入起点所属线程，并继承 `origin_idea`。失败的打开不建线程。初始化根线程的 `origin_action` 为 init，不进入打开类动作的 $G_2$、$G_4$ 与恢复统计。线程无显式关闭：暂时没有节点再被选中则休眠，历史节点再次成为起点则重新活跃。

Pivot 与 Transfer 的参照质量 $q_{\mathrm{origin}}$ 为起点当时的 $q$。Restart 的 $q_{\mathrm{origin}}$ 为创建时刻的全局最好质量。线程预算与有效质量序列分开：凡正式评价的起点属于该线程，或该次评价正是创建本线程的打开动作，无论成败，`opportunities_used` 加一；无有效子代时最好质量保持不变。于是 $h$ 是该线程消耗的正式评价次数，

$$
B_h(\Gamma)=\max\{\,q(P):P\text{ valid in }\Gamma\text{ after }h\text{ primary slots}\,\},

$$

$$
G_h(\Gamma)=B_h(\Gamma)-q_{\mathrm{origin}},\qquad h\in\{1,2,4\}.

$$

$G_h$ 只描述一条线程在消耗若干评价机会后是否最终兑现，不直接参与分配公式。仅当 `opportunities_used >= h` 时该线程进入 $G_h$ 的统计，缺失不补零。

五个算子都向估价器提供尝试次数、有效率、一步改善率与一步质量变化。仅 Pivot、Transfer、Restart 额外提供 $P(G_2>0)$、$P(G_4>0)$ 与恢复所需次数。这些量是观察事实。

## 3. 设计机会空间

五个动作按下一步设计与已评价程序的关系划分：

$$
\mathcal{O}=\{\mathrm{Develop},\ \mathrm{Pivot},\ \mathrm{Transfer},\ \mathrm{Restart},\ \mathrm{SemanticRepair}\}.

$$

Develop 与 SemanticRepair 保持对当前核心假设的接受；Pivot、Transfer 与 Restart 改变这一关系，并在得到有效子代时新建线程。

| 动作           | 转移语义                                                                 |
| ---------------- | -------------------------------------------------------------------------- |
| Develop        | 保留当前核心算法思想并继续兑现；参数、局部结构与内部组织可由模型自行决定 |
| SemanticRepair | 在程序可执行的前提下，修正思想与实现之间的不一致                         |
| Pivot          | 仍从当前程序出发，撤回对核心机制的信任，允许替换主要决策逻辑             |
| Transfer       | 指定参照$s_j$，把其中已由评价支持的机制迁入 $s_i$                        |
| Restart        | 不绑定父代，重新提出假设                                                 |

Develop 与 Pivot 的语义边界在于是否延续当前核心设计假设：Develop 保留既有假设并深化实现，Pivot 从当前脚手架出发探索新机制，Restart 则独立生成全新假设。语法、运行时、接口与超时实现问题在进入正式评价前至多修复两次，计入 repair 调用，不计入五类设计动作。再次选中档案中的祖先，即回到该状态继续。

## 4. 候选构造与机会估价

### 4.1 起点筛选

档案过大，价值判断不能对全部节点展开。先用有界筛选指数取出 $K_s=8$ 个起点。质量用标准化 mid-rank：唯一最好为 1，唯一最差为 0；$N=1$ 或全体质量相等时取 $0.5$。否则

$$
Q_t(s)=\frac{\#\{s':q(s')<q(s)\}+\frac12\bigl(\#\{s':q(s')=q(s)\}-1\bigr)}{N-1}.

$$

$n_t(s)$ 为该节点作为起点被分配的次数，含失败尝试。筛选指数为

$$
R_t(s)=Q_t(s)+\frac{1}{\sqrt{1+n_t(s)}}.

$$

两项均在 $[0,1]$。该指数仅承担候选压缩作用，不被解释为节点价值或未来潜力。取 $R_t$ 最高的 8 个有效节点；若全局最好不在其中，替换指数最低者。筛选只决定 Critic 看见哪些状态。

### 4.2 机会构造

对每个起点 $s$，从其他线程中质量最高、且非祖先非后代、已有真实评价的节点中取 $K_d=2$ 个参照。互补性留给估价。每个 $s$ 产生 Develop、Pivot、SemanticRepair，以及每个参照一张 Transfer。全局恰有一张 Restart。SemanticRepair 始终进入 $\Omega_t$；估价器给不出具体不一致时标为不适用，不进入竞争集合。

### 4.3 机会估价

每次正式评价之前调用一次语言模型，只做相对判断，不生成代码。该调用计入 critic 的次数与 token，不计入 primary budget。

输入包括任务、当前槽位、剩余预算、全局最好、按第二节分工的算子观察，以及每个起点的状态卡：质量与 mid-rank、代码、形成路径、线程的 `origin_idea` 与 $G$ 摘要。账本每个算子只展示合计次数与最近一次尝试。同一节点代码按编号去重。Transfer 另附参照的 Idea 摘要与代码。估价提示与模型回复须一起落在服务上下文窗口内：代码总量按保守字符预算控制，先等比截断各份代码，不够时按适应度升序略去参照代码（保留 Idea 摘要），起点代码保底；是否裁剪记入决策日志。

估价依据即时收益、后续开发可能、信息价值、兑现延迟与剩余预算。信息价值指：一次评价若能消解将实质改变后续分配的不确定性，即使即时收益不确定，仍可进入竞争集合。当前质量或一步退步单独都不足以决定入选。

输出为至多 $K_c=4$ 个机会的竞争集合 $\mathcal{C}_t$，表示当前证据下仍具有决策竞争力的候选，按 `rank` 排序。引用必须能对应到真实的路径或账本条目。SemanticRepair 进入 $\mathcal{C}_t$ 时，字段 `semantic_mismatch` 必须给出具体不一致。解析失败则重试一次；仍失败则取短名单中质量最高的 $K_c$ 个 Develop，并记录 `critic_invalid`。

```json
{
  "competitive_set": [
    {
      "opportunity_id": "O17",
      "rank": 1,
      "reason": "...",
      "evidence_refs": ["S17-H3"],
      "expected_payoff_horizon": "short",
      "semantic_mismatch": null
    }
  ],
  "not_applicable": [
    {"opportunity_id": "O11", "reason": "No semantic mismatch is evidenced."}
  ]
}
```

## 5. 覆盖分配与条件化实现

价值判断已经收在 $\mathcal{C}_t$ 内。覆盖只在该集合上抑制对已反复试验的 state–action 区域的集中。覆盖粒度为 $(s,o)$，

$$
C(a)=\bigl(n(s,o),\;n(\Gamma(s),o),\;n_G(o)\bigr).

$$

Transfer 使用 $n(s,\mathrm{Transfer})$。Restart 的线程覆盖为零。选取 $C(a)$ 字典序最小者；全同则取更小的 `rank`；再同则用运行种子派生的随机数。剩余预算由估价器直接阅读观察事实。

选定 $a$ 之后，生成器默认看见任务、当前算法、形成路径与本次算子指令。Transfer 附加参照的代码与路径。SemanticRepair 附加 `semantic_mismatch`。Restart 附加至多 $N_{\mathrm{card}}=3$ 张已验证改进卡，每张仅含 Idea、结果与 fitness 变化，不含代码。输出为一句 Idea 与一份完整程序，函数契约与既有生成协议相同。

**Develop.** Preserve the current algorithm's core design hypothesis. Improve it coherently. You may tune parameters, strengthen a local or deep mechanism, simplify harmful details, or restructure supporting logic, but do not replace the main algorithmic idea.

**Pivot.** Use the current algorithm as a starting point, but do not assume its core design hypothesis is correct. Replace or substantially redesign one central decision mechanism and create a coherent alternative direction.

**Transfer.** Preserve the useful structure of the source algorithm. Identify one mechanism in the donor that is supported by its evaluator history and integrate that mechanism coherently. Do not copy code mechanically.

**Restart.** Propose a new algorithmic hypothesis not anchored to an existing parent.

**SemanticRepair.** Preserve the intended algorithmic hypothesis. Correct the identified semantic inconsistency between the intended mechanism and its actual implementation. Do not redesign unrelated parts.

## 6. 评价与状态更新

候选先经过语法、签名与有界执行检查。正式评价占用一个 primary slot。有效且非重复的程序加入档案，并更新 $\tau$、$A$、$\Gamma$、$G_h$ 与覆盖。随后立即构造下一轮机会。

## 7. 实验协议与常数

$$
K_s=8,\quad K_d=2,\quad K_c=4,\quad H_\tau=8,

$$

$$
H_G=\{1,2,4\},\quad N_{\mathrm{root}}=8,\quad N_{\mathrm{card}}=3.

$$

估价每槽一次；执行修复至多两次。$K_d$、$N_{\mathrm{card}}$、$H_G$、$N_{\mathrm{root}}$ 视为协议常数。以后若考察敏感性，优先看 $K_s$、$K_c$ 与 $H_\tau$。

第一轮将 V10 与同平台 V9.16 以及主实验外部方法对照。生成模型为 Qwen3.8-27B，五个正式任务，1000 次 primary 评价，三次独立搜索。主张限定为固定评价预算下的搜索质量。V10 每个 primary slot 增加一次估价调用，因此同时报告 primary 评价次数、生成调用与 token、估价调用与 token、修复调用与 token。过程量包括 best-at-100/250/500/750/1000、训练最好、held-out、算子占用、打开类线程的 $G_1/G_2/G_4$、竞争集合内容与 `critic_invalid` 次数。

若联合系统出现信号，再拆开轨迹、账本、联合选择、估价、覆盖与五算子，分别对照。
