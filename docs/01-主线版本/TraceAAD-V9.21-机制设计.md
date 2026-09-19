# TraceAAD V9.21：思想假设的双重实现搜索

## 1. 版本决定

V9.21 把自动算法设计看作一连串可检验的算法思想实验，搜索对象设定为包含实现路径的**思想假设**。

每轮从同一个已经测量过的算法状态出发，独立提出两个思想：

1. `continue`：继续、修正或重新实现当前假设；
2. `branch`：提出一个新的算法假设，并可参考一张来自其它分支的真实改进卡。

每个思想随后独立生成两份完整代码。四份代码都调用一次真实 evaluator，结果
在本轮结束后记录。这个 `2 ideas x 2 realizations` 是同状态的假设辨识实验，
没有候选之间的上下文传递，也不预支后续预算。

V9.21 首跑只验证这条因果链：

$$
\text{idea hypothesis}
\rightarrow
\text{independent realizations}
\rightarrow
\text{evaluator evidence}
\rightarrow
\text{next opportunity}.
$$

## 2. 任务对象与第一性原理

一个 AAD 任务由问题描述、可编辑程序模板和 evaluator 组成：

$$
\mathcal T=(d_{\mathcal T},K,\mathcal E),
\qquad P_r=K[r],
\qquad q(r)=\mathcal E(P_r;S_{train}).
$$

模型提出或改写算法函数 `r`，evaluator 在固定训练实例上返回质量。所有五个
正式任务都转换为 higher-is-better fitness。

一次 `Idea + Code` 的结果混合了两个随机因素：思想是否适合任务，以及模型这次
是否正确实现了思想。单次退步只约束这次实现；它不能单独判定思想已经没有价值。
同样，单次高分只证明这一份实现已经有效。

人类专家会保留三件相互关联的事实：

- 可以返回的高质量基础实现；
- 当前正在尝试兑现的工作实现；
- 每次尝试、退步、修复和回退的记录。

V9.21 将这三件事实组织成一个可重复实验的假设状态。轨迹的作用是让下一次
决策知道当前代码为何成为现在的样子，以及某个思想在实现上还留下什么问题。

## 3. 科学状态

### 3.1 Program Node

每个有效、非重复的候选形成一个 `ProgramNode`：

```text
id, code, fitness, parent_id, hypothesis_id
declared idea, role, primary slot
```

`parent_id` 表示真实形成边。兄弟实现使用同一个冻结的 parent，不把一个兄弟
的代码传给另一个兄弟。无效、超时和重复响应也写入 evaluator 记录，但不形成
新的 Program Node。

### 3.2 Idea Hypothesis

一个假设 `h` 保存：

```text
entry_idea
source_node_id
stable_scaffold_node_id
working_node_id (optional)
parent_hypothesis_id
public_card provenance (optional)
realization responses
```

`entry_idea` 是模型提出的可检验主张。它不是已经确认的算法家族标签，也不用于
硬聚类。

### 3.3 Stable Scaffold

`stable_scaffold` 是这个假设随时可以退回的最高质量有效代码。

- root 假设的 scaffold 是 root program；
- branch 假设的 scaffold 是提出它的 parent scaffold；
- 某次 realization 严格超过当前 scaffold 后，新的节点成为 scaffold；
- scaffold fitness 在搜索中单调不降。

新思想的首版实现即使退步，也不会覆盖 source scaffold。这个保存动作只保护
可恢复的实验起点，不给退步本身附加质量信用。

### 3.4 Working Implementation

`working` 是当前最接近兑现该思想的有效实现。它可以低于 stable scaffold：

- 有效 realization 比当前 working 更好时更新 working；
- realization 超过 scaffold 时同时更新两者；
- 没有有效 realization 时 working 为空；
- 下一次生成同时看到 scaffold 和 working，模型可以继续工作版本，也可以从
  scaffold 重新实现思想。

### 3.5 Realization Evidence

每次 realization 保存：

```text
hypothesis, idea, frozen parent, fitness or failure
response = normalized(candidate fitness - frozen scaffold fitness)
outcome relative to parent, primary slot, repair count
```

失败代码保留在档案中，默认上下文只呈现失败类型、结果和发生顺序，不把缺陷代码
当作示范答案。

## 4. 原子实验：两个思想、各两次实现

### 4.1 初始化

每次运行先独立生成 8 个 root `Idea + Code`，每个有效 root 消耗一个 primary
evaluator slot 并建立一个 Hypothesis。root 阶段的 `Idea + Code` 是启动搜索所需
的初始多样性；2×2 协议从普通 batch 开始。

8 个 root 完成后冻结质量尺度。设 root fitness 为 $q_1,\ldots,q_R$：

$$
m_0=\operatorname{median}(q_1,\ldots,q_R),
$$

$$
s_0=1.4826\operatorname{MAD}(q_1,\ldots,q_R).
$$

若 MAD 和 root 极差都为零，取 $s_0=1$。之后只用这个运行内尺度：

$$
z(q)=\operatorname{clip}\left(\frac{q-m_0}{s_0},-8,8\right).
$$

### 4.2 冻结 parent 状态

每个 batch 开始时选择一个 Hypothesis $h$，冻结：

- stable scaffold 及其 fitness；
- working implementation 及其 fitness；
- scaffold 的真实 formation path；
- $h$ 最近的 realization evidence；
- 一张可选的 public experiment card。

batch 中四份代码都使用这份快照。continue 产生的结果不会进入同一 batch 的
branch prompt。

### 4.3 独立提出两个思想

先调用一次 idea prompt，再为该 idea 调用两次 realization prompt。

`continue` prompt 要求模型保留 entry hypothesis，选择一个由真实 evidence 支持
的修复、回退、重新实现或精炼。它可以改变代码结构，但不能悄悄换成无关问题。

`branch` prompt 要求模型提出一项实质不同的新假设。它只额外看到一张 public
card；没有 card 时只使用 parent 的 scaffold 和 formation path。

两个 idea 请求相互独立。idea 文本只用于定义本轮待检验假设，不作为性能标签。

### 4.4 两次独立实现

对每个 idea，使用完全相同的冻结上下文调用两次 code prompt：

- request seed 不同；
- sibling 不可见；
- 两次都从同一个 scaffold/working 快照开始；
- 两次都完整输出目标函数实现。

四份代码逐一调用 evaluator。primary budget 剩余不足四个 slot 时，按剩余数量
完成前缀并记录尾部 batch 的实际大小。

同一 idea 的两次结果使下面两个问题可以分开观察：

1. 一个思想是否至少有一种稳定实现；
2. 当前实现失败是否只是一次落地偶然性。

## 5. 机会政策

### 5.1 一步实验上界

机会值只回答一个有限问题：**如果现在再给这个假设一次实现机会，它的结果
上界有多高？**

对假设 $h$，令 $n_h$ 是已结算 realization 次数，$\bar r_h$ 是响应均值，
$t$ 是已完成的普通 batch 数。定义：

$$
U_t(h)=\sqrt{\frac{\log(t+2)}{n_h+1}},
$$

$$
O_t(h)=z(q_{scaffold,h})+\bar r_h+U_t(h).
$$

这是单一的一步 UCB：

- $z(q_{scaffold,h})$ 表示已经兑现的质量；
- $\bar r_h$ 表示该思想最近实现相对 scaffold 的响应；
- $U_t(h)$ 表示尚未充分实现的实验不确定性。

三个量使用同一个冻结 root scale，没有手工的轨迹特征权重、行为权重或语义
权重。无 realization 的新假设取 $\bar r_h=0,n_h=0$，只获得一次随实验次数
增加而缓慢下降的上界；重复失败会同时降低响应均值并消耗不确定性。

`invalid`、`timeout` 和 `duplicate` 的响应为 $-2$。有效候选的响应为：

$$
r(c)=\operatorname{clip}\left(
\frac{q(c)-q_{scaffold}}{s_0},-2,2\right).
$$

响应衡量本轮实现是否缩短当前思想与稳定脚手架之间的距离。它不声称预测
思想的最终上限，也不把过去累计增益再次计入信用。

### 5.2 选择 parent

每个普通 batch 计算全部 Hypothesis 的 $O_t$，选择最大者；精确并列时用运行
内 RNG 打破平局。选择快照和全部机会值写入 `mechanism_events.jsonl`。

这是一项可审查的 sequential decision policy：输入、目标和不确定性项都固定，
不再叠加 ESS、覆盖分数、访问奖励或独立节点后验。首要因果对照可以把 $O_t$
替换成 scaffold quality，保持其余协议完全相同。

### 5.3 探索与利用

每个 batch 同时包含 continue 和 branch：

- continue 把预算用于当前思想的兼容性、修复和精炼；
- branch 把预算用于新的思想假设，并可借鉴公共证据。

这个配对结构提供固定的探索宽度和发展深度，同时允许 parent 政策依据已兑现
质量和实现响应把更多 batch 投向有希望的假设。没有跨任务手工设定
`develop/explore/crossover` 比例。

## 6. 全局共享记忆

### 6.1 Public Experiment Card

全局记忆只保存真实的严格改进边：

```text
parent fitness -> child fitness
recorded Idea
child code
parent/child node ids and slot
```

每次 branch 最多检索一张来自其它 formation branch 的 card。它由真实 evaluator
结果产生，提供一个已经兑现的设计变化供模型判断是否兼容。

card 不和 parent 的路径串成虚拟历史，不生成长期反思摘要，不建立算法家族名录。
相邻 card 之间没有被假设的因果关系。选择时从最近若干张真实改进卡中抽取，避免
一个早期全局 best 锚定整个搜索。

### 6.2 记忆的科学含义

公共记忆检验的是“别的分支已经兑现的机制，能否帮助当前 scaffold 形成兼容的
新实现”。它传递的是可复核的实验事实，不传递控制器对思想价值的主观判断。

如果 public card 与随机高质量代码没有差异，V9.21 只保留私人 formation path，
不把共享记忆写成有效机制。

## 7. 生成上下文契约

### 7.1 Continue

上下文包含：

- 任务描述和目标函数契约；
- stable scaffold 完整代码与 fitness；
- working implementation（若不同）与 fitness；
- entry idea；
- scaffold 最近 8 条真实 formation edges；
- 当前 hypothesis 最近 6 条 realization evidence。

模型可以继续工作实现、回到 scaffold、撤销上次变化或重新落地同一思想。输出
idea 和 code 的职责分开：idea 先定义实验主张，code 再独立实现它。

### 7.2 Branch

上下文包含 continue 的任务和 scaffold 信息，但 entry idea 标记为新方向，并
附带最多一张 public card。模型必须解释性地借用兼容机制，保持目标函数签名和
有效性约束。

### 7.3 失败反馈

模型输出无效、执行异常或超时时，最多进行两次 bounded repair。repair prompt
保留待测 idea，只提供失败类型、错误尾部和失败代码；每次 repair evaluator 调用
单独计数，不能改变 primary slot 数量。

## 8. 预算、错误和恢复

正式搜索预算是 1000 个 primary evaluator slots：

- 8 个 root 各消费一个 slot；
- 普通 batch 最多消费 4 个 slot；
- duplicate、invalid 和 timeout 仍消费 primary slot；
- repair evaluator 调用计入 `evaluator_call_count`，不增加 primary slot；
- 模型调用、repair 调用和 evaluator 调用分别报告。

每次候选在 evaluator 前写入 pending checkpoint，在结算后清除。checkpoint 保存：

```text
nodes, hypotheses, realization evidence, attempts
primary/evaluator/LLM counters
root normalization, RNG state, public memory ids
pending candidate and unfinished paired batch context
```

恢复时先结算 pending，再完成同一 batch 的剩余 sibling；已经落盘的 response 不
重新生成。失败必须在 `evaluations.csv` 和 `mechanism_events.jsonl` 中可见，不能
用默认 fitness 或静默 fallback 伪装成结果。

## 9. 在线复杂度边界

V9.21 在线只运行标准任务 evaluator。它不生成 PSTraj、不计算 BehaveSim、不维护
行为距离矩阵，也不为每个候选额外执行 probe。

BehaveSim 若要使用，只作为预先匹配的离线过程画像，分析最终 best、恢复成功的
hypothesis 和 public-card child。它不能进入 parent 机会、全局 card 检索或生成
提示。这样 OBP 等任务的 evaluator 成本仍与历史标准方法同口径。

后续对 V9.19 两套实现、旧画像墙钟、OBP 长前缀距离规模和 ACO 多随机流稳定性
的完整追溯见BehaveSim 在线可行性复审。该复审支持本节的退出决定，并进一步明确“群体几何有信息”与“在线维护几何划算”是两个独立命题。

per-instance performance vector 是有价值的后续候选：标准 evaluator 当前接口
只返回聚合 fitness，首跑不修改任务评价语义。若后续加入，必须在同一次 evaluator
执行中输出，并用 aggregate-only 的匹配对照识别其增量。

## 10. 可检验预测

1. 首次实现低于 source scaffold 的 hypothesis 中，会出现非连续恢复并最终超过
   scaffold 的实例。
2. 两次 realization 的均值比单次结果更能识别“思想可实现性”，同时保留一次
   偶然好实现的记录。
3. stable scaffold + working implementation 的上下文会提高回退后重新实现的
   有效率，减少沿坏代码继续堆叠。
4. $O_t$ 对下一批 realization response 的滚动排序优于只看 scaffold quality，
   才能支持机会政策的增量主张。
5. public card 只有在其机制与当前 scaffold 兼容时才会产生改进；盲目复制会以
   invalid、低响应或重复率表现出来。

## 11. 历史坑审计

| 已知问题 | V9.21 的处理 |
| --- | --- |
| 首次退步立即删除路线 | scaffold 保留，working 单独记录，后续可从 scaffold 重做 |
| 固定多步 landing | 每个 batch 结算后重新计算机会值，只有当前 batch 内的 2×2 是原子协议 |
| 把累计 gain 当未来信用 | 响应只相对当前冻结 scaffold 计算，历史 gain 不重复加分 |
| 每节点独立后验导致臂爆炸 | 后验按 hypothesis 聚合，不按每个代码节点建立臂 |
| 多代 pending 和右删失 | 每个 candidate 立即 evaluator，pending 只用于崩溃恢复 |
| 行为新颖性冒充价值 | BehaveSim 完全退出在线控制 |
| 全局广播污染上下文 | 每个 branch 最多一张真实 card，continue 不看公共代码 |
| 预定义算子标签与实际改动错位 | 模型自行决定修复、精炼、回退或融合；continue/branch 只定义实验问题 |
| 高维代理堆叠导致无法归因 | 首跑只保留 scaffold、working、response、UCB 四个直接量 |
| 标量均值遮住条件优势 | 作为后续 per-instance matched ablation，不在首跑偷偷改变目标 |

仍需警惕：UCB 不确定性可能奖励胡乱表达，双实现可能降低决策频率，public card
可能诱发复制，聚合 fitness 可能掩盖实例条件。这些风险有明确的消融和停止信号，
不能用机制故事预先覆盖。

## 12. 最小识别实验

所有正式比较统一 Qwen3.6-27B、1000 primary slots、三次独立搜索和相同任务
evaluator。先做以下 matched 对照：

### 12.1 机会政策

- `Scaffold-Q`：保持 2×2 生成和上下文，只按 stable scaffold quality 选 parent；
- `Hypothesis-UCB`：使用本规范的 $O_t$。

报告滚动下一批 response 排序、恢复率、scaffold breakthrough、best-at-budget
和 held-out 结果。

### 12.2 思想与实现

- `1 realization` 对 `2 realizations`：同一 idea、同一 parent、相同总 primary
  预算口径；
- 只改变实现次数，不能同时改变 prompt、parent 政策或 repair。

### 12.3 公共记忆

- `private-only`；
- `one real public card`；
- `shuffled card` 或随机高质量 donor。

只有匹配实验出现增量，才分别讨论 2×2、机会政策或公共记忆的作用。完整联合
版本只评价系统整体，不把终局差异分配给每个组件。

## 13. 固定运行参数

| 参数 | V9.21 |
| --- | ---: |
| Primary evaluator slots | 1000 |
| Initial roots | 8 |
| Ideas per ordinary batch | 2 |
| Realizations per idea | 2 |
| Nominal slots per ordinary batch | 4 |
| Formation history shown | latest 8 edges |
| Realization evidence shown | latest 6 events |
| Response clipping | [-2, 2] |
| UCB | `sqrt(log(t + 2) / (n + 1))` |
| Public cards per branch | at most 1 |
| Public card archive | latest 64 strict improvements |
| Max bounded repairs | 2 per candidate |
| Online BehaveSim | disabled |
| Idea embedding | disabled |
| Per-instance objective | disabled in core |

模型服务、采样温度、request seeds、evaluator 参数、任务实例顺序和完整 prompt
写入 `run_config.json` 及 `decisions.jsonl`。不同后端只表示同一 Qwen3.6-27B 的
服务源，不构成模型因素。

## 14. 工件

每次运行至少保存：

- `evaluations.csv`：primary slot、batch、proposal、hypothesis、fitness、response、
  repair 和错误；
- `decisions.jsonl`：idea 与 realization 的 exact prompt/response、seed 和 public
  card provenance；
- `hypotheses.jsonl`：每次更新后的 scaffold、working、响应均值与来源；
- `global_memory.jsonl`：真实严格改进卡；
- `mechanism_events.jsonl`：batch 快照、机会值、结算事件和恢复事件；
- `best_program.py`、`best_history.jsonl`；
- `checkpoints/latest.json`、`checkpoints/view.json`、`logs/summary.json`。

原始候选代码只保留本地工件，Git 跟踪入口、评估和凝练结果。正式结果页在三次
搜索和 held-out 完成后更新。

## 15. 研究叙事

V9.21 的主张顺序固定为：

1. AAD 需要在有限预算中提出、实现、检验和精炼算法思想；
2. 单次代码评价无法稳定区分思想质量与实现偶然性；
3. 轨迹提供思想形成和实现缺口的事实，stable scaffold 使失败后可以回到可用
   基础；
4. 同状态两个思想、每个思想两次实现，直接测量思想的可实现性；
5. 一步 UCB 只分配下一次实验机会，不把它解释成长期潜力；
6. 一张真实公共改进卡检验跨分支借鉴，结果由匹配消融决定。

直觉上，这个版本比把访问、行为距离、累计增益和固定 continuation 叠加在节点
分数上更接近 AAD 的任务对象，因为每个控制动作都对应一个可观察的算法实验。
性能优越性仍是待验证命题；首跑的成功标准还包括失败可见、恢复可复现、因果链
可归因。
