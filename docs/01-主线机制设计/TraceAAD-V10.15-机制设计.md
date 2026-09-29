# TraceAAD V10.15 机制设计

## 0. 版本身份

V10.15 是一个独立版本，不是 V10.14 的后代，版本号只作标识。它把 V9.7、V9.14、V9.16、V9.19、V10.11、V10.13 和 V10.14 中有证据支持的部分重新组合成一个简单、完整的机制。它只复用 V10.14 的工程组件：解析、评价器封装、候选记账和独立选择集。

一句话主张：**找到好的算法思想，并在来时路的护栏下持续精炼。**

---

## 1. 设计原理

### 1.1 最终质量从哪里来

五任务强结果的形成过程（见[强结果形成路径](../03-机制探索与验证/2026-09-09-强结果形成路径.md)）有一个共同结构：

> 最终质量 ≈ 找到好的算法思想（骨架）× 在这个骨架上连续改进的复利

- TSP："在选点函数里构造剩余路径并做 2-opt"这一骨架被引入（两路明确来自 Explore）后，随后的 Refine 连续改进，最好谱系平均深约 77 步（V9.14）。
- CVRP：在一条来源上连续精炼几何与容量规则，有效深度 65–73 步（V9.7）；最后几步往往只是调常数。
- VRPTW：先经交叉迁入前瞻模拟，再修正局部评价方式。
- OP：以跨程序组合和参数精炼为主，精炼很早就饱和。
- OBP：保结构的小幅数值与项的调整。

因此机制要回答两件事：怎样更早、更多地产生好骨架，以及怎样让骨架上的改进一步步累积起来。

### 1.2 轨迹作为护栏：已有证据

"护栏"指在生成时让模型看到当前程序是怎么一步步形成的，使它保留已经起作用的部分，少做破坏性改写。

| 证据 | 结果 |
|---|---|
| V9.7 固定锚点配对（同一锚点，有无来时路） | TSP/CVRP/OP 落入最差十分位的改写从 16%/23%/16% 降到 6%/4%/8%；修改比例从 0.71–0.80 降到 0.36–0.64；12 个"任务 × 质量层"的效应点估计全部为正，7 个区间不跨 0 |
| V9.8 配对研究 | 父代历史相对只给代码，72 个锚点中 58 个方向更好 |
| V9.14、V9.16、V9.19、V10.11 | 这几个版本都带来时路，在 TSP/VRPTW 的 held-out 上都排在前列 |
| V9.7 配对：在来时路之外追加同父已试子代 | 35 好 / 34 差 / 3 平；CVRP 改进率从 18.5% 降到 9.3%。护栏应当是来时路，而不是同父已试 |

### 1.3 为什么轨迹当"指南针"没有用

"指南针"指用轨迹估计哪里值得继续投资，再据此分配预算。历次尝试包括：
- V9.3 的 rollout；
- V9.8 的假设信用；
- V9.10 的 Thompson 抽样：到预算结束时，36–53% 的动作还没有结算；
- V9.15 的增长项：改变了 30–76% 的选择，TSP/CVRP 反而变差；
- V9.16 的三步续段：最终最好程序没有一个来自续段；
- V9.19 的轨迹响应项：对分配的贡献最小，而且和算子信号几乎不相关；
- V9.20 的延续价值；
- E2-B' 的垫脚石干预：没有观察到垫脚石；
- V10.14 的局部重验构想。

这些尝试都没有带来稳定收益。原因可以从第一性原理说清：

1. **预测需要统计，统计需要重复样本。** 一个节点通常只被选中几次，而一次改写的结果方差很大：Refine 的严格改进率只有 4–28%，多数结果持平或退步。凭几次观测去估计"继续投资的价值"，得到的几乎全是噪声。
2. **轨迹描述的是祖先，不是当前程序。** 程序一旦改变，祖先身上的改进率就不再适用。V9.7 诊断也发现，深度、历史改进比例、路径净增益都不是稳定的效应修饰量。
3. **能预测的部分已经在当前分数里了。** 子代是父代的局部扰动，父代分数就是子代分数最好的单一预测量。再加的项主要是添加噪声、打乱质量排序。
4. **这个地形里几乎没有需要"先退后进"的垫脚石。** TSP 的关键骨架出生当步就刷新了前沿；续段和强制开发带来的主要是从跌落中恢复。延迟价值本来就不存在，去估计它自然没有用。

护栏则不同：它不需要跨样本统计，信息在单次调用中就被模型直接用上。它的作用是降低提议分布的方差，减少灾难性改写，而这一点在每次调用里都成立。

> 指南针要预测未来，需要大量同分布样本；护栏只需要解释过去，一次调用就够。

### 1.4 设计原则

1. **轨迹只进入生成上下文，不进入分配分数。**
2. **分配只看当前训练质量，每次评价后立即重选。** 不做续段、阶段、延迟信用，也不保护新方向。
3. **历史呈现事实。** 代码实际改动由 diff 计算，分数是实测值；模型当时写的 Idea 标为"意图"。
4. **每次 Refine 只做一处聚焦修改。** 这样护栏才有效，改进才能累积。
5. **换骨架有两条路，都是出生即评价、按质量竞争：** Explore 提出新的核心机制；Crossover 从另一分支移植一个机制。
6. **提示词要精简，不写让模型怀疑历史的规则。** V10.14 的证据规则里有一条"Do not preserve a component merely because an earlier version improved after adding it"，这条与护栏作用正好相反。
7. **交付要简单：** 一句 Idea 加一份完整程序；解析要稳健，但不做隐藏重试。

---

## 2. 取长补短：各组成的来源与证据强度

| 组成 | 取自 | 依据 | 证据强度 |
|---|---|---|---|
| Refine 带来时路 | V9.7 / V9.14 | V9.7 固定锚点配对；V9.8 配对 58/72 | 强 |
| 不放同父已试 | V9.7 | 配对 35/34，CVRP 改进率反降 | 强 |
| 不做续段、不保护新方向 | V9.14 / V9.16 | E2-B' 随机干预；V9.16 续段未产出最终最好程序 | 强 |
| 选父只看质量，不加信用项 | V9.14 / V9.16 | V9.10、V9.15、V9.19 的过程诊断 | 中 |
| 质量 Boltzmann 抽样，ESS=0.1N | V9.16（V9.19/V9.20 同） | 采用这一规则的 V9.16、V9.19 都在五任务综合排名的前三名中；尚未与 ESS=8 做配对比较 | 中 |
| 历史中的 Change 字段（由代码 diff 计算） | V9.7 | V9.7 在 CVRP 上居前；V9.16 诊断把"更丰富的实际改动历史"列为候选解释 | 中 |
| Refine / Explore 指令骨架 | V9.14 | TSP 至少两路的关键骨架明确由 V9.14 的 Explore 引入（评价 24、676） | 中 |
| Crossover 与动作配比 0.45/0.30/0.25 | V9.19 | V9.19 旧版 Crossover 严格改善率 17.5%，四类算子中最高，OP 三路的最好程序都直接来自它 | 中 |
| "改动必须能改变决策"提示 | V10.14 Tune 指引 + OBP 分析 | V10.7 在 OBP 上有 341/365 次 Refine 与父代同分 | 中 |
| 混合初始化 | V10.13 初始化对照 | CVRP 上较好，其余任务不差 | 中 |
| 独立选择集 | V10.14 | 训练前沿相对选择集系统性高估 | 中 |
| 代码规范化视图、要求不写注释 | V10.11 / V9.19 | 注释会误导（强结果分析）；也让 diff 和上下文膨胀 | 弱–中 |
| 最近一步显示完整 diff | 新增 | 第一性原理：护栏需要知道最近动了哪里 | 弱 |
| "函数可以做不止一个公式的计算"提示 | 强结果形成路径 | TSP 突破来自在选点函数内部求解剩余路径 | 弱 |
| Crossover 参考用代码差异代替行为距离 | V9.19 改写 | BehaveSim 已退役 | 弱 |
| 失败后一次付费修复 | V10.10 / V9.19 | 无对照 | 弱 |

---

## 3. 搜索机制

### 3.1 预算与记账

- **预算 B = 1000 次候选尝试。** 每一次产出候选的模型生成都计 1 次，包括：
  - 初始化和修复；
  - 解析失败、重复、已知失败；
  - 运行错误、无效输出和超时。
- **不计入预算：** 模型服务错误（连接失败、限流、服务端 5xx），这类错误自动重试。
- **另行记录：** 评价器调用数、模型调用数、输入/输出 token 和墙钟时间。重复候选和已知失败不调用评价器。

### 3.2 程序表示与去重

- **规范形式** `canonical(code)`：先解析为 AST，删除模块、函数和类的 docstring（注释在解析时自然消失），再 `ast.unparse`。
- **程序身份** `key = sha256(canonical(code))`。
- 给模型看的所有程序（当前算法、参考算法、已有根）一律用规范形式，所有 diff 也在规范形式上计算。原始响应和原始代码另行完整保存。
- **重复：** `key` 与任一有效节点相同，记为 `duplicate`。如果与本轮 Crossover 的参考程序相同，记为 `copied_reference`，单独统计复制率。
- **已知失败：** `key` 与此前评价失败的程序相同，记为 `known_failure`。
- 这三类都不评价、不建节点、不修复，但计入预算。

### 3.3 初始化

- **目标：** 得到 8 个有效且互不相同的根，初始化尝试上限 16 次。
- **前 4 个有效根**使用独立提示，只含任务和目标函数。
- **第 5–8 个根**使用参考提示：给出已有全部有效根（规范代码、分数、Idea），要求采用不同的核心决策原则。上下文不够时，从最早的根开始省略，至少保留 1 个。
- 选用哪种提示，按"当前已有几个有效根"决定，不按尝试序号决定。
- 达到上限时，只要有至少 1 个有效根就进入搜索；一个都没有，则运行以 `no_valid_root` 结束。
- 不做 bootstrap。所有根直接参与竞争。

### 3.4 父代选择

令 `q(a)` 为最大化方向的训练质量（最小化任务取 `q = −score`），`N` 为可选节点数。

```
p_β(a) = exp(β · (q(a) − q_max)) / Σ_b exp(β · (q(b) − q_max))
ESS(β) = 1 / Σ_a p_β(a)²
目标 T = min(N, max(2, 0.1 · N))
```

- `β ≥ 0` 用二分求解，使 `ESS(β) = T`：先把上界从 1 起倍增，直到 ESS 不超过 T，再二分 80 次。
- 所有质量相同时取均匀分布。如果最高分并列的节点数 `m ≥ T`，就在这 m 个节点上均匀抽。
- β 随分数尺度自动调整，不需要做任务间归一化。
- 三个动作共用同一个分布。被选次数只做记录，不进入分数。
- 标记为 `too_long` 的节点（见 4.4）不参与抽样。

### 3.5 动作选择

选定父代后独立抽取动作：**Refine 0.45、Explore 0.30、Crossover 0.25**。Crossover 找不到合格参考时改为 Refine，并记录 `crossover_fallback`。

配比取自 V9.19 在中性状态下的实际配比。V9.14/V9.16 是 Refine 0.7 / Explore 0.3、没有交叉；这里把约 0.25 从 Refine 移给 Crossover。Crossover 同样是在保留主程序框架的前提下改进，所以"在已有骨架上改进"的份额仍有 0.70。

### 3.6 Crossover 参考选择

给定主程序 `a`：

1. **候选集 E₀**：满足以下全部条件的有效节点 `b`：
   - `b ≠ a` 且 `key(b) ≠ key(a)`；
   - `q(b) ≥` 全档案 `q` 的中位数；
   - `b` 不是 `a` 的祖先，`a` 也不是 `b` 的祖先。
2. **放宽**：E₀ 为空时去掉谱系条件，得到 E₁。E₁ 也为空，则没有参考。
3. **代码差异**：`sim(a, b)` 为两份规范代码的 token 集合（NAME/NUMBER/OP）的 Jaccard 相似度。只保留 `sim ≤` 候选集相似度中位数的那一半（并列计入）。
4. 在保留下来的节点中**均匀抽取**一个。

这对应 V9.19 的"质量分位与行为距离分位都不低于中位数"，其中行为距离换成了代码差异，另外排除同一谱系，保证参考来自另一分支。参考的整体分数常常低于主程序，这是预期内的。

### 3.7 评价、失败与修复

- 候选状态分为：`delivery_failed`（没有可用代码或被截断）、`invalid_source`（语法错误或接口不符）、`duplicate`、`copied_reference`、`known_failure`、`runtime_error`、`invalid_output`、`timeout`、`valid`。
- **有效候选：** 建为新节点，父代是本轮选中的节点，同时记录动作、参考 id、Idea 和分数。下一次选择时它就与全体节点一起竞争。
- **修复：** `invalid_source`、`runtime_error`、`invalid_output`、`timeout` 这四类，只要预算还有剩余，就进行一次修复生成，计入预算。
  - 修复后的候选走同一条管线。若有效，它作为原父代的子节点建立，动作沿用原动作，并标记 `repaired`。
  - 修复失败就结束，不修第二次；修复产生的候选本身也不再修复。
  - `delivery_failed`、`duplicate`、`copied_reference`、`known_failure` 不修复。

### 3.8 最终程序选择与测试隔离

与 V10.14 相同：

1. 搜索结束后，按训练质量取前 5 个互不相同的 `key` 作为候选，同分取较早的节点。
2. 在独立选择集上评价这 5 个候选：
   - CVRP、OP 用 `val_50`（64 个实例）；
   - TSP、VRPTW、OBP 用与训练相同的生成配置、种子改为 `20260927`。
3. 取选择集分数最高者；同分取训练排名较前者。
4. held-out 测试只在选择完成后单独运行。选择集和测试集都不进入搜索提示或父代选择。

---

## 4. 上下文构造

### 4.1 各动作看到什么

| 区块 | 初始化 | Refine | Explore | Crossover | 修复 |
|---|---|---|---|---|---|
| Task + Evaluation | ✓ | ✓ | ✓ | ✓ | ✓ |
| Target Function | ✓ | ✓ | ✓ | ✓ | ✓ |
| Current Algorithm（规范代码 + 分数） | | ✓ | ✓ | ✓ | |
| 形成历史（≤8 步，最近一步给 diff） | | ✓ | | | |
| 思路列表（≤8 步，只有 Idea 与分数） | | | ✓ | | |
| 最近改动（≤3 步，只有摘要） | | | | ✓ | |
| Reference Algorithm | | | | ✓ | |
| 已有根 | 第 5–8 个根 | | | | |
| 全局最好分数 | | | ✓ | | |
| Failed Program + Error | | | | | ✓ |
| 动作指令 + Output Format | ✓ | ✓ | ✓ | ✓ | ✓ |

刻意不给的：同父已试、搜索统计、行为探针、档案卡、证据解读规则，以及训练实例的规模和数量（避免模型针对训练规模做特化）。

### 4.2 形成路径

- 当前节点 `a_k` 的形成路径是 `a_0 (根) → a_1 → … → a_k`，每一步称为一条边。
- 显示最近 `min(k, 8)` 条边，从旧到新排列。
- 若 `k ≤ 8`，在最前面加一行 Start，给出根的分数和 Idea。若 `k > 8`，写明"路径共 k 步，显示最近 8 步"，让模型知道谱系深度。
- Crossover 产生的边标为 `Crossover with an algorithm scoring <参考分数>`。

### 4.3 每条边的呈现

````text
Step i · <Action> · score <父代分数> → <子代分数> (<improved | worse | same score>)
  Idea: <当时的 Idea，≤300 字符>
  Change: <改动摘要>                         ← 除最近一步外
  Code diff (previous → current):            ← 仅最近一步
  ```diff
  <unified diff>
  ```
````

- **分数：** 原任务单位，6 位有效数字。
- **判定：** 容差 `1e-9 · max(1, |父代分数|)`，并按任务方向判断 improved / worse / same score。
- **Change 摘要**（沿用 V9.7 格式，≤400 字符）：
  - 若两份规范代码只在数值常量上不同（把数值常量全部掩码后 AST 相同），写 `numeric constants only, in <函数>: 0.15 → 0.12; 3 → 4`。数值对按源码位置排序，最多列 4 对。
  - 否则写 `+A/−R lines in <改动的顶层函数，最多 4 个>; removed: <第一行> | <最后一行>; added: <第一行> | <最后一行>`，每行最多 110 字符。
- **最近一步的 diff：** 在规范形式上计算，上下文 2 行，最多 60 行，超出部分写 `… (N more diff lines not shown)`。

### 4.4 Token 预算与裁剪

- 输入上限 24,320 token（按服务端 tokenizer 精确计数），输出上限 8,192 token。
- 形成历史区块单独限 3,000 token，初始化时的已有根区块限 8,000 token。
- 超限时按以下顺序裁剪：
  1. 从最旧的一步开始删历史，至少保留 1 步；
  2. 最近一步的 diff 改为摘要；
  3. Crossover 删去主程序的最近改动区块；
  4. Crossover 仍然超限时，改为 Refine 并记录 `crossover_context_fallback`；
  5. 最小提示（任务、目标函数、当前程序、指令、输出格式）仍然超限时，把该节点标为 `too_long`，退出父代抽样，重新抽父代。这一步不计预算。

---

## 5. 提示词

### 5.1 写法原则

- **短。** 只保留任务、目标函数、程序、历史、指令和输出格式。V9.14 的固定指令与格式文本约 650 字符；V10.14 仅证据解读规则一项就约 900 字符，加上改写范围、Idea 要求、交付格式和 JSON 证据，固定文本超过 2,000 字符。而 V9.14 的 TSP held-out 在各版本中最好。
- **用方括号小节标题（V9 风格），不用 JSON。** 模型读的是自然文本。
- **事实和意图分开标注：** Score 是实测值，Change 和 diff 由代码计算，Idea 是当时的意图。
- **指令写"做什么"，用正面表述。** 每个动作只有 4–5 条要点，每条都对应一条证据。
- **不写让模型怀疑历史的规则，也不塞诊断量。**
- **所有提示保持同一套词汇：** Score、Idea、Change、Step、Current Algorithm。
- **用英文，与此前各版本一致。**

以下模板中 `{…}` 为占位符。区块之间空一行。

### 5.2 公共片段

**Task 与 Evaluation**

```text
[Task]
{task_description}

{design_notes}                                   ← 仅当任务定义了它（目前只有 OP）

[Evaluation]
Each candidate program is run on a fixed set of training instances.
Score: {score_meaning}. {Lower|Higher} is better.
The whole evaluation must finish within {timeout} seconds, so keep the computation efficient.
```

| 任务 | score_meaning | 方向 | 训练评价 timeout |
|---|---|---|---|
| tsp_construct | the average length of the constructed tours | Lower | 20 |
| cvrp_aco | the average total length of the best routes found by the ant colony | Lower | 120 |
| op_aco | the average total prize of the best tours found by the ant colony | Higher | 60 |
| online_bin_packing | the average number of bins used | Lower | 30 |
| vrptw_construct | the average total travel distance of the constructed routes | Lower | 30 |

`task_description`、`design_notes` 和 `template_program` 取自冻结的任务契约，原文不改。给出时限，是因为超时属于失败，这是模型设计时需要知道的事实。

**Target Function**

````text
[Target Function]
```python
{template_program}
```
Keep the function name, arguments and return contract exactly as shown. The program must be self-contained: include every import, constant and helper it uses.
````

**Output Format**（`{what}` 按动作替换，见各节）

````text
[Output Format]
Reply with exactly one Idea line followed by one Python code block:
Idea: <one sentence, at most 300 characters, describing {what}>
Code:
```python
<the complete program>
```
Write no comments or docstrings in the code, and nothing after the code block.
````

**Current Algorithm**

````text
[Current Algorithm]
Score: {score}
```python
{canonical_code}
```
````

当前程序的 Idea 不在这里重复，它在历史的最近一步里；根节点的 Idea 在历史或思路列表的 Start 行里。

### 5.3 初始化

**独立根（第 1–4 个）**，`{what}` = `the algorithm`：

```text
[Your Task: Design an Initial Algorithm]
Design one complete, competitive algorithm for this task.
- Base it on a clear decision principle and implement that principle carefully.
- Use what the inputs make available. The function may compute more than a single formula, for example derive intermediate quantities or examine the consequences of a choice, as long as the evaluation stays within the time limit.
- Do not return a placeholder or a trivial baseline.
```

**参考根（第 5–8 个）**，在 Target Function 之后插入已有根：

````text
[Algorithms Designed So Far]
Algorithm 1 · Score {score} · Idea: {idea}
```python
{canonical_code}
```

Algorithm 2 · Score {score} · Idea: {idea}
```python
{canonical_code}
```
````

````text
[Your Task: Design Another Initial Algorithm]
Design one complete, competitive algorithm whose core decision principle differs from every algorithm above.
- Notice what the algorithms above have in common, and build yours on a different principle or on information they do not use.
- You may reuse a helpful detail, but the main idea must be different.
- Do not return a placeholder or a trivial baseline.
````

设计说明：
- 第二条要点让模型先找出已有根的共性。初始化对照发现，OBP 独立生成的根在 112 对中有 98 对行为零距，"代码不同"并不等于"思路不同"。
- "may compute more than a single formula"放宽的是设计空间，并不指定答案。它对构造类任务意味着可以前瞻，对 ACO 意味着可以构造更丰富的边特征。

### 5.4 Refine

区块顺序：Task → Evaluation → Target Function → Current Algorithm → How the Current Algorithm Was Formed → Your Task → Output Format。`{what}` = `the change you made`。

**历史区块**

````text
[How the Current Algorithm Was Formed]
These are the most recent steps on the path that produced the current algorithm, oldest first. "Change" is computed from the code; "Idea" is what was intended at the time and may not match the code exactly. Scores are measured.

Start · initial algorithm · score {root_score}
  Idea: {root_idea}

Step 1 · Refine · score {a} → {b} (improved)
  Idea: {idea}
  Change: {change_summary}

Step 2 (latest: produced the current algorithm) · Explore · score {b} → {c} (improved)
  Idea: {idea}
  Code diff (previous → current):
```diff
{unified_diff}
```
````

根节点没有历史时，这一区块改为：

```text
[How the Current Algorithm Was Formed]
The current algorithm is an initial design; no changes have been recorded yet.
Idea: {root_idea}
```

**指令（有历史时）**

```text
[Your Task: Refine]
Improve the current algorithm with one focused change.
- Build on what the history shows is working: parts introduced by improving steps are probably doing useful work, so keep them unless your change needs to alter them.
- Let the recent steps guide the next one: push further in a direction that improved the score, or correct or undo a recent change that made it worse.
- The change must be able to alter the decisions the function makes. Rescaling all scores, or applying the same monotone transform to them, leaves the chosen option unchanged.
- If the structure is sound, recalibrating a few influential parameters is a valid focused change.
- Prefer replacing or simplifying logic over stacking new layers, and leave unrelated parts of the program unchanged.
```

**指令（根节点）**：把前两条要点换成一条：

```text
- Identify the part of the algorithm that most limits the quality of its decisions, and improve that part.
```

逐条说明：
- **"one focused change"：** V9.7/V9.14 原句的精神。小步修改才能让护栏生效、让改进累积。
- **第 1 条：** 护栏本身。说"probably"而不是"must"，既让历史起约束作用，又不禁止必要的重写。
- **第 2 条：** 让模型读历史的方向，而不只是看历史。"correct or undo"覆盖了路径上那些被保留下来的退步边：V9.14 一条含 90 个节点的 TSP 最好链中，有 39 条边相对父代退步。
- **第 3 条：** 针对"改了但没改变决策"。V10.7 在 OBP 上有 341/365 次 Refine 与父代同分。对 ACO 任务，这一条同样适用于采样概率。
- **第 4 条：** 把 Tune 并进 Refine。CVRP 最后几次改进和 CALM 在 OBP 上的收益都是保结构的常数调整。
- **第 5 条：** 控制程序膨胀。W36 记录过 CVRP 程序膨胀导致上下文超限。

### 5.5 Explore

区块顺序：Task → Evaluation → Target Function → Current Algorithm → Earlier Ideas → Your Task → Output Format。`{what}` = `the new algorithm`。

**思路区块**

```text
[Earlier Ideas on This Line of Development]
Oldest first. Scores are measured; each idea is the intent stated when that version was written.
Start · score {root_score} · Idea: {root_idea}
Step 1 · Refine · score {a} → {b} (improved) · Idea: {idea}
Step 2 · Explore · score {b} → {c} (improved) · Idea: {idea}

Best score found so far in this search: {best_score}.
```

根节点时，前两行改为 `The current algorithm is an initial design.` 和 `Idea: {root_idea}`。

**指令**

```text
[Your Task: Explore]
Find a materially different way to solve this task better than the current algorithm.
- First identify the main limitation of the current approach: information it ignores, decisions it systematically gets wrong, or situations it cannot represent.
- Then change the core of the algorithm to remove that limitation: what it computes from the inputs, how it evaluates a choice before committing to it, or how it turns signals into a decision. Tuning parameters or making a small local edit is not enough.
- You may keep useful parts of the current program or start from scratch, but do not simply restate an idea listed above.
- The new algorithm must be complete and competitive on its own, and must stay within the time limit.
```

设计说明：
- **给当前程序：** 沿用 V9 的做法。V9.14 的 Explore 看得到当前代码，TSP 至少两路的关键骨架明确由它引入。V10.14 改为独立上下文的 Pivot 后，TSP 训练前沿均值为 6.19（V9.14 为 5.78）。这不能单独归因于 Pivot，但也没有证据支持独立上下文更好。
- **只给思路、不给 diff：** 历史对 Explore 的作用是避免换汤不换药，不是当护栏。
- **第 1 条先诊断局限：** 沿用 V10.11 Pivot 指令里有效的部分。
- **第 2 条的三个方向覆盖三类任务的骨架变化：** 构造类的前瞻、ACO 的边特征、OBP 的打分组合。
- **给全局最好分数：** 让模型知道"有竞争力"的标准。Refine 不给，以免诱发过大的改写。

### 5.6 Crossover

区块顺序：Task → Evaluation → Target Function → Current Algorithm → Recent Changes → Reference Algorithm → Your Task → Output Format。`{what}` = `the mechanism you transplanted and how it is integrated`。

**最近改动区块**：同 5.4 的历史区块，但标题换成 `[Recent Changes to the Current Algorithm]`，最多 3 步，全部用 Change 摘要，不给 diff。

**参考区块**

````text
[Reference Algorithm]
A different algorithm from another branch of this search.
Score: {ref_score}
Idea: {ref_idea}
```python
{ref_canonical_code}
```
````

**指令**

```text
[Your Task: Crossover]
Improve the current algorithm by transplanting one mechanism from the reference algorithm.
- Compare the two programs and find one computation in the reference that the current algorithm lacks and that addresses one of its weaknesses, for example an additional signal, a feasibility or look-ahead check, or a different way of combining terms.
- Integrate that mechanism into the current algorithm and adapt it so that it works with the existing parts. Keep the current algorithm's framework and its working components.
- The reference may score lower overall and still contain a useful mechanism.
- Do not copy the reference or return a program that is essentially one of the two inputs. The transplanted mechanism must be able to change the current algorithm's decisions.
```

设计说明：
- **"one mechanism"加"keep the framework"：** 让交叉成为保留主干的迁移，而不是两份程序的拼接。VRPTW 的前瞻模拟和 OP 的奖赏/距离组合都是这样迁入的。
- **第 3 条：** V9.19 旧版中，参考优于主程序的比例除 OBP 外只有 10.7–17.8%，交叉却仍有最高的改善率。明说这一点，模型才不会因为参考分数低而忽略它。
- **第 4 条针对复制：** V10.8 中有 353/3400 个 Fuse 与 donor 完全相同，其中 254 个复制的是更差的 donor。复制结果也会被 `copied_reference` 拦下并统计。

### 5.7 修复

区块顺序：Task → Evaluation → Target Function → Failed Program → Error → Your Task → Output Format。`{what}` = `the fix`。

````text
[Failed Program]
This program was written to improve an algorithm with score {parent_score}, but it failed during evaluation.
Idea: {idea}
```python
{failed_code}
```

[Error]
{error_text}

[Your Task: Repair]
Fix the program so that it runs correctly, while keeping its intended design.
- Change only what is needed to remove the failure.
- If the evaluation timed out, reduce the cost of the most expensive computation instead of dropping the idea.
- If the output was invalid, make sure the function returns exactly what the target function's contract requires.
````

- 初始化失败时，第一句改为 `This program was written as an initial algorithm, but it failed during evaluation.`
- `failed_code`：能解析就用规范形式，否则用原始代码。
- `error_text`：
  - `invalid_source`：写 `The program could not be used: {原因}`，如 SyntaxError 及行号、缺少目标函数、签名被改动。
  - `runtime_error`：traceback 的最后 15 行，最多 1500 字符。
  - `invalid_output`：评价器给出的原因。
  - `timeout`：写 `The evaluation did not finish within {timeout} seconds.`

修复提示不给历史和参考，只给修复所需的信息。

### 5.8 输出解析

沿用 V10.14 `edits.py` 中已经验证过的交付规则：
1. `finish_reason` 必须是 `stop`，截断的响应记为 `delivery_failed`。
2. 去掉闭合的 `<think>…</think>`；出现未闭合的思考块，记为 `delivery_failed`。
3. **Idea：** 取 `Idea:` 之后、`Code:` 或第一个代码块之前的文字，合并空白。原文完整保存，提示中最多显示 300 字符。缺少 Idea 不影响候选有效性。
4. **代码：** 取最后一个"恰好定义一次目标函数"的完整 Python 代码块。只有一个代码块、但不含目标函数时，仍把它交给修复。没有代码块时，按裸源码处理。
5. 补全模板要求的 import（`complete_template_dependencies`），再检查语法和接口（`validate_source`）。
6. 规范化，计算 `key`。

### 5.9 渲染示例（TSP，Refine）

下面是用原型渲染出的完整提示。代码与分数是示意用的，但格式、换行、规范化和 Change/diff 都由实现规则真实生成。

````text
[Task]
The Traveling Salesman Problem asks for a shortest tour that visits each node once and returns to the start. Instances are generated from node coordinates, but the constructive heuristic does not receive coordinates. At each step it receives the current node id, the destination/start node id, an array of unvisited candidate node ids, and the pairwise distance matrix, and must return the id of the next node to visit. Help me design a novel algorithm to select the next node in each step.

[Evaluation]
Each candidate program is run on a fixed set of training instances.
Score: the average length of the constructed tours. Lower is better.
The whole evaluation must finish within 20 seconds, so keep the computation efficient.

[Target Function]
```python
import numpy as np
def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    """
    Design a novel algorithm to select the next node in each step.

    Args:
    current_node: ID of the current node.
    destination_node: ID of the destination node.
    unvisited_nodes: Array of IDs of unvisited nodes.
    distance_matrix: Distance matrix of nodes.

    Return:
    ID of the next node to visit.
    """
    next_node = unvisited_nodes[0]

    return next_node
```
Keep the function name, arguments and return contract exactly as shown. The program must be self-contained: include every import, constant and helper it uses.

[Current Algorithm]
Score: 5.8654
```python
import numpy as np

def _greedy_path(start, nodes, dm):
    path, rest = ([], list(nodes))
    cur = start
    while rest:
        j = min(rest, key=lambda x: dm[cur, x])
        path.append(j)
        rest.remove(j)
        cur = j
    return path

def _two_opt(path, start, end, dm, sweeps=6):
    route = [start] + path + [end]
    for _ in range(sweeps):
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                delta = dm[route[i - 1], route[j]] + dm[route[i], route[j + 1]] - dm[route[i - 1], route[i]] - dm[route[j], route[j + 1]]
                if delta < -1e-12:
                    route[i:j + 1] = route[i:j + 1][::-1]
                    improved = True
        if not improved:
            break
    return route[1:-1]

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    if len(unvisited_nodes) <= 2:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    k = min(8, len(unvisited_nodes))
    near = unvisited_nodes[np.argsort(distance_matrix[current_node, unvisited_nodes])[:k]]
    best, best_len = (None, np.inf)
    for s in near[:3]:
        rest = [x for x in unvisited_nodes if x != s]
        p = [int(s)] + _greedy_path(int(s), rest, distance_matrix)
        p = _two_opt(p, current_node, destination_node, distance_matrix)
        L = distance_matrix[current_node, p[0]] + sum((distance_matrix[a, b] for a, b in zip(p, p[1:]))) + distance_matrix[p[-1], destination_node]
        if L < best_len:
            best, best_len = (p, L)
    path = best
    return int(path[0])
```

[How the Current Algorithm Was Formed]
These are the most recent steps on the path that produced the current algorithm, oldest first. "Change" is computed from the code; "Idea" is what was intended at the time and may not match the code exactly. Scores are measured.

Start · initial algorithm · score 6.7665
  Idea: Nearest neighbour, lightly penalising nodes far from the destination.

Step 1 · Refine · score 6.7665 → 6.1525 (improved)
  Idea: Subtract a regret term so nodes that will be expensive to reach later are taken now.
  Change: +6/−2 lines in select_next_node; removed: `back = distance_matrix[unvisited_nodes, destination_node]` | `score = d + 0.1 * back`; added: `if len(unvisited_nodes) <= 2:` | `score = d - 0.3 * regret`

Step 2 · Explore · score 6.1525 → 6.0324 (improved)
  Idea: Build a greedy path over the remaining nodes, improve it with 2-opt, and move to its first node.
  Change: +26/−7 lines in _greedy_path, _two_opt, select_next_node; removed: `d = distance_matrix[current_node, unvisited_nodes]` | `return int(unvisited_nodes[np.argmin(score)])`; added: `def _greedy_path(start, nodes, dm):` | `return int(path[0])`

Step 3 (latest: produced the current algorithm) · Refine · score 6.0324 → 5.8654 (improved)
  Idea: Start the remaining-path construction from the three nearest candidates and keep the shortest 2-opt route.
  Code diff (previous → current):
```diff
@@ -11,5 +11,5 @@
     return path

-def _two_opt(path, start, end, dm, sweeps=3):
+def _two_opt(path, start, end, dm, sweeps=6):
     route = [start] + path + [end]
     for _ in range(sweeps):
@@ -28,5 +28,14 @@
     if len(unvisited_nodes) <= 2:
         return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
-    path = _greedy_path(current_node, unvisited_nodes, distance_matrix)
-    path = _two_opt(path, current_node, destination_node, distance_matrix)
+    k = min(8, len(unvisited_nodes))
+    near = unvisited_nodes[np.argsort(distance_matrix[current_node, unvisited_nodes])[:k]]
+    best, best_len = (None, np.inf)
+    for s in near[:3]:
+        rest = [x for x in unvisited_nodes if x != s]
+        p = [int(s)] + _greedy_path(int(s), rest, distance_matrix)
+        p = _two_opt(p, current_node, destination_node, distance_matrix)
+        L = distance_matrix[current_node, p[0]] + sum((distance_matrix[a, b] for a, b in zip(p, p[1:]))) + distance_matrix[p[-1], destination_node]
+        if L < best_len:
+            best, best_len = (p, L)
+    path = best
     return int(path[0])
```

[Your Task: Refine]
Improve the current algorithm with one focused change.
- Build on what the history shows is working: parts introduced by improving steps are probably doing useful work, so keep them unless your change needs to alter them.
- Let the recent steps guide the next one: push further in a direction that improved the score, or correct or undo a recent change that made it worse.
- The change must be able to alter the decisions the function makes. Rescaling all scores, or applying the same monotone transform to them, leaves the chosen option unchanged.
- If the structure is sound, recalibrating a few influential parameters is a valid focused change.
- Prefer replacing or simplifying logic over stacking new layers, and leave unrelated parts of the program unchanged.

[Output Format]
Reply with exactly one Idea line followed by one Python code block:
Idea: <one sentence, at most 300 characters, describing the change you made>
Code:
```python
<the complete program>
```
Write no comments or docstrings in the code, and nothing after the code block.
````

这份提示约 6.8k 字符（约 1.8k token）。原型中其余动作的规模：Explore 约 5.0k 字符，Crossover 约 6.7k，参考初始化约 3.3k，修复约 4.1k，都远低于输入上限。规范化会带来一些外观变化，如 `1e9` 变成 `1000000000.0`、多出一些括号，但不改变语义。

---

## 6. 完整流程

```text
Input: task contract, evaluator, selection evaluator, LLM, budget B = 1000, seed

roots ← []
while |roots| < 8 and init_attempts < 16 and attempts < B:
    prompt ← InitPrompt(independent if |roots| < 4 else with roots)
    cand ← Generate(prompt); attempts += 1; init_attempts += 1
    outcome ← Process(cand, parent=None, action=Init)          # parse → dedup → evaluate → maybe one repair
    if outcome is a valid node: roots.append(node)
if roots is empty: stop with no_valid_root

while attempts < B:
    a ← SampleParent(archive, T = min(N, max(2, 0.1N)))         # quality-only Boltzmann, excluding too_long
    act ← Draw({Refine: .45, Explore: .30, Crossover: .25})
    ref ← PickReference(a) if act = Crossover else None
    if act = Crossover and ref is None: act ← Refine            # log crossover_fallback
    prompt ← BuildPrompt(act, a, ref)                           # trims per §4.4; may mark a too_long and resample
    cand ← Generate(prompt); attempts += 1
    Process(cand, parent=a, action=act, reference=ref)

Process(cand, parent, action, reference):
    parse; on delivery failure → record delivery_failed; return
    k ← key(canonical(code))
    if k = key(reference) → record copied_reference; return
    if k ∈ valid keys      → record duplicate; return
    if k ∈ failed keys     → record known_failure; return
    result ← Evaluate(code)
    if valid: add node(parent, action, reference, idea, code, score); return
    record failure; add k to failed keys
    if repairable and not already a repair and attempts < B:
        cand' ← Generate(RepairPrompt(...)); attempts += 1
        Process(cand', parent, action, reference) as repair

finalists ← top-5 distinct keys by training score
evaluate finalists on the selection set; best_program ← argmax selection score
```

每次生成之后立即重选父代。没有排队、没有续段，也没有待结算的信用。

---

## 7. 参数表

| 参数 | 值 |
|---|---|
| 预算 | 1000 次候选尝试（含初始化、修复、失败、重复） |
| 根 | 8 个有效且互不相同；前 4 个独立生成，后 4 个参考已有根；初始化尝试上限 16 |
| 选父 | 质量 Boltzmann，ESS 目标 `min(N, max(2, 0.1N))`；三个动作共用 |
| 动作 | Refine 0.45 / Explore 0.30 / Crossover 0.25 |
| Refine 历史 | 最近 ≤8 条边；最近一步给 diff（≤60 行），其余给 Change 摘要；区块 ≤3,000 token |
| Explore 思路列表 | 最近 ≤8 条边的 Idea 与分数，加全局最好分数 |
| Crossover | 主程序最近 ≤3 条边的摘要；参考要求质量 ≥ 中位数、非同谱系、代码相似度 ≤ 候选中位数，均匀抽取 |
| Idea 显示 | ≤300 字符；原文全保存 |
| 判定容差 | `1e-9 · max(1, |父代分数|)` |
| 修复 | 每个失败候选最多 1 次，计入预算 |
| 模型 | Qwen3.8-27B AWQ；temperature 1.0，top_p 0.95，top_k 20；thinking 关闭 |
| Token | 输入 ≤24,320，输出 ≤8,192 |
| 最终选择 | 训练前 5 个不同 key，在独立选择集上选最优 |
| 随机数 | 选父、动作、参考三路独立，都由运行种子（`repeat − 1`）派生 |

---

## 8. 运行记录与诊断

### 8.1 每次尝试的记录

- 精确的 prompt、原始响应、`finish_reason`、输入/输出 token、耗时；
- 父代 id、动作、参考 id、β、ESS、父代被抽中的概率；
- 本次显示的历史边 id，以及是否发生过裁剪、怎样裁剪；
- 解析出的 Idea（原文与显示文本）、原始代码、规范代码、`key`；
- 状态、错误信息、训练分数，是否为修复、修复的对象。

这些记录足以复原每一次决策，也可以直接作为后续"从搜索经验到模型学习"的条件偏好数据。

### 8.2 健康检查指标

这些指标只用来确认机制按设计运行，不作为成功判据。

| 指标 | 参照值 | 需要警惕的情况 |
|---|---|---|
| Refine 严格改进率 | V9.14：TSP 27.6%、CVRP 22.8%、OP 17.6%、OBP 3.8%；V10.7：TSP 7.2%、OBP 0.8% | TSP 低于 12%：护栏在这套提示下没有起作用 |
| Refine 与父代同分的比例 | V10.7 在 OBP 上 341/365 | OBP 超过 50%：改动没有改变决策 |
| 最好程序的谱系深度 | V9.14：77/115/95/47（TSP/CVRP/OP/OBP）；V9.16：15–41 | TSP、CVRP 小于 15：复利没有发生 |
| Crossover 严格改善率与复制率 | V9.19 旧版改善率 17.5%；V10.8 约 10% 与 donor 完全相同 | 复制率超过 5% |
| Explore 严格改善率，以及新骨架刷新前沿的次数 | V9.19 旧版 1.5–2.3% | 为 0 且最好谱系始终停留在初始根上 |
| 解析失败、重复、已知失败、超时、修复成功率 | — | 解析失败超过 3%；修复成功率低于 20%，说明修复不值得其成本 |
| 程序长度随时间的变化、`too_long` 次数 | W36 记录过 CVRP 膨胀 | 出现 `too_long` |

---

## 9. 不包含的机制

| 机制 | 最近使用 | 不采用的原因 |
|---|---|---|
| 续段、发展票、着陆、阶段机 | V9.11–V9.17、V10.9、V10.14 | 突破出生即兑现，续段只带来恢复（E2-B'、V9.16） |
| 增长项、延迟信用、Thompson、延续价值 | V9.8–V9.10、V9.15、V9.20 | 轨迹当指南针，见 1.3 |
| 行为距离调度、行为探针 | V9.19、V9.20、V10.14 | 距离预测不了收益；BehaveSim 已退役 |
| 同父已试进入上下文 | V10.13、V10.14 | V9.7 配对中没有增益，CVRP 反而下降 |
| 给所有算子加档案卡或 Idea 参考 | V10.12、V11.0 | V10.12 让 TSP/VRPTW 退步并触发预设门槛；只给 Idea 带不进具体计算 |
| 独立上下文 Pivot | V10.14 | V9 的 Explore 看得到当前程序，TSP 关键骨架至少两路明确由它产生 |
| Tune 作为独立算子 | V10.9–V10.14 | 没有独立证据；并入 Refine 的第 4 条要点 |
| 证据解读规则、搜索统计、JSON 证据 | V10.14 | 前者削弱护栏，后两者让提示膨胀 |
| 局部编辑（search/replace）交付 | V10.13、V10.14 | 失配会损失合法候选；完整程序更稳 |
| AST 语法组共享计数、rank+count 选父 | V10.14 | 选父只看质量；AST 只用于去重 |

---

## 10. 实现与正式实验

### 10.1 实现要点

- 新建 `traceaad/v10_15/`，不导入 `v10_14*` 的搜索逻辑。
- 可以复制并冻结 V10.14 的这些工程函数：`_delivery`、`extract_idea`、`complete_template_dependencies`、`validate_source`、评价器封装、选择集评价。
- 新增模块：
  - `canonical.py`：规范化、key、token 相似度；
  - `history.py`：路径、Change 摘要、数值对、diff；
  - `selection.py`：ESS 求解、参考选择；
  - `prompts.py`：本文 §5 的模板逐字实现；
  - `traceaad.py`：§6 的主循环。
- 单元测试至少覆盖：
  - ESS 求解（含全同分、顶部并列、N=1）；
  - 规范化和 key（注释、docstring、格式差异得到同一 key）；
  - 数值改动的识别与排序；
  - Change 摘要与 diff 截断；
  - 各动作提示的快照（逐字对比 §5 的模板）；
  - 裁剪顺序；
  - 预算记账（重复、已知失败、修复都计入）；
  - 单次修复不嵌套；
  - `copied_reference` 的识别；
  - 选择集与训练集不同。

### 10.2 正式实验协议

- **规模：** 五个任务各 4 路，种子为 `repeat − 1`，每路 1000 次候选尝试。模型与采样设置见 §7。
- **报告：**
  - 训练前沿、选择集分数、held-out 各规模成绩；
  - §8.2 的全部指标；
  - 评价器调用数、token 和墙钟时间。
- **历史比较：** 与 V9.14、V9.16、V9.19、V10.11、V10.14 比较时，要注明以下混杂因素：
  - V9 系列用的是 Qwen3.6；
  - 服务端 int4/g128 别名问题（8 月 28 日起）；
  - VRPTW 任务描述改写窗口（影响 V9.19–V10.5）。
- **判读：** 以 held-out 为主。4 路重复只能检出较大的差异，不宜把 1–2% 的均值差写成机制优势。过程指标（§8.2）用来解释结果是怎样发生的。
