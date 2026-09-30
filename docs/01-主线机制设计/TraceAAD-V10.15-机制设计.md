# TraceAAD V10.15 机制设计

2026-09-30 调整：Refine 与 Crossover 所展示的历史步骤全部使用完整代码 diff，取消历史单独的 3,000 token 限额与 60 行截断，仅按总输入预算裁剪最旧步骤；删除独立初始化中“函数可以做不止一个公式的计算”的提示。此前运行使用旧协议，已有结果不回写为新协议。

同日进一步调整：Refine 保持上述协议；Explore 用最多 4 张档案参考的 Idea 与分数替代本谱系思路历史；Crossover 展示主程序与参考程序各最近最多 4 步的完整 diff。多参考的思想触发与双侧历史的收益仍需独立检验。

首批正式实验（`batch_20260930`）结束后再调整（§10.3）：父代与 finalist 按训练分数类分配，目标 ESS 固定为 8；Explore 参考卡每个分数类至多一张；接受缺少结尾围栏的完整代码块；OBP 按实例隔离，VRPTW 写明 depot 规则，采样改为官方非 thinking 配置。

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
| 代码实际改动进入历史 | V9.7 的 Change 字段，呈现方式改写 | V9.7 在 CVRP 上居前；新版每步直接给完整 diff，呈现方式尚无独立对照 | 原来源中，新呈现方式弱 |
| Refine / Explore 指令骨架 | V9.14 | TSP 至少两路的关键骨架明确由 V9.14 的 Explore 引入（评价 24、676） | 中 |
| Crossover 与动作配比 0.45/0.30/0.25 | V9.19 | V9.19 旧版 Crossover 严格改善率 17.5%，四类算子中最高，OP 三路的最好程序都直接来自它 | 中 |
| "改动必须能改变决策"提示 | V10.14 Tune 指引 + OBP 分析 | V10.7 在 OBP 上有 341/365 次 Refine 与父代同分 | 中 |
| 混合初始化 | V10.13 初始化对照 | CVRP 上较好，其余任务不差 | 中 |
| 独立选择集 | V10.14 | 训练前沿相对选择集系统性高估 | 中 |
| 代码规范化视图、要求不写注释 | V10.11 / V9.19 | 注释会误导（强结果分析）；也让 diff 和上下文膨胀 | 弱–中 |
| 展示的每一步历史均给完整 diff | 新增改写 | 避免首尾行摘要遗漏结构改动；是否提高质量尚待检验 | 弱 |
| Explore 用多参考 Idea 与分数替代轨迹 | V10.11 rand_ctx，抽样方式改写 | CVRP 与部分装箱设置有正信号；未单独识别 Explore 效应，代码差异只是思想多样性的代理 | 弱–中 |
| Crossover 两侧各 4 步形成历史 | 新增改写 | 让模型同时理解主程序结构与参考机制的形成；未与单侧历史独立对照 | 弱 |
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
- 同一次生成连续 3 次服务错误时暂停运行并保留可恢复状态；尚未产出候选，因此不占候选预算。
- **另行记录：** 评价器调用数、模型调用数、输入/输出 token 和墙钟时间。重复候选和已知失败不调用评价器。

### 3.2 程序表示与去重

- **规范形式** `canonical(code)`：先解析为 AST，删除模块、函数和类的 docstring（注释在解析时自然消失），再 `ast.unparse`。
- **程序身份** `key = sha256(canonical(code))`。
- 给模型看的所有程序（当前算法、参考算法、已有根）一律用规范形式，所有 diff 也在规范形式上计算。原始响应和原始代码另行完整保存。
- 评价器运行规范代码，使程序身份与实际受评程序一致；补全依赖前的模型原始代码另存。
- **重复：** `key` 与任一有效节点相同，记为 `duplicate`。如果与本轮 Crossover 的参考程序相同，记为 `copied_reference`，单独统计复制率。
- **已知失败：** `key` 与此前评价失败的程序相同，记为 `known_failure`。
- 这三类都不评价、不建节点、不修复，但计入预算。

### 3.3 初始化

- **目标：** 得到 8 个有效且互不相同的根，初始化尝试上限 16 次。
- 初始化阶段的修复生成也计入这 16 次尝试。
- **前 4 个有效根**使用独立提示，只含任务和目标函数。
- **第 5–8 个根**使用参考提示：给出已有全部有效根（规范代码、分数、Idea），要求采用不同的核心决策原则。上下文不够时，从最早的根开始省略，至少保留 1 个。
- 选用哪种提示，按"当前已有几个有效根"决定，不按尝试序号决定。
- 达到上限时，只要有至少 1 个有效根就进入搜索；一个都没有，则运行以 `no_valid_root` 结束。
- 不做 bootstrap。所有根直接参与竞争。

### 3.4 父代选择

在单一训练种子、确定性评价下，训练分完全相同的程序几乎总是行为等价。把训练分相同的可选节点归为一个**分数类**。"相同"按判定容差 `1e-9 · max(1, |a|, |b|)` 比较：同一行为按不同顺序求均值时末位会不同（OP：14.704000000000002 与 14.703999999999999）；每个节点与当前类中最好的成员比较，接近的分数不会串联合并。这是行为等价的近似，离散目标上行为不同的程序也可能恰好同分而被合并。令 K 为分数类数，`q_k` 为第 k 类的最大化方向训练质量（最小化任务取 `q = −score`）：

```
p_β(k) = exp(β · (q_k − q_max)) / Σ_j exp(β · (q_j − q_max))
ESS(β) = 1 / Σ_k p_β(k)²
目标 T = min(K, 8)
```

- `β ≥ 0` 用二分求解，使 `ESS(β) = T`：先把上界从 1 起倍增，直到 ESS 不超过 T，再二分 80 次。K ≤ 8 时取均匀分布。
- 抽中一个类后，在类内均匀抽一个成员作为父代。同分变体再多也只占一个类的份额；最高分并列时次优类仍保有份额。
- 目标 ESS 取 V10.13 的质量 ESS 8，不随档案规模增长。首批实验使用 `T = min(N, max(2, 0.1N))` 按节点抽样，并在最高分并列节点数 ≥ T 时只在顶簇内均匀抽样；档案越大分配越分散（§10.3）。
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

**Explore 的多参考选择**：

1. 从有效档案排除当前程序、相同代码、空 Idea 和与当前 Idea 相同的候选。Idea 先合并空白、截取显示的 300 字符、忽略大小写后比较。
2. 优先使用与当前程序没有祖先或后代关系的候选；这类候选为空时才放宽谱系条件。
3. 相同规范代码、相同 Idea（全文合并空白、忽略大小写后比较）或同一分数类只保留一个代表，优先质量较高、同分较早者。不设质量中位数门槛：较低分程序也可以提供参考思想。
4. 贪心选择最多 4 个参考。每次选与当前程序及所有已选参考的最大 token Jaccard 相似度最低者，避免只挑出多个彼此相似的参考；并列时优先质量较高者，再用参考随机数流抽取。
5. 有几个合格参考就展示几个，不为凑满 4 个重复材料。没有参考时仍可根据当前代码执行 Explore。

这保证代码与显示 Idea 不重复，但不保证语义、行为或算法机制真正不同。Idea 卡用于启发新计算，不作为已验证机制解释；参考选择统计与实际展示 id 分别记录。

### 3.7 评价、失败与修复

- 候选状态分为：`delivery_failed`（没有可用代码或被截断）、`invalid_source`（语法错误或接口不符）、`duplicate`、`copied_reference`、`known_failure`、`runtime_error`、`invalid_output`、`timeout`、`valid`。
- **有效候选：** 建为新节点，父代是本轮选中的节点，同时记录动作、参考 id、Idea 和分数。下一次选择时它就与全体节点一起竞争。
- **修复：** `invalid_source`、`runtime_error`、`invalid_output`、`timeout` 这四类，只要预算还有剩余，就进行一次修复生成，计入预算。
  - 修复后的候选走同一条管线。若有效，它作为原父代的子节点建立，动作沿用原动作，并标记 `repaired`。
  - 修复失败就结束，不修第二次；修复产生的候选本身也不再修复。
  - `delivery_failed`、`duplicate`、`copied_reference`、`known_failure` 不修复。

### 3.8 最终程序选择与测试隔离

与 V10.14 相同：

1. 搜索结束后，按训练质量取前 5 个**不同的分数类**，每类取最早的节点作为候选。首批实验按不同 `key` 取前 5，同分变体可占满 finalist，OP rep1、rep4 与 OBP rep1 的 5 个选择分因此完全相同。
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
| 形成历史（≤8 步，每步完整 diff） | | ✓ | | | |
| 档案参考思路（≤4 个，Idea 与分数） | | | ✓ | | |
| 主程序形成历史（≤4 步，每步完整 diff） | | | | ✓ | |
| Reference Algorithm | | | | ✓ | |
| 参考程序形成历史（≤4 步，每步完整 diff） | | | | ✓ | |
| 已有根 | 第 5–8 个根 | | | | |
| 全局最好分数 | | | ✓ | | |
| Failed Program + Error | | | | | ✓ |
| 动作指令 + Output Format | ✓ | ✓ | ✓ | ✓ | ✓ |

刻意不给的：同父已试、搜索统计、行为探针、证据解读规则，以及训练实例的规模和数量（避免模型针对训练规模做特化）。Explore 不给谱系轨迹与参考完整代码；Refine 不加跨分支思路卡。

### 4.2 形成路径

- 当前节点 `a_k` 的形成路径是 `a_0 (根) → a_1 → … → a_k`，每一步称为一条边。
- Refine 显示最近 `min(k, 8)` 条边；Crossover 的主程序与参考程序分别显示最近 `min(k, 4)` 条边，各自从旧到新排列。
- 若显示范围包含根，在最前面加一行 Start，给出根的分数和 Idea；否则写明路径总步数与实际展示步数，让模型知道谱系深度。参考区块明确标注 reference algorithm。
- Crossover 产生的边标为 `Crossover with an algorithm scoring <参考分数>`。

### 4.3 每条边的呈现

````text
Step i · <Action> · score <父代分数> → <子代分数> (<improved | worse | same score>)
  Idea: <该步产生的算法的 Idea，最多 2,400 字符>
  Code diff (previous → current):            ← 每一步都展示完整 diff
  ```diff
  <unified diff>
  ```
````

- **分数：** 原任务单位，6 位有效数字。
- **判定：** 容差 `1e-9 · max(1, |父代分数|)`，并按任务方向判断 improved / worse / same score。
- **每一步的 diff：** 在规范形式上计算 unified diff，上下文 2 行，保留所有变化块和所有增删行，不按行数截断。这里的“完整”指完整的改动，不重复展示整份旧程序。
- 不再用 Change 摘要代替代码变化；模型可以直接阅读数值调整与结构改动。Idea 描述该步产生的完整算法（§5.2），是作者的陈述；分数变化不构成某一组件的因果证明。

### 4.4 Token 预算与裁剪

- 总上下文为 32,768 token。输入上限 24,320 token（按服务端 tokenizer 精确计数），输出上限 8,192 token，另留 256 token 余量。
- 形成历史不设单独 token 限额；初始化时的已有根区块限 8,000 token。
- 超限时按以下顺序裁剪：
  1. Refine 从最旧的一步开始删除整个历史步骤，至少保留最近 1 步的完整 diff；根节点没有改动历史。若仍超限，将该节点标为 `too_long`，重新抽父代，不计候选预算。
  2. Crossover 每次从仍有多于 1 步的两侧历史中，选择 token 较多的一侧删除最旧整个步骤；两侧有历史时各至少保留最近 1 步完整 diff，根节点显示无历史说明。
  3. 两侧最短历史加完整代码仍超限时，改为 Refine 并记录 `crossover_context_fallback`；再按 Refine 规则处理。
  4. Explore 从最后选中的参考卡开始删除，必要时可删完全部卡；当前程序、任务、指令与输出格式仍超限时按 `too_long` 处理。

历史裁剪只删除整个步骤，不截断 diff，也不将 diff 转换为摘要。

---

## 5. 提示词

### 5.1 写法原则（2026-09-30 重写）

- **从元提示出发。** 所有算子只提一个要求：利用提示中的信息，写出预期得分更高的算法。算子之间只在"依据哪些信息、离当前算法多远"上表达偏好：Refine 保留当前算法的核心思想加以改进，Explore 以不同的核心思想另起，Crossover 把参考算法中当前算法所缺的长处融合进来。
- **只陈述事实与目标，不写具体技巧。** 删除了"改动必须影响决策""单调变换无效""可重新标定参数"等规则：它们把设计者对某些失败的猜测写成操作指南，既不通用，也会被模型当作待办事项。V10.15-2 前 25% 预算中，Refine 有 307 次生成与祖父节点相同的程序，其中 82% 撤回的是使分数变差的一步，模型的 Idea 多数直接写 "revert"；旧提示中恰有 "correct or undo a recent change that made it worse"。在相同父代与历史下只替换提示的对照中（从 V10.15-2 档案抽取最近一步变差的 45 个父代，同一采样配置），撤回从 19 次降到 2 次；TSP 与 OBP 的 32 个父代上，有效新程序从 11 个增至 28 个，超过父代的从 9 个增至 17 个。样本较小，只支持"这句指令是主要诱因"，不排除模型也会自主回退。
- **公共事实：** Returning a previously evaluated candidate consumes an attempt without another evaluation. 这是搜索规则本身，对撤回、复制参考和原样返回同样适用，不针对某一种失败单独立禁令。
- **算子范围调整：** Refine 从"一个聚焦的改动"放宽为"保留核心思想的改进版本"，Crossover 从"移植一个机制"放宽为"融合参考算法的长处"，允许一次改动多个部分。这给模型更大的设计空间，代价是分数变化更难归因到单处改动。
- **历史是证据。** 形成历史用于说明这条开发线上哪些改动有帮助、哪些没有，而不是下一步的待办清单。
- **用方括号小节标题，英文，短。** 事实（Score、Code diff）与作者陈述（Idea）分开标注。

### 5.2 Design（原 Idea）的定义

Design 描述**所给代码中的完整算法**，而不是"这次改了什么"：它的核心思想、计算的关键量，以及这些量如何组合成每一步决策，约 100–200 个英文词的平实文字，只写最终设计，不写推敲过程、备选方案、公式和与旧版本的比较。每一步的改动由 Code diff 呈现，Design 负责说明结果算法是什么，因此历史、Explore 参考卡、Crossover 参考与修复提示中的 Design 都能独立说明一个算法。所有算子使用同一输出格式：先写 Design，再写代码。提示中显示时合并空白、最多 2,400 字符。内部字段仍名为 `idea`；解析接受 Design、Idea、Thought 三种标签，取最后一个。

**为什么这样定：**

- 首批与 V10.15-2 的 Idea 是一句话描述改动，无法独立说明算法。
- V10.15-3 最初要求在代码前写 150–250 词的 Idea：关闭 thinking 时模型把它当作草稿本，168 条中 43% 含 "Actually/Wait/Let's" 等自我修正，73% 超过 250 词（最长 1,696 词），79% 含 LaTeX 推导，并保留被放弃的方案，与代码不一致。该批在 681 次尝试时停止。
- 增加一个不保留的 Reasoning 区块后，Reasoning 本身写 700–1,300 词，输出 token 中位数升至 1,700–2,800；Idea 放在代码之后时，OP 有 4/9 回复缺少 Idea。
- 不设 Reasoning 的配对测试（TSP、OBP 各 12 个最近一步变差的 Refine 父代，外加 6 次 CVRP 初始化，同一采样配置）：Design 在前与在后均无草稿用语、无缺失，Refine 的改进（12/24 对 13/24）、撤回与重复相当；Design 在前的输出 token 中位数更低（599 对 634），7/30 含少量公式（在后为 1/30）。选择 Design 在前：模型可先定设计再写代码，也不会遗漏说明。

### 5.3 公共片段与各算子指令

公共区块依次为 `[Task]`（任务描述与 design notes）、`[Evaluation]`、`[Target Function]`；其后是各算子的上下文区块、任务指令和输出格式。

````text
[Evaluation]
Each candidate program is run on a fixed set of training instances.
Score: {score_meaning}. {Lower|Higher} is better.
The whole evaluation must finish within {timeout} seconds, so keep the computation efficient.
Returning a previously evaluated candidate consumes an attempt without another evaluation.

[Output Format]
Reply with a Design followed by one Python code block:
Design: <the design of the algorithm you will implement, in about 100-200 words of plain prose: its core idea, the key quantities it computes, and how they are combined into each decision. State only the final design; leave out deliberation, alternatives, formulas and comparisons with earlier versions>
Code:
```python
<the complete program>
```
Write no comments or docstrings in the code, and nothing after the code block.
````

| 算子 | 上下文区块 | 任务指令 |
|---|---|---|
| 初始化（前 4 个根） | — | Design a complete algorithm for this task that you expect to score well, built on a clear core idea. |
| 初始化（第 5–8 个根） | 已有根的代码、分数与 Design | Design a complete algorithm for this task that you expect to score well, built on a core idea different from those of the algorithms above. |
| Refine | 当前算法；形成历史（≤8 步，每步分数变化、Design、完整 diff） | Write an improved version of the current algorithm that keeps its core idea. Use the formation history as evidence of what has and has not helped along this line of development.（根节点无历史时只有第一句） |
| Explore | 当前算法；≤4 张参考 Design 卡；全局最好分数 | Write a new algorithm that you expect to outperform the current one, built on a different core idea. The reference designs show other approaches found in this search; draw on them as inspiration. |
| Crossover | 当前算法及其历史（≤4 步）；参考算法及其历史（≤4 步） | Write an improved version of the current algorithm by combining it with the reference algorithm: bring in what the reference does well that the current algorithm lacks, and keep the current algorithm's strengths. |
| 修复 | 失败程序（Design 与代码）；错误信息 | The program failed during evaluation. Fix it so that it runs correctly within the time limit, keeping the algorithm it was meant to implement. |

历史引导语：The steps that produced the current algorithm, oldest first. Each step shows the score change, the Design of the algorithm it produced, and the code diff from the previous version.

### 5.4 逐字文本

实际发送给模型的完整提示以测试快照为准：`tests/snapshots/v1015/` 下的 `init_independent`、`init_reference`、`refine_root`、`refine_history`、`explore`、`crossover`、`repair`，由 `tests/method/test_traceaad_v1015_prompts.py` 逐字比对。

### 5.5 输出解析

沿用 V10.14 `edits.py` 中已经验证过的交付规则：
1. `finish_reason` 必须是 `stop`，截断的响应记为 `delivery_failed`。
2. 去掉闭合的 `<think>…</think>`；出现未闭合的思考块，记为 `delivery_failed`。
3. **Idea：** 取 `Idea:` 之后、`Code:` 或第一个代码块之前的文字，合并空白。原文完整保存，提示中最多显示 300 字符。缺少 Idea 不影响候选有效性。
4. **代码：** 取最后一个"恰好定义一次目标函数"的完整 Python 代码块。只有一个代码块、但不含目标函数时，仍把它交给修复。没有代码块时，按裸源码处理。只有一个开头围栏而没有结尾围栏时，若围栏之后的内容能解析且恰好定义一次目标函数，按完整代码接受（响应已以 `stop` 结束）；Idea 也以该开头围栏为终点。
5. 补全模板要求的 import（`complete_template_dependencies`），再检查语法和接口（`validate_source`）。
6. 规范化，计算 `key`。


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
    refs ← PickExploreReferences(a, max=4) if act = Explore else []
    prompt ← BuildPrompt(act, a, ref, refs)                     # Crossover: histories ≤4 steps each; trims per §4.4
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
| 选父 | 分数类上的质量 Boltzmann，ESS 目标 `min(K, 8)`，类内均匀；三个动作共用 |
| 动作 | Refine 0.45 / Explore 0.30 / Crossover 0.25 |
| Refine 历史 | 最近 ≤8 条边；每一步给完整 diff；只按总输入上限从最旧步骤开始裁剪 |
| Explore 参考 | 最多 4 个档案 Idea 与分数；优先非祖先/后代、显示 Idea 与代码去重，再按代码差异贪心选择；加全局最好分数，不给谱系轨迹 |
| Crossover | 主程序和参考程序各最近 ≤4 条边的完整 diff；参考选择仍要求质量 ≥ 中位数、优先非祖先/后代、代码相似度 ≤ 候选中位数，均匀抽取 |
| Idea | 描述完整算法，约 150–250 词；显示最多 2,400 字符，原文全保存 |
| 判定容差 | `1e-9 · max(1, |父代分数|)` |
| 修复 | 每个失败候选最多 1 次，计入预算 |
| 模型 | Qwen3.8-27B AWQ；thinking 关闭；官方非 thinking 采样 temperature 0.7、top_p 0.8、top_k 20、min_p 0、presence_penalty 1.5、repetition_penalty 1.0，全部显式发送 |
| 评价时限 | 墙钟超时。训练：TSP/VRPTW/OBP 30 s、OP 60 s、CVRP 120 s；选择：构造类为训练的 2 倍，ACO 按实例数放大；held-out：TSP 3,000 s、VRPTW/OBP 1,000 s、ACO 3,600 s；BLAS/OpenMP 单线程 |
| Token | 输入 ≤24,320，输出 ≤8,192 |
| 最终选择 | 训练前 5 个不同分数类（每类最早节点），在独立选择集上选最优 |
| 随机数 | 选父、动作、参考三路独立，都由运行种子（`repeat − 1`）派生 |

---

## 8. 运行记录与诊断

### 8.1 每次尝试的记录

- 精确的 prompt、原始响应、`finish_reason`、输入/输出 token、耗时；
- 父代 id、动作、参考 id、β、ESS、父代被抽中的概率；
- 本次显示的主程序历史边 id、参考程序历史边 id、Explore 参考个体 id，以及是否发生过裁剪、怎样裁剪；
- 解析出的 Idea（原文与显示文本）、原始代码、规范代码、`key`；
- 状态、错误信息、训练分数，是否为修复、修复的对象，子代是否与父代同分（`same_as_parent`）；
- 每次评价的墙钟秒数与 CPU 秒数。

`diagnostics.json` 另外汇总各动作刷新训练前沿的次数（`new_frontiers_by_action`）、分数类数与最大类规模，以及有效评价 CPU 时间的中位数、p95 和最大值。

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
| 给所有算子统一加档案卡 | V10.12、V11.0 | V10.12 在保留轨迹的同时给所有算子追加两张卡，TSP/VRPTW 退步并触发预设门槛；V11.0 同时改分配与上下文，不能单独识别参考效果。这些结果不排除仅给 Explore/Crossover 提供参考，也不证明 Idea 无法触发有用计算 |
| 独立上下文 Pivot | V10.14 | V9 的 Explore 看得到当前程序，TSP 关键骨架至少两路明确由它产生 |
| Tune 作为独立算子 | V10.9–V10.14 | 没有独立证据；并入 Refine 的第 4 条要点 |
| 证据解读规则、搜索统计、JSON 证据 | V10.14 | 前者削弱护栏，后两者让提示膨胀 |
| 局部编辑（search/replace）交付 | V10.13、V10.14 | 失配会损失合法候选；完整程序更稳 |
| AST 语法组共享计数、rank+count 选父 | V10.14 | 选父只看质量；AST 只用于去重 |

上下文选择仍有未解决的问题。原协议的“主程序最近 3 步、参考无历史”与新协议“两侧各 4 步”都没有独立收益对照；主程序历史用于保留结构，参考历史可能帮助理解可迁移机制，两者都不能仅凭历史分数变化判断组件的因果贡献。当前协议采用后者，历史实验结果仍按各自原协议解释。

V10.11 `rand_ctx` 曾以最多 8 张无顺序的 Idea+fitness 档案卡替代形成历史，Fuse 另保留完整 donor。历史实现按质量排名 softmax 无放回抽取，未保证行为或机制多样性。[三重复结果](../02-实验结果/03-辅助与分支版本结果.md)显示 CVRP 与部分装箱设置较好、TSP/VRPTW 较差；这是整个上下文方案的结果，不能单独归因于某个算子。短 Idea 或关键词可能触发模型已有知识并生成新的计算；这种语义触发机制与直接移植参考代码都值得保留为竞争解释，需检查实际生成和固定条件对照，不应由成功案例或文本复制率直接断言。

---

## 10. 实现与正式实验

### 10.1 实现要点

- 新建 `traceaad/v10_15/`，不导入 `v10_14*` 的搜索逻辑。
- 可以复制并冻结 V10.14 的这些工程函数：`_delivery`、`extract_idea`、`complete_template_dependencies`、`validate_source`、评价器封装、选择集评价。
- 新增模块：
  - `canonical.py`：规范化、key、token 相似度；
  - `history.py`：路径、完整 diff；旧 Change 与数值对辅助函数保留，但不进入生成提示；
  - `selection.py`：ESS 求解、Crossover 单参考选择、Explore 多参考选择；
  - `prompts.py`：本文 §5 的模板逐字实现；
  - `traceaad.py`：§6 的主循环。
- 单元测试至少覆盖：
  - ESS 求解（含全同分、顶部并列、N=1）；
  - 规范化和 key（注释、docstring、格式差异得到同一 key）；
  - 数值改动的识别与排序；
  - 多步完整 diff、超过 60 行的 diff 与超过 3,000 token 的历史保留；
  - Explore 参考去重与谱系筛选、没有轨迹的参考卡提示、实际展示 id 的记录；
  - Crossover 两侧各 4 步历史、独立的边 id 与超限时的双侧裁剪；
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
  - V10.15 提示中的时限与训练评价器一致（OBP、VRPTW 为 30 秒）；现行 V10.14 运行入口为 30 秒，两版本一致。
- **判读：** 以 held-out 为主。4 路重复只能检出较大的差异，不宜把 1–2% 的均值差写成机制优势。过程指标（§8.2）用来解释结果是怎样发生的。

### 10.3 首批实验后的修正（2026-09-30）

首批 20 路（`batch_20260930`）在旧协议下运行；以下修正之后的运行不与其混为同一协议。数字来自 16 路已完成运行，细节见 [2026-W40 周报](../06-总结与周报/2026-W40.md)。

| 现象 | 证据 | 判断 |
|---|---|---|
| TSP 超时大多是假的 | 30 个超时程序在空闲主机上重测，20 个在 20 秒内完成；当时成功的对照程序同样被拖慢约 2.5 倍 | 墙钟计时受并发负载影响：20 路搜索同时运行，CPU 密集进程多于核数，且 numpy 未限制 BLAS 线程 |
| OBP 分数依赖实例顺序 | rep3 选中的程序用全局变量累计物品统计；同一程序换 24 种实例顺序，箱数在 725.25–726.75 之间变化 | 评价器用同一次 exec 的函数依次评价所有实例，状态跨实例残留 |
| VRPTW 有 159 次无效输出 | 全部是"在 depot 时又返回 depot"；模板说明写的是"返回下一个节点，或返回 depot 开始新路线"，未说明在 depot 时不可返回 depot | 接口说明有歧义，模型只能在失败后从修复提示得知规则 |
| 父代抽样退化 | Refine、Crossover 分别有 26%、35% 的有效子代与父代同分，OBP 最高达 68%；每路 800–900 个节点只有 67–164 种分数；OP rep4 最高分并列 216 个 | 同分变体作为新节点进入档案，形成大并列簇；V10.15 在并列节点数 ≥ 目标 ESS 时退化为顶簇内均匀抽样，而目标 ESS 为档案规模的 10%，档案越大分配越分散 |
| finalist 失去比较对象 | OP rep1、rep4 和 OBP rep1 的 5 个 finalist 选择分完全相同 | finalist 只按代码 key 去重，同分变体可以占满 top-5 |
| 完整代码因缺少结尾围栏被拒 | 69 次 `delivery_failed`，模型以 `stop` 结束、代码完整，只缺结尾的 ``` | 交付规则过严 |
| 以"比父代好"衡量 Explore 失真 | 按父代比较，Explore 的改进率比 Refine 低约 7 倍；按刷新训练前沿，只低约 2.5 倍（4.8 对 12.1 次/千次），并贡献了 157 次刷新中的 24 次 | 父代按质量挑选，Explore 的子代离开父代骨架，天然难以超过父代 |

**搜索机制：** 父代与 finalist 按分数类分配（§3.4、§3.8），Explore 参考卡按分数类去重（§3.6），放宽缺少结尾围栏的交付（§5.5），并补充诊断（§8.1）。

**评测协议（对所有方法生效，不算 V10.15 的机制贡献）：**

#### 评价时限

超时仍按墙钟时间判定，不针对具体主机的 CPU 做计量。首批的 TSP 超时明显受主机负载影响：30 个超时程序在空闲主机上重测，有 20 个在 20 秒内完成，当时成功的对照程序也被拖慢约 2.5 倍。曾尝试改用 CPU 时间计量，但在这台混合架构主机（i9-14900KF）上，满载时同一程序的 CPU 时间也会膨胀 4–5 倍（降频与超线程争用），并不比墙钟更稳定，因此未采用。实验入口在导入 numpy 前把 BLAS/OpenMP 限为单线程，避免每个评价进程再开线程池；评价进程另外记录 `cpu_seconds` 作诊断。

时限按程序的真实计算量确定，而不是按超时率。空闲主机上的实测（训练规模 n=50）：

| 任务 | 参考实现 | 首批选中程序 | 训练时限 |
|---|---|---|---|
| TSP | O(n) 0.02 s；O(n²) 向量化 0.03 s；O(n²) 纯 Python 0.14 s；每步对每个候选模拟补完路径（O(n³)）1.4 s | 1.5–4.4 s | 20 → **30 s** |
| VRPTW | O(n³) 补完 0.7 s | 0.07–0.27 s | 30 s |
| OBP | O(bins) 0.12 s；O(bins log bins) 0.44 s | 0.4–1.0 s | 30 s |
| OP | 模板 1.1 s（蚁群固定开销为主） | 1.3–1.5 s | 60 s |
| CVRP | 模板 6.4 s（蚁群固定开销为主） | 5.4–6.3 s | 120 s |

- **训练：** 时限覆盖 O(n³) 级别的逐步前瞻，并给最慢的选中程序留出至少约 5 倍的负载余量。TSP 原先的 20 s 只有 4.5 倍余量，改为 30 s，与 VRPTW、OBP 一致；该值写在任务配置中，对所有方法生效。
- **选择：** 构造类任务取训练时限的 2 倍。首批 TSP 有 5 个 finalist 在选择阶段以 20.0–20.1 s 超时，它们都曾在搜索中通过同样的时限。ACO 的选择时限仍按实例数放大（768 s）。
- **held-out：** 只衡量解的质量，时限只防止挂起，并与共享基线评价器一致：TSP 3,000 s，VRPTW 与 OBP 1,000 s，CVRP 与 OP 3,600 s，不再随规模放大。原先按 (n/50)² 放大（TSP 20/80/320 s）过紧：O(n³) 参考实现从 n=50 到 n=200 耗时增长约 136 倍（1.4 → 194 s），VRPTW 约 105 倍（0.7 → 75 s）；OBP 在 1 万件物品规模上约为训练的 9 倍，却一直只有 30 s；ACO 在 test_200 上以 8 个 worker 运行已达 697 s。

报告时仍应给出超时率，并把它视为受运行负载影响的量。

#### OBP 实例隔离

评价每个实例前重新执行一次候选程序，模块级状态不能跨实例保留，分数也就不再依赖实例顺序。不使用全局状态的程序，分数与之前完全一致。

#### VRPTW 接口说明

模板和任务描述写明：调用时 `unvisited_nodes` 从不为空；只有车辆不在 depot 时才能返回 depot 结束当前路线，在 depot 时必须返回一个客户。评价器的判定规则本身不变。

**可比性：** 模板文本改变后，VRPTW 结果不能与旧模板下的历史结果直接比较，报告时需注明模板版本。

**采样参数：** 搜索均关闭 thinking，但此前所有运行使用的是官方 thinking 模式的采样参数（temperature 1.0、top_p 0.95、无 presence penalty）。现默认改为官方非 thinking 模式配置：temperature 0.7、top_p 0.8、top_k 20、min_p 0、presence_penalty 1.5、repetition_penalty 1.0。这是对所有方法生效的协议变化，与历史结果比较时需要注明；presence_penalty 会惩罚已出现过的 token，而代码天然会重复标识符，因此小规模验证要同时检查有效率和交付失败率是否下降。

**模型服务：** local（llama.cpp，GGUF UD-Q4_K_XL，KV q8_0，MTP 投机解码）与远程（vLLM 0.28.0，AWQ INT4）的排查结果：

- 聊天模板渲染和分词完全相同，上下文均为 32768；
- 客户端此前只发送 temperature、top_p、top_k，其余采样参数由两种服务各自取默认值，现已全部显式发送；
- 同一个 TSP 提示各采样 40 次，local 有 11 次代码块未闭合，server3 为 0 次。关闭 MTP 后 local 仍有 4/18，而且真实 logprob 显示，模型在最后一行代码后直接给结束符 0.06–0.99 的概率，远程同一位置约为 0.0003。因此差异来自量化后的模型数值（GGUF 权重，可能还有 KV 量化），而不是投机解码。§5.5 的放宽规则消除了它对交付的主要影响，但两种服务的输出分布并不相同；正式实验应只使用同一种权重和服务。

**验证计划：** TSP、OBP 各 2 路小规模运行，记录超时率与运行时的主机负载，检查有效率与交付失败率在新采样下是否变差、父代分配 ESS 是否稳定在约 8、finalist 是否不再全部并列；通过后再做五任务正式实验。本次同时改动了提示词（文首）、分配机制、评测协议和采样参数，正式结果只能评价整体，不能归因到单项。
