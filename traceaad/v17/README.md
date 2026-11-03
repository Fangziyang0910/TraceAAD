# TraceAAD V17

研究问题与机制设计见[独立方法说明](../../docs/机制设计/V17.md)。

V17 以同龄表现分配谱系开发机会，并将有来源证据的局部改动作为跨谱系共享单位。生成上下文展示修改及其结果；交叉优先导入其他谱系中已经带来改进的组件。

| 部分 | 解决的问题 | 做法 | 开关 |
| --- | --- | --- | --- |
| 同龄比较 | 自身仍在进步，不足以说明继续投入优于其他方向 | 冠军以外的谱系在检查点上不差于冠军谱系在相同投入时的成绩，才继续开发 | `projection=False` |
| 组件导入 | Crossover 的参考程序以整段设计出现，回复容易照搬或改不动 | 把另一谱系中带来改进的一步修改，以修改说明和代码差异的形式导入当前谱系 | `imports` |

默认 `imports=True`、`changes=True`、`foreign_reference=True`、`projection=False`、`progress_rule=False`。关闭 `imports` 后，交叉使用其他谱系的设计描述；可选的 `progress_rule` 按近期自身进步分配机会，具体规则见下文。

开发机会与修改粒度的轨迹分析见 [allocation.json](../../experiments_result/analysis/v16_allocation_20261028/allocation.json) 与 [counterfactual.json](../../experiments_result/analysis/v17_progress_20261028/20261028_1356/counterfactual.json)，研究判断保存在 [W44 记录](../../docs/记录/2026-W44.md)。

## 同龄比较

谱系由无父代程序及其沿主父代关系产生的后代组成；投入为已建立的程序记录数，失败、修复和回访均计入。持有全局最好程序的谱系为冠军。首个检查点 `first = 1 + mature_steps`（默认 5），之后逐次加倍。冠军以外、有有效程序的谱系：投入少于 `first` 时处于发育期，活跃；否则取它已到达的最高检查点 `r`，若前 `r` 条记录的最好成绩不差于冠军前 `min(r, 冠军投入)` 条记录的最好成绩，活跃，否则暂停。下一次尝试给投入最少的活跃挑战谱系，没有合格挑战者时给冠军。发育期执行 `Mature`；自上次新最优或上次发散起连续 `stagnation_window` 次没有新最优时发散。

## 进步规则（`progress_rule=True`）

记号：首个检查点 `first = 1 + mature_steps`（默认 5）；`W = stagnation_window`（默认 40）；谱系的**停滞数**是它在首次达到自身最好成绩之后的尝试数。

每次尝试按以下顺序分配：

1. **发育。** 有有效程序、投入少于 `first` 的谱系先执行 `Mature`，投入最少的先做。冠军也不例外。
2. **探索是否打开。** 从当前全局最好程序之后的记录算起，冠军谱系的尝试数为 `c`，其他谱系的为 `o`。`o < c − W` 时探索打开，否则这次尝试给冠军。因此每次出现新全局最好后，冠军先得到 `W` 次尝试；之后最多每两次尝试有一次给其他谱系。
3. **停滞后发散。** 探索打开时，若距上次全局改进和上次发散都已有 `W` 次尝试，且 `diverge=True`，这次尝试用只含描述的设计地图请求新的起点。
4. **活跃谱系。** 否则，若冠军以外有活跃谱系，尝试给停滞数最小的一个（再按投入少、谱系早排序）。谱系活跃的条件是停滞数小于 `max(first, 投入/2)`，即它的最好成绩出现在后一半尝试中。这条规则不与冠军比较，也不估计潜力。
5. **恢复。** 没有活跃谱系且 `imports=True` 时，看暂停的谱系：若某个成绩比它好的谱系在它最后一次尝试之后产生了新组件，就把这个组件导入它的最好程序。多个暂停谱系符合时，成绩最好的先恢复。导入使它改进，它就重新活跃；否则它要等下一个新组件。
6. **新谱系。** 以上都不满足且 `diverge=True` 时，也发散；否则给冠军。

冠军谱系内的尝试、活跃谱系内的尝试，都按操作权重抽样，父代只在该谱系的有效程序中按质量与直接改进经验抽样。发散只在探索打开后发生（第 3、6 步），所以不会占用冠军在每次新全局最好之后的 `W` 次尝试。

## 组件导入

**组件。** 一个有效程序 `p` 的修改是组件，当且仅当：

- `p` 刷新了所在谱系的最好成绩；
- 它由 Refine、Explore、Crossover、Mature 或 Import 从本谱系的一个有效程序 `q` 产生（修复后的程序按失败版本的出发点和操作计）；
- 修改的行数（删去加新增）不超过 `COMPONENT_LINES = 60`；
- 相对谱系原最好成绩的改进比例，不低于满足上述条件的修改的中位数。

**导入。** 任一谱系抽到 Crossover 时，若其他谱系有本谱系尚未导入过的组件，这次尝试改为 `Import`：导入其中来源程序成绩最好的组件，同成绩取较新的。没有可导入的组件时，通过其他谱系的设计描述交叉。按来源程序成绩排序，避免只因粗糙设计的一次较大相对改善而反复偏向它。每个组件向同一目标谱系至多导入一次，失败也计入尝试。

**提示。** Import 提示提供形成路径、此前尝试、当前算法的设计与代码。之后是 `[Component From Another Algorithm]`：来源父子程序的成绩、修改说明和代码差异 `diff(q, p)`。来源算法不作为完整参考程序单独展示。任务要求将组件中的计算适配到当前代码，保留当前算法的主决策规则。回复写出 `change` 字段。回访拒绝的重问仍是 Import 提示。

## 记录

`request.allocation` 默认记录 `lineage`、`champion`、`effort`、`checkpoint`、`projected`（`projection=False` 时总为假）。打开 `progress_rule` 时记录 `lineage`、`champion`、`effort` 和 `reason`（`young`、`champion`、`active`、`resume`、`diverge`）；探索判断之后的分配另记 `account`（`c` 与 `o`），活跃谱系记 `stall`，恢复记 `component`，发散请求的 `lineage` 为空。Import 程序的 `reference_id` 是组件程序。

`summary.json` 的 `v15` 字段保存谱系分配统计，`v17` 保存两个开关的取值、各分配原因的首次尝试数（只在 `progress_rule=True` 时有，否则为空）、组件数、Import 次数及其中改进父代的次数、恢复次数及其中改进的次数（只在 `progress_rule=True` 时可能非零）、导入结果与组件程序的代码距离中位数，以及距离低于 `map_distance` 的次数（用来看导入是否变成照搬）。父代选择的经验计数只统计 Refine、Explore、Crossover，不含 Mature 与 Import。

```bash
.venv/bin/python -m experiments.traceaad_v17.run --task tsp_construct --dry-run
.venv/bin/python -m experiments.traceaad_v17.launch_host \
  --batch <batch> --suite co6 --repeats 3 --budget 1000 \
  --backends server3 --eval-workers 8 \
  --scheduler-socket /tmp/traceaad-1003/scheduler.sock
```

## Python 包入口

本目录自带本版的配置、搜索、提示、选择、个体、评价、历史、状态、存储、执行限制与 `runtime/` 运行实现。公开入口为 `Config`、`Program`、`TraceAAD`、`Evaluation`、`LLM`、`OpenAIAPI`，原有类名 `TraceAADV17` 也可使用。任务评价对象与模型客户端由调用方传入；本包提供源码解析、模型请求、隔离执行和调度客户端。运行需要 Python 3.14 与 NumPy、OpenAI、Requests 等第三方依赖。

```python
from traceaad.v17 import Config, TraceAAD

search = TraceAAD(evaluation=task, llm=client, run_dir="runs/example", config=Config())
summary = search.run()
```

`loop.py` 负责生成、评价、修复与检查点，`search.py` 和包内辅助模块组织本机制。各模块使用包内相对导入；复制整个目录并改名后可作为独立 Python 包导入。实验入口位于 `experiments/traceaad_v17/`，自带运行准备、模型路由、批次启动与存储工具；标准实验从 `benchmarks` 读取任务和数据。
