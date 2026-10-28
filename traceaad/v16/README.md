# TraceAAD V16

V16 在 [V15](../v15/README.md) 上改了两处：比较点加入乐观估计，上下文改为“修改说明 + 跨谱系参考程序”。其余规则与默认值都与 V15 相同。论文中的方法即 V16，见[论文第 3 节](../../docs/论文：TraceAAD.md#3-traceaad)；论文称 V15 为“早期实现”。

## 乐观估计

V15 中，挑战谱系 D 在比较点 r（5、10、20……）只有一条晋级条件：前 r 次尝试的最好成绩不差于冠军谱系（论文中称参考谱系）前 r 次尝试的最好成绩。仍在快速改进的谱系可能在 r 处落后，暂停后就不再得到开发。

V16 增加第二条条件。两条满足任一条，D 就继续：

1. **同龄比较**（V15）：`b_D(r) ≤ b_C(min(r, e_C))`；
2. **乐观估计**：`b_D(r) − s_D · r ≤ b_C(min(2r, e_C))`。

其中 `b(e)` 是前 `e` 次尝试中的最好有效成绩，`e_C` 是冠军谱系当前的尝试数。`s_D` 是 D 在窗口 `[h, r]` 上每次尝试的平均改进，`h = max(首个有效尝试的年龄, ⌈r/2⌉)`；`h = r` 时 `s_D = 0`，第二条退化为与冠军在 `2r` 处的成绩比较。

窗口从 `r/2` 开始，是为了不把粗糙初版到第一次修改之间的大幅跳变算进斜率。V14 与 V15 轨迹上的校准见 [slope_projection.json](../../experiments_result/analysis/v15_20261027/slope_projection.json)：半窗口在 V15 的 215 个比较事件中多放行 39 个（同龄比较放行 32 个），全窗口多放行 76 个。

## 上下文

V15 的轨迹每一步都给完整设计描述，修改本身埋在大段相同的文字里；Crossover 的参考程序多来自同一谱系，回复常与参考程序成绩相同。统计见 [context_reuse.json](../../experiments_result/analysis/v16_context_20261027/context_reuse.json)（论文表 B9–B10）。V16 的改动：

1. **修改说明**（`changes=True`）。修改操作的回复在 `design` 之后多写一个 `change` 字段：相对修改对象改了哪些计算或参数值，改前与改后，至多 60 词。形成路径各步、修改对象此前的尝试和全局最好成绩的改进记录都以 `Change` 加结果出现；修改对象、路径起点、参考程序和无父代的程序仍以 `Design` 出现，修改对象的 `Design` 写在它的代码之前。此前尝试的列表注明这些程序已经评价过、不要重复。修复程序沿用原程序的 `change`。
2. **Crossover 指令**（随 `changes`）。要求保留当前算法的主决策规则，只把参考设计中的计算实现到当前代码中。
3. **跨谱系参考程序**（`foreign_reference=True`）。参考程序在其他谱系的有效程序中按原规则（成绩不劣于中位数，均匀抽取）选择；没有其他谱系的有效程序时用全档案。

## 开关与记录

`projection`、`changes`、`foreign_reference` 默认都为 `True`。三者都为 `False` 时搜索与 V15 完全相同（测试 `test_a_slow_challenger_pauses_and_without_projection_the_search_is_v15`）。只关 `projection` 即论文的“无乐观估计的 TraceAAD”。

程序记录多了 `change` 字段（V14、V15 的记录中为空）。`request.allocation` 在 V15 的字段之外记录 `projected`：该谱系是否只因乐观估计而保持活跃。`summary.json` 沿用 V15 的 `v15` 字段。

```bash
.venv/bin/python -m experiments.traceaad_v16.run --task tsp_construct --dry-run
.venv/bin/python -m experiments.traceaad_v16.launch_host \
  --batch <batch> --suite co6 --repeats 3 --budget 1000 \
  --backends server3 --eval-workers 8 \
  --scheduler-socket /tmp/traceaad-1003/scheduler.sock
```
