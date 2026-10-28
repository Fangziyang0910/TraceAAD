# TraceAAD V15

V15 针对[适应度引导的保守化](../../docs/论文：TraceAAD.md)。LLM 算子照着提示中看到的内容写，因此选择有两条通道：繁殖（修改哪个程序）与**曝光**（算子看到什么）。V15 在 V14 上加入一个曝光策略，用两条规则同时控制这两件事。V14 的回访拒绝、设计地图和发育保留，并成为策略中的部件。

| 规则 | 回答的问题 | 做法 | 开关 |
| --- | --- | --- | --- |
| 一：等投入分配 | 谁被曝光 | 投入按谱系分配；挑战谱系在相同投入下与冠军谱系比较 | `allocation=effort`（`fitness` 为 V14） |
| 二：信息形式曝光 | 以什么形式曝光 | 只有被修改的程序给代码；其他程序给设计描述与评价结果 | `exposure=info`（`code` 为 V14） |

`allocation=fitness` 与 `exposure=code` 同时使用时，提示与动作和 V14 完全相同（测试 `test_with_both_rules_off_the_search_is_v14`）。

## 规则一：等投入分配

**谱系。** 一个谱系由一个无父代回复及其全部派生程序组成。初始化与发散各开启一个谱系。修复与回访后的重问留在原回复的谱系中。谱系的**投入**是它的程序记录数，即它消耗的候选预算，包括失败、修复和回访。持有当前最好程序的谱系是**冠军**。

**检查点。** 首个检查点为 `1 + mature_steps`（默认 5），之后逐次加倍：5、10、20、40……

**每次尝试的分配。** 对冠军以外、至少有一个有效程序的谱系：

1. 投入少于首个检查点：处于发育期，活跃；
2. 否则取它已到达的最高检查点 `r`。若它前 `r` 次尝试的最好成绩不差于冠军前 `r` 次尝试的最好成绩，活跃；否则暂停。冠军投入少于 `r` 时，取冠军的当前成绩。

下一次尝试给投入最少的活跃谱系，同投入时给较早的谱系；没有活跃谱系时给冠军。冠军更替后，所有谱系按新冠军的曲线重新判断。

**谱系内的生成。** 投入少于首个检查点的谱系（冠军也一样）执行 `Mature`：从谱系最好程序出发，使用 V14 的发育提示。其余情况按基础权重抽样 Refine／Explore／Crossover，父代只在该谱系的有效程序中按基础父代权重抽样。Crossover 的参考程序仍从全档案中位数以上的程序中均匀抽取。

**新谱系。** 自上次新最优或上次发散回复起，连续 `stagnation_window` 次候选没有新最优时，发出一次 V14 的发散请求（设计地图，只含描述）。新谱系随后按上述规则发育与比较。近期发散列表展示每个发散谱系的首个非回访回复及该谱系目前最好成绩。

分配全部由程序记录推出，不另存状态；恢复运行时分配不变。

## 规则二：信息形式曝光

被修改的程序始终给完整代码。其他程序的形式如下：

| 提示 | 基础循环与 V14 | V15 |
| --- | --- | --- |
| Refine、Mature 的形成轨迹 | 每步设计描述、评价结果与代码差异 | 每步设计描述与评价结果 |
| Crossover 的参考程序 | 完整代码；双方形成轨迹含代码差异 | `[Reference Design]`：设计描述与评价值；双方形成轨迹不含代码差异 |
| 第 2–5 个初始程序 | 已有初始程序的完整代码，并要求参照其代码 | 已有初始程序的设计描述与评价值 |
| Explore、发散 | 只含描述 | 不变 |
| 修复 | 失败程序的代码 | 不变（失败程序就是被修改的程序） |

**设计描述写在代码之后。** 回复字段顺序改为 `analysis`、`code`、`design`，描述须写出决定输出的量、公式和参数值，以及本次修改，至多 200 词。描述因此说明已写成的代码。记录说明、Crossover 与后续初始化的指令同时改写，不再要求“参照代码”。

## 默认值与记录

V14 默认值不变：`stagnation_window=40`、`mature_steps=4`、`map_size=8`、`map_distance=0.3`、`diverge_shown=4`、输出上限 8192。V15 默认 `allocation=effort`、`exposure=info`。

普通尝试与发育的 `request.allocation` 记录 `lineage`（谱系首个程序编号）、`champion`、`effort`（分配时该谱系已有尝试数）与 `checkpoint`（比较所用检查点，发育期为空）。`summary.json` 的 `v15` 字段汇总有效谱系数、越过首个检查点的谱系数、投入最多谱系的占比，以及冠军谱系、其来源与投入。

```bash
.venv/bin/python -m experiments.traceaad_v15.run --task tsp_construct --dry-run
.venv/bin/python -m experiments.traceaad_v15.launch_host \
  --batch 20261027_v15_co6 --suite co6 --repeats 3 --budget 1000 \
  --backends server3 --eval-workers 8 \
  --scheduler-socket /tmp/traceaad-1003/scheduler.sock
# 2×2 消融的另两格（TSP、CVRP、OP 各三路）
... --experiment traceaad_v15_who --allocation effort --exposure code
... --experiment traceaad_v15_form --allocation fitness --exposure info
```

正式批次 `20261027_v15_co6` 与两格消融的冻结快照、启动与同步见[操作说明](../../experiments/infra/AGENT_OPERATIONS.md#v15-正式批次与-22-消融10月27日)。
