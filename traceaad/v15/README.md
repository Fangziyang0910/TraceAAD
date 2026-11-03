# TraceAAD V15

研究问题与机制设计见[独立方法说明](../../docs/机制设计/V15.md)。

V15 以同投入竞争减轻开发程度差异对选择的影响，以设计描述共享其他程序的计算思路。开发分配决定修改哪个谱系，信息曝光决定生成器以什么形式利用已有程序。回访拒绝、设计地图与初期发育分别支持有效生成、新起点建立和早期开发。

| 规则 | 回答的问题 | 做法 | 开关 |
| --- | --- | --- | --- |
| 一：等投入分配 | 谁获得开发 | 投入按谱系分配；挑战谱系在相同投入下与冠军谱系比较 | `allocation=effort` |
| 二：信息形式曝光 | 以什么形式共享经验 | 只有被修改的程序给代码；其他程序给设计描述与评价结果 | `exposure=info` |

消融可用 `allocation=fitness` 直接按程序权重分配父代，或用 `exposure=code` 在参考上下文中展示源码。

## 规则一：等投入分配

**谱系。** 一个谱系由一个无父代程序及其沿主父代关系产生的后代组成。初始化与发散各开启一个谱系。修复与回访后的重问留在原回复的谱系中。谱系的**投入**是已建立的程序记录数，包括失败、修复和回访；无法解析建档的回复只计全局生成预算。持有当前最好程序的谱系是**冠军**。

**检查点。** 首个检查点为 `1 + mature_steps`（默认 5），之后逐次加倍：5、10、20、40……

**每次尝试的分配。** 对冠军以外、至少有一个有效程序的谱系：

1. 投入少于首个检查点：处于发育期，活跃；
2. 否则取它已到达的最高检查点 `r`。若它前 `r` 次尝试的最好成绩不差于冠军前 `r` 次尝试的最好成绩，活跃；否则暂停。冠军投入少于 `r` 时，取冠军的当前成绩。

下一次尝试给投入最少的活跃谱系，同投入时给较早的谱系；没有活跃谱系时给冠军。冠军更替后，所有谱系按新冠军的曲线重新判断。

**谱系内的生成。** 投入少于首个检查点的谱系（冠军也一样）执行 `Mature`：从谱系最好程序出发，保留主决策规则，修补弱点、调整参数与完善辅助计算。其余情况按操作权重抽样 Refine／Explore／Crossover，父代只在该谱系的有效程序中按质量与直接改进经验抽样。Crossover 的参考程序从全档案成绩不劣于中位数的程序中均匀抽取。

**新谱系。** 自上次新最优或上次发散回复起，连续 `stagnation_window` 次候选没有新最优时，发出一次只含设计地图描述的发散请求。新谱系随后按上述规则发育与比较。近期发散列表展示每个发散谱系的首个非回访回复及该谱系目前最好成绩。

分配全部由程序记录推出，不另存状态；恢复运行时分配不变。

## 规则二：信息形式曝光

被修改的程序始终给完整代码。其他程序的形式如下：

| 提示位置 | 提供的信息 |
| --- | --- |
| Refine、Mature 的形成轨迹 | 每步设计描述与评价结果 |
| Crossover 的参考程序 | `[Reference Design]`：设计描述与评价值；双方形成轨迹不含代码差异 |
| 第 2–5 个初始程序 | 已有初始程序的设计描述与评价值 |
| Explore、发散中的参考材料 | 设计描述 |
| 修复对象 | 失败程序的完整代码 |

**设计描述写在代码之后。** 回复字段顺序为 `analysis`、`code`、`design`，描述须写出决定输出的量、公式和参数值，以及本次修改，至多 200 词。描述因此说明已写成的代码，供后续生成共享具体计算思路。

## 默认值与记录

默认 `stagnation_window=40`、`mature_steps=4`、`map_size=8`、`map_distance=0.3`、`diverge_shown=4`、输出上限 8192，`allocation=effort`、`exposure=info`。

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

## Python 包入口

本目录自带本版的配置、搜索、提示、选择、个体、评价、历史、状态、存储、执行限制与 `runtime/` 运行实现。公开入口为 `Config`、`Program`、`TraceAAD`、`Evaluation`、`LLM`、`OpenAIAPI`，原有类名 `TraceAADV15` 也可使用。任务评价对象与模型客户端由调用方传入；本包提供源码解析、模型请求、隔离执行和调度客户端。运行需要 Python 3.14 与 NumPy、OpenAI、Requests 等第三方依赖。

```python
from traceaad.v15 import Config, TraceAAD

search = TraceAAD(evaluation=task, llm=client, run_dir="runs/example", config=Config())
summary = search.run()
```

`loop.py` 负责生成、评价、修复与检查点，`search.py` 和包内辅助模块组织本机制。各模块使用包内相对导入；复制整个目录并改名后可作为独立 Python 包导入。实验入口位于 `experiments/traceaad_v15/`，自带运行准备、模型路由、批次启动与存储工具；标准实验从 `benchmarks` 读取任务和数据。
