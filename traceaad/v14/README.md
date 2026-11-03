# TraceAAD V14

研究问题与机制设计见[独立方法说明](../../docs/机制设计/V14.md)。

V14 通过回访拒绝、无代码发散与定额发育，减少重复评价和代码锚定，并为新设计保留初期开发机会。搜索使用初始化、Refine、Explore、Crossover 与至多一次修复。

| 失效点 | 机制 | 作用 |
| --- | --- | --- |
| 回复与已评价程序等价，重复评价缺少新信息 | 回访拒绝 | 跳过重复评价，并反馈实际重复 |
| 上下文中的代码锚定生成，长期没有新最优 | 无代码发散 | 用设计地图引导新的决策计算 |
| 新设计初始成绩较差，难以获得后续机会 | 定额发育 | 先完成一段修改，再参与普通父代竞争 |

## 机制

**回访拒绝。** 去掉文档字符串与注释后，比较规范化 AST。与任一已评价程序相同的回复记为 `revisit`，不评价；同一请求再问一次，并说明它与“当前算法／参考算法／已评价算法”相同。重问计入候选预算，最多一次，与普通修复共用 `repair_of` 记账。

**停滞后发散。** 自上次新最优或上次发散结束起，连续 `stagnation_window` 次候选没有新最优时，开始一个发散期。发散提示列出已有设计的思路描述，不含代码。设计按适应度排序，依次贪心选取与已选设计距离 ≥ `map_distance` 的程序，最多 `map_size` 个。提示要求写出主决策规则不同的新算法，不能对已有设计重新加权、扩展或组合。最近 `diverge_shown` 次发散的设计及其发育后最好成绩也列出，避免重复同一方向。若首个回复无效，再请求一次新设计。

**发育期。** 发散得到有效程序后，以发散期内最好的成员为父代，做 `mature_steps` 次 Mature 修改。Mature 是 Refine 提示加一段说明：保留主决策规则，修弱点、调参数、补辅助计算。发育期结束后，所有成员照常进入档案与普通父代选择。

## 默认值与记录

`stagnation_window=40`、`mature_steps=4`、`map_size=8`、`map_distance=0.3`、`diverge_shown=4`、输出上限 8192。距离为 5 词元片段的 Jaccard 距离，数字统一记为 NUM。`--revisit=False` 或 `--diverge=False` 可分别关闭前两个机制；`mature_steps=0` 关闭发育期。

`resume.json` 的 `state.v14` 保存当前发散期与已结束发散期（成员、最好程序、是否产生新最优）。`summary.json` 的 `v14` 字段汇总回访次数、发散期数、产生新最优的发散期数、发散与发育尝试数。程序记录的 `request.episode` 标明所属发散期，`map_ids` 与 `tried_ids` 记录发散提示中出现的设计。

```bash
.venv/bin/python -m experiments.traceaad_v14.run --task tsp_construct --dry-run
.venv/bin/python -m experiments.traceaad_v14.launch_host \
  --batch 20261027_v14_co6 --suite co6 --repeats 3 --budget 1000 \
  --backends server3 --eval-workers 8 \
  --scheduler-socket /tmp/traceaad-1003/scheduler.sock
```

## Python 包入口

本目录自带本版的配置、搜索、提示、选择、个体、评价、历史、状态、存储、执行限制与 `runtime/` 运行实现。公开入口为 `Config`、`Program`、`TraceAAD`、`Evaluation`、`LLM`、`OpenAIAPI`，原有类名 `TraceAADV14` 也可使用。任务评价对象与模型客户端由调用方传入；本包提供源码解析、模型请求、隔离执行和调度客户端。运行需要 Python 3.14 与 NumPy、OpenAI、Requests 等第三方依赖。

```python
from traceaad.v14 import Config, TraceAAD

search = TraceAAD(evaluation=task, llm=client, run_dir="runs/example", config=Config())
summary = search.run()
```

`loop.py` 负责生成、评价、修复与检查点，`search.py` 和包内辅助模块组织本机制。各模块使用包内相对导入；复制整个目录并改名后可作为独立 Python 包导入。实验入口位于 `experiments/traceaad_v14/`，自带运行准备、模型路由、批次启动与存储工具；标准实验从 `benchmarks` 读取任务和数据。
