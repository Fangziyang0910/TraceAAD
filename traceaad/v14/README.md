# TraceAAD V14

V14 针对[适应度引导下的保守化](../../docs/论文：TraceAAD.md)：LLM 算子看到的内容由选择决定，生成侧与选择侧耦合，修改回到已有内容甚至回到父代。V14 在基础 TraceAAD 循环（初始化 → Refine／Explore／Crossover → 一次修复）上加入三个机制。每个机制只对应一个失效点。它不继承 V13／V13.1 的变换规格与重放。

| 失效点 | 机制 | 来源 |
| --- | --- | --- |
| 回复的代码与已评价程序相同，评价被浪费，已接受设计的选择权重被抬高 | 回访拒绝 | 禁忌搜索 |
| 上下文中的父代代码锚定生成，长期没有新最优 | 停滞后发散：只给设计地图，不给代码 | 新颖性搜索、重启策略 |
| 新设计初始分数低，进入普通父代选择后很快被淘汰 | 发育期：先改进几次再参与竞争 | NEAT 物种保护、ALPS 年龄分层 |

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
