# TraceAAD V20

V20 在 [Base](../traceaad_base/) 上加入三个机制：轨迹评分、轨迹引导的上下文与算子、轨迹反思与经验学习。机制说明见[V20](../docs/机制设计/V20.md)。评价、修复、预算、记录与恢复都沿用 Base。

| 文件 | 内容 |
| --- | --- |
| [search.py](search.py) | Base 的子类：选择、生成材料、反思三个扩展点 |
| [selection.py](selection.py) | P·h 评分、Laplace 在线更新与 Thompson 抽样 |
| [context.py](context.py) | 祖先、子代、失败、参考与经验的选取和呈现，三种算子指令 |
| [experience.py](experience.py) | 观察事件、review、学习片段与经验召回 |
| [representation.py](representation.py) | 源码身份、AST 结构、变更单元与 RRF |
| [prompts.py](prompts.py) | 去掉字数上限的输出说明与简短失败反馈 |
| [embedding.py](embedding.py) | Qwen 嵌入端点 |

`Config` 默认开启四个开关：`trajectory_scoring`、`trajectory_context`、`reference_selection`、`experience_learning`。`TraceAAD` 可注入 `embedder` 与 `reflection_model`。

V20 在 Base 记录之外另写 `decisions.jsonl`（每次选择的全部候选评分）与 `experience.json`（每次尝试后的经验状态）。选择记录、展示材料与辅助用量存于各尝试的 `record`。恢复时按顺序重放选择记录得到 P/h 后验，再读入经验状态。

```bash
.venv/bin/python -m pytest -q tests/v20
```
