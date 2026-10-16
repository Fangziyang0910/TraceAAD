# V12.1 程序个体

搜索、提示、评价、存储和配置都在本目录。

`Program` 是一次成功解析的程序生成结果。模型输出的 `analysis`、设计说明 `idea` 和源码 `code` 直接属于这个对象。父个体、参考个体、修复来源、调用编号、评价状态、成绩、错误和设计核验结果也保存在个体上。搜索与提示使用对象属性；JSON 字典只用于请求参数、评价明细和序列化。

```text
程序生成请求 → 解析输出 ┬→ 失败：保存调用和错误日志，不创建个体
                       └→ 成功：创建 Program → 校验源码 → 评价 → 更新个体
                                                    └→ 失败可产生一个 Repair 个体
```

语法错误、接口错误、运行错误和超时都是个体的失败状态。解析失败仍消耗生成预算。提出设计、匹配设计和核验代码的调用只记录调用，不单独创建程序个体。

每次解析成功均分配新个体编号，不比较源码是否重复，不复用旧成绩。同一源码可以属于多个个体，每个都重新评价并保存来源。源码哈希只用于结果完整性检查和读取源码，不决定个体身份。已有设计的语义匹配和 Crossover 的相似度选择仍是搜索机制，不执行重复代码拒绝。

| 文件 | 职责 |
| --- | --- |
| `program.py` | `Program` 类与个体属性 |
| `traceaad.py` | `TraceAADV121` 搜索、生成、修复、补全与最终选择 |
| `prompts.py`、`history.py`、`selection.py` | 从个体组织历史、提示与选择统计 |
| `linking.py`、`verification.py` | 已有设计匹配及代码核验 |
| `delivery.py` | 解析模型回复和校验已创建个体的源码 |
| `evaluation.py`、`instance_evaluation.py`、`probe.py` | 隔离评价、时限与函数计量 |
| `state.py`、`storage.py` | 个体事件、调用日志、源码与恢复点 |
| `config.py` | 当前版本的完整配置 |

`events.jsonl` 的候选事件直接包含 `program` 元数据，不再保存 `attempt` 或个体映射。解析失败为 `delivery_failed` 日志。恢复点保存 `individual_format=program-individual-v1`、进度与随机状态；从已提交事件恢复 `Program` 对象。调度器中断时保存待评价个体，恢复后继续评价，不重新生成。旧 attempt/program 格式不由当前搜索续跑；历史结果保留，复现使用对应源码快照。

运行见[实验入口](../../experiments/traceaad_v12_1/README.md)，条件见[实验规范](../../docs/08-实验规范与共识/README.md)。
