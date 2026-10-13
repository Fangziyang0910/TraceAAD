# V12.0：复提接续

机制规则见[设计文档](../../docs/01-搜索方法/复提接续-V12.0.md)。搜索使用 Refine、Explore、Crossover 与随机短程开发，在 Explore 中增加按计算继承：新提案从同一主计算目前最好的已评价实现出发实现。

V12.0 的搜索、配置和提示已独立实现，仅依赖 `traceaad/common/`，不再继承旧版本。迁移保持默认配置、提示、随机序列与恢复身份不变。旧版本和决策模型代码已删除，历史文档、结果与凭据配置保留。

三组共用同一实现，只差配置：

| 组 | 配置 | 用途 |
| --- | --- | --- |
| `base` | `BaseConfig`，关闭接续 | E1 对照；E0 在其档案上离线重放匹配 |
| `linking` | `Config`，总是展示接续程序 | E1 主实验 |
| `display` | `DisplayConfig`，接续程序以 0.5 概率展示 | 随机展示诊断 |

`linking` 与 `display` 需要 `DEEPSEEK_API_KEY` 与 `DASHSCOPE_API_KEY`（嵌入 `text-embedding-v4`）；`base` 只需要前者。条件沿用[实验规范](../../docs/08-实验规范与共识/README.md)：DeepSeek `deepseek-v4.1-flash`，六任务各三路，1000 候选，平均函数 CPU 预算。

## 运行

```bash
# 检查条件（不启动模型）
uv run python -m experiments.traceaad_v12_0.run --task tsp_construct --dry-run
uv run python -m experiments.traceaad_v12_0.run_base --task tsp_construct --dry-run

# 生成计划；加 --launch 才启动
uv run python -m experiments.traceaad_v12_0.launch_host --arm base --batch <批次> --suite co6 --repeats 3 --budget 1000
uv run python -m experiments.traceaad_v12_0.launch_host --arm linking --batch <批次> --suite co6 --repeats 3 --budget 1000
uv run python -m experiments.traceaad_v12_0.launch_host --arm display --batch <批次> --suite co6 --repeats 3 --budget 1000

uv run pytest -q tests/method/test_traceaad_v12_0.py
```

状态查询、同步与测试沿用[执行入口](../infra/AGENT_OPERATIONS.md)与 `experiments.infra.evaluate`。

## 接续报告

```bash
# E0：在基础组档案上离线重放匹配，可在运行前缀上进行
uv run python -m experiments.traceaad_v12_0.linking_report --run-dir <基础组运行目录...> --replay --examples 10

# 接续组或诊断组：汇总在线记录
uv run python -m experiments.traceaad_v12_0.linking_report --run-dir <运行目录...> --examples 10
```

输出写到 `experiments_result/traceaad_v12_0/linking/`。每路一个 JSON（逐提案行与汇总），`--examples` 另写匹配例子的 Markdown，含接续程序与首版的完整代码，用于抽查。汇总字段：

| 字段 | 含义 |
| --- | --- |
| `linked_share` | 找到同计算程序的 Explore 提案比例 |
| `matched_size_median` | 每次匹配到的程序数中位数 |
| `first_beats_linked` | 新实现超过同计算已有最好实现的比例，即实现状态推进的比例；分全部、展示、未展示 |
| `linked_beats_first` | 接续程序好于该提案自己首版的比例（首版无效也计为好于） |
| `linked_beats_parent` | 接续程序好于提案父代的比例 |
| `first_beats_parent` | 首版好于父代的比例，按已接续／未接续、展示／未展示分开 |
| `copied_linked` | 候选与接续程序为同一程序（重复）的比例 |

重放只使用每次提案之前已评价的程序，匹配请求的 token 另行报告。重复有效程序使用其已知成绩；复制接续程序记为同分，Repair 后的复制也计入复制率。

实现提示经两级裁剪仍超限时，保存已有请求成本并将父代记为上下文不可用，转向其他父代；未生成候选，不计候选预算。

## 2026-10-12 冒烟检查

TSP，预算 12，2 个根，Explore 份额设为 1，真实 DeepSeek、嵌入与评价。只检查流程，不代表效果。

- 停止序列生效：8 次提出全部在代码前停止，回复不含代码块。
- 第一次运行中，匹配回复全部未被解析。DeepSeek 先逐条推理、最后才给编号，而原解析只读第一行。改为要求以 `Same:` 行结尾、读最后一行，输出上限 256 → 1024。第二次运行 8 次匹配都给出 `Same:` 行，5 次找到接续程序。
- 匹配有宽松的情况。“用最便宜插入补全剩余路径”被接到“最近邻加一步插入惩罚的打分”，两者只共享词语，主计算不同。E0 需要抽查代码。
- 第 9 次从 7.81 的父代出发，接续 6.19 的程序，得到 6.20，代码与接续程序相似度 0.73。它主要继承了接续程序，没有超过它。
- 每次 Explore 的平均 token：提出 1305 入／145 出，匹配 307 入／182 出，实现 1725 入／746 出。
