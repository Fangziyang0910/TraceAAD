# V12.1：设计实现检查

Explore 提出设计后，可以参考同一计算的已有实现，再检查代码是否实现设计、是否保留起点中未被替换的计算。检查指出缺失且程序未优于起点时，最多补写两步。规则见[方法说明](../../docs/01-搜索方法/设计核验-V12.1.md)，程序对象与保存方式见[实现说明](../../traceaad/v12_1/README.md)。

每次成功解析都创建独立个体并评价，相同代码也不例外。解析失败只保存调用与错误日志，仍消耗生成预算。新实验使用新目录；旧双结构档案的历史条件与当前规则不同，复现使用其源码快照。

## 配置

| 组 | 配置 | 补写方式 |
| --- | --- | --- |
| `verified` | `Config` | 按代码检查意见补写 |
| `draw` | `DrawConfig` | 需要补写的探索以 0.5 概率获得补写 |
| `resample` | `ResampleConfig` | 相同触发条件，原样重发首次实现请求 |

各组需要 `DEEPSEEK_API_KEY` 与 `DASHSCOPE_API_KEY`，嵌入服务使用 `text-embedding-v4`。共同条件见[实验规范](../../docs/08-实验规范与共识/README.md)。

## 运行

```bash
# 检查条件，不调用模型
uv run python -m experiments.traceaad_v12_1.run --task tsp_construct --dry-run

# 生成六任务计划；加 --launch 才启动
uv run python -m experiments.traceaad_v12_1.launch_host --arm verified --batch <新批次> --suite co6 --repeats 3 --budget 1000
uv run python -m experiments.traceaad_v12_1.launch_host --arm draw --batch <新批次> --suite co6 --repeats 3 --budget 1000
uv run python -m experiments.traceaad_v12_1.launch_host --arm resample --batch <新批次> --suite co6 --repeats 3 --budget 1000

uv run pytest -q tests/method/test_traceaad_v12_1.py tests/method/test_traceaad_individuals.py
```

状态查询、同步与独立测试见[执行入口](../infra/AGENT_OPERATIONS.md)。同步时使用 `--experiment traceaad_v12_1`。

`--backends` 默认 `deepseek deepseek`。使用 server3 的两个 Qwen 端点时，设置 `--backends server3 server3b` 并提供调度器 socket。DeepSeek 运行提供 socket 时，调度器只分配 CPU，模型请求直接发往网关。

## 检查报告

```bash
# 汇总已保存的代码检查与补写结果
uv run python -m experiments.traceaad_v12_1.verification_report --run-dir <当前运行目录> --examples 10

# 对当前格式档案的 Explore 首版重新做离线代码检查，会调用模型
uv run python -m experiments.traceaad_v12_1.verification_report --run-dir <当前运行目录> --replay --examples 10
```

输出位于 `experiments_result/traceaad_v12_1/verification/`。每路一个 JSON，保存逐首版记录与汇总。`--examples N` 另写 Markdown，按检查类别各选 N 例，展示起点与首版代码，供人工核对。

| 字段 | 含义 |
| --- | --- |
| `readings` | 已判断首版中，`test`（检查通过）、`missing`（未实现设计）、`dropped`（丢失原有计算）的比例 |
| `readings_if_worse`、`readings_if_better` | 按首版是否优于起点分别统计 |
| `worse_first_followed` | 不优于起点的首版此后被选为父代的比例，按检查是否通过分组 |
| `completion` | 按是否补写分组，统计最后版本是否通过检查、通过检查的版本是否优于起点、探索最好是否优于起点，以及后续被选为父代的比例 |

重放使用该运行保存的任务说明，并单独报告代码检查请求的 token 消耗。

## 离线配对

```bash
# 比较同一首版之后的已保存步骤、独立重采样与按检查补写，会生成并评价程序
uv run python -m experiments.traceaad_v12_1.completion_pairs --run-dir <当前运行目录>

# 只按已有配对记录重算汇总
uv run python -m experiments.traceaad_v12_1.completion_pairs --summary-only
```

输出位于 `experiments_result/traceaad_v12_1/completion_pairs/`：`rows.jsonl` 每个首版一行，`summary.json` 按任务汇总，`run.log` 保存日志。

早期批次、提示检查和配对结果见[2026-10-14 诊断](../../docs/03-现象与检验/2026-10-14-设计实现检查与补写.md)。历史结果不使用本页命令重写。
