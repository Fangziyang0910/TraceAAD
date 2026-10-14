# V12.1：设计核验

机制规则见[设计文档](../../docs/01-搜索方法/设计核验-V12.1.md)。在 V12.0 复提接续上，每个 Explore 版本由两次请求阅读：是否实现了 Design，是否保留了起点中 Design 没有替换的计算。不是有效检验且不优于起点的版本被补全：缺设计时在原版本上补，丢了起点时把设计放回起点，至多两步，补全返回已有程序即停。未实现设计的程序不被继承。随机续写取消。

代码位于 `traceaad/v12_1/`，继承 `traceaad/v12_0/`。三组只差配置：

| 组 | 配置 | 用途 |
| --- | --- | --- |
| `verified` | `Config`，需要补全的探索都按阅读补全 | C4、C5 主实验 |
| `draw` | `DrawConfig`，需要补全的探索以 0.5 概率补全 | C2 随机补全组 |
| `resample` | `ResampleConfig`，触发与候选数相同，补全改为原样重发第一次实现请求 | C3 重采样组 |

C4、C5 的对照是 V12.0 的 `linking` 与 `base` 组（[V12.0 入口](../traceaad_v12_0/README.md)）。各组都需要 `DEEPSEEK_API_KEY` 与 `DASHSCOPE_API_KEY`（嵌入 `text-embedding-v4`）。条件沿用[实验规范](../../docs/08-实验规范与共识/README.md)。

## 运行

```bash
# 检查条件（不启动模型）
uv run python -m experiments.traceaad_v12_1.run --task tsp_construct --dry-run

# 生成计划；加 --launch 才启动
uv run python -m experiments.traceaad_v12_1.launch_host --arm verified --batch <批次> --suite co6 --repeats 3 --budget 1000
uv run python -m experiments.traceaad_v12_1.launch_host --arm draw --batch <批次> --suite co6 --repeats 3 --budget 1000
uv run python -m experiments.traceaad_v12_1.launch_host --arm resample --batch <批次> --suite co6 --repeats 3 --budget 1000

uv run pytest -q tests/method/test_traceaad_v12_1.py
```

状态查询、同步与测试沿用[执行入口](../infra/AGENT_OPERATIONS.md)；同步时用 `--experiment traceaad_v12_1`。

`--backends` 默认 `deepseek deepseek`；在 server3 上用 Qwen 时写 `--backends server3 server3b`，请求经调度器分配到两个 GPU 端点。DeepSeek 运行加 `--scheduler-socket` 时，调度器只分配 CPU，模型请求直接发往网关。

## 2026-10-14 批次

server3，主实验组 `verified`，六任务各两路（种子 0、1），预算 1000，`--scheduler-socket /tmp/traceaad-1005/scheduler.sock`：

| 批次 | 模型 | 对照 |
| --- | --- | --- |
| `20261014_qwen_co6` | Qwen3.8（server3 两个端点） | V12.0 接续组 `20261013_qwen_co6` 的第 1、2 路，同模型、同协议版本 |
| `20261014_deepseek_co6` | DeepSeek `deepseek-v4.1-flash` | DeepSeek 基线 |

本机 tmux `v12_1_sync_qwen`、`v12_1_sync_deepseek` 每 60 秒同步。

DeepSeek 网关偶发返回 400 “Upstream request failed … enable_thinking”，同一请求重发即成功。`core/llm.py` 现在把它当作可重试错误。此前 `tsp` 两路与 `jssp` 第 1 路因此中断，已按批次清单原命令在同名 tmux 会话中重启，从 `events.jsonl` 续跑（`resumed=True`）。

## 核验报告

```bash
# C1：在 V12.0 档案上离线阅读每个 Explore 首版，可在运行前缀上进行；使用该运行自己的模型设置
uv run python -m experiments.traceaad_v12_1.verification_report --run-dir <V12.0 运行目录...> --replay --examples 10

# V12.1 运行：汇总在线阅读与补全
uv run python -m experiments.traceaad_v12_1.verification_report --run-dir <运行目录...> --examples 10
```

输出写到 `experiments_result/traceaad_v12_1/verification/`。每路一个 JSON（逐首版行与汇总）；`--examples N` 另写 Markdown，三类阅读各 N 例，含起点与首版完整代码，用于抽查阅读是否正确。汇总字段：

| 字段 | 含义 |
| --- | --- |
| `readings` | 已判断的首版中，`test`（有效检验）、`missing`（缺设计）、`dropped`（丢了起点）的比例 |
| `readings_if_worse`、`_if_better` | 按首版是否优于起点分开 |
| `worse_first_followed` | 不优于起点的首版此后被选为父代的比例，分 `test`／`not_test` |
| `completion` | V12.1：需要补全的探索按是否补全分组，报告最终版本为有效检验（`ends_test`）、有效检验优于起点（`test_improved`）、探索最好优于起点、此后被选为父代的比例；`completed_by_reading` 再按缺设计与丢了起点分开 |

重放的阅读提示与在线阅读相同，任务说明取自该运行的第一条提示；阅读请求的 token 另行报告。

## 离线配对检查

```bash
# 在 V12.0 档案上，对不优于起点且不是有效检验的首版比较 develop / resample / complete 三组下一步；可中断续跑
uv run python -m experiments.traceaad_v12_1.completion_pairs --run-dir <V12.0 运行目录...>
# 只按已有行重算汇总（截断回复重新判为未判断）
uv run python -m experiments.traceaad_v12_1.completion_pairs --summary-only
```

输出在 `experiments_result/traceaad_v12_1/completion_pairs/`：`rows.jsonl` 逐首版一行，`summary.json` 按任务汇总，`run.log`。2026-10-14 在 `20261013_qwen_co6` 的 18 路上完成 602 行，结果与解读见[设计文档](../../docs/01-搜索方法/设计核验-V12.1.md)第 6 节。

## 2026-10-14 冒烟检查

TSP 四次、JSSP 一次，预算 12–16，2 个根，Explore 份额设为 1，真实 DeepSeek、嵌入与评价，结果写在临时目录。用来定提示，不代表效果。

| 次 | 任务 | 阅读 | 请求 | 观察 | 改动 |
| --- | --- | --- | ---: | --- | --- |
| 1 | TSP | 只读实现 | 35 | 9 次中 6 次缺设计；抽查两例正确 | 引用句改为 `Read against the Design: {句子}` |
| 2 | TSP | 一次请求读实现与其他改动 | 36 | 缺设计 0/10，其他改动 8/10，多为常数与细节 | 拆成两次请求；保留阅读只问丢掉的计算 |
| 3 | TSP | 两次请求 | 45 | 一次“删掉 X，因为 Design 要求”却判为丢失；两次从起点补全返回已有程序 | 逐项判断；返回已有程序即停 |
| 4 | TSP | 两次请求，逐项 | 62 | 实现阅读 13 次中 5 次缺设计；保留阅读其余全为保留，两次达到 1,024 输出上限未判断 | 输出上限提到 2,048 |
| 5 | JSSP | 同上，上限 2,048 | 58 | 10 个首版全部读为有效检验，没有补全；保留阅读最长 1,313 tokens | 无 |

第 1 次：首版 8.86、起点 8.14，补全得 8.03，仍读为缺设计；本路最好的 6.26 来自从它出发的 Explore。另一次探索的两次补全得 8.68、8.84，都没有改正第一次阅读已指出的权重方向。

第 4 次：60,749 入／19,047 出 token，235 秒，最好 6.2616。平均每次实现阅读 760 入／140 出，保留阅读 1,180 入／640 出，补全 1,660 入／865 出。7 次探索的首版直接读为有效检验。第 7 次探索首版缺设计，补全后读为有效检验，8.517 不如起点 7.855，是一次否定。第 12 次探索首版缺设计、得 25.78，补全得 6.2616，为本路最好。缺设计的句子具体，如 “does not perform greedy nearest-neighbor chaining”。

第 5 次：最好 1911.44（起点 1935.44）。抽查两例：一例正确；另一例的 Design 要求在关键弧上做禁忌移动，阅读写出代码只是交换排列中的工件、解码也不是图上的最长路，仍判为实现。这是一个偏宽的阅读，C1 抽查时专门看这一类。JSSP 程序长，平均每次实现阅读 3,900 入／97 出，保留阅读 6,570 入／720 出；阅读占全部输入 token 的 44%、输出的 12%（本次只用 Explore）。
