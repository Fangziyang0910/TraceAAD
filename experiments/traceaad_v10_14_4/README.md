# TraceAAD V10.14-4

本版针对 V10.14-3 完成批次的停滞和跨路分化，修复已确认的调度实现问题，并以完整预算检验更明确的发现、发展、选择链条。诊断见[机制修订说明](../../docs/03-机制探索与验证/2026-09-29-V10.14-4-机制修订与实验协议.md)。这是联合改动，不能据此单独归因。

## 冻结协议

- **发现**：八提案 hybrid 初始化及唯一语法根的一次 bootstrap 保留。搜索预算中最多 12% 用于不带父代源码、Idea 和历史的独立 Pivot，按已消耗预算逐步释放。成功发现的根在下一步获得一次付费 Refine；失败或重复照常计费。普通开发动作在可用的 Refine、Tune、Transfer 中等概率选择。
- **发展**：先按训练探针形成的区域抽样，再在该区域内按经验中分位质量加 `0.35/sqrt(语法组使用次数+1)` 选父。初始化最多占四个区域，另外四个名额留给搜索期新根。区域权重为冠军名次倒数，乘以有界停滞项，再除以 `sqrt(1+区域主循环使用次数)`。概率输出使用平均绝对差，离散选择使用不一致比例；没有探针时进入来源区域或默认区域。区域仅是有限训练探针的调度近似，不是行为等价类，总数最多八个。
- **反馈**：提示继续使用 `Idea:` / `code:` 加一个完整 Python 代码块。Idea 建议 2-4 句、最多 320 token；原文保留，提示视图按 tokenizer 截断。父子最近一次的训练分数差和固定探针差分进入下一轮提示，最多展示两个变化场景。历史证据为 3,000 token、4 事件、4 层；原始日志不截断。Idea 和探针都不当作因果证明。
- **选择**：先保留训练前两名不同 AST，再从其余探针区域各取最优程序，空位按训练分数补足，最多五个不同 AST。冻结后仍用与训练分离的选择集评价；最终 held-out 不参与生成、调度或这次选择。分层可以提高选择集覆盖，但可能牺牲训练第 3-5 名的机会，需要在结果中核查。
- **其余**：训练面板、评价器超时、模型采样、8192 输出 token、候选计费、完整模块解析和安全执行沿用 V10.14-3。默认关闭在线重验、固定三步票、行为资格门槛。`discovery_fraction=.12`、`exploration_constant=.35` 是待检验的工程设定，不是由现有结果估计的最优值。

五任务各四重复，每路 1000 次候选；复用 V10.14-3 的 task、repeat、backend、seed 映射，新建结果目录，不续跑旧状态。主要判据为训练前沿与独立选择分数的跨路均值和最差路，以及结构发现后被开发、改变固定探针、产生净提升的比例。Idea 非空率和区域数量只用于机制核验。正式 held-out 仅在候选冻结后单独执行。

## 运行与核验

```bash
uv run python -m experiments.traceaad_v10_14_4.run --task tsp_construct --dry-run
uv run python -m experiments.traceaad_v10_14_4.launch_batch \
  --from-batch experiments_result/traceaad_v10_14_3/batch_20260928_v1014_3_template2.json \
  --batch 20260929_v1014_4_r1 --dry-run
uv run python -m experiments.traceaad_v10_14_4.verify_batch \
  --manifest experiments_result/traceaad_v10_14_4/batch_20260929_v1014_4_r1.json \
  --output experiments_result/traceaad_v10_14_4/startup_20260929_v1014_4_r1.json
```

首次 `20260929_v1014_4` 在 20 路合计 259 次尝试后中止并保留于 `experiments_result/traceaad_v10_14_4_before_region_fix/`：TSP 初始化占满八个区域，搜索期无法创建新区。修订后批次 `20260929_v1014_4_r1` 已从零启动，20/20 路通过初始核验，状态为 `running_verified`；不合并先前成绩。`verify_batch` 检查实际请求、配置、源码哈希、评价器和采样设置；初始窗口不能证明最终性能。实验监控运行在 `http://127.0.0.1:8766/`，默认显示 `traceaad_v10_14_4`。
