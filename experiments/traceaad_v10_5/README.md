# TraceAAD V10.5

实现 [V10.5 设计文档](../../docs/01-主线版本/TraceAAD-V10.5-机制设计.md)中的唯一推荐首发配置。正式批次为五任务 × 三重复，每路独立初始化并执行 1000 次真实评价。

| 机制 | 首发参数 |
| --- | --- |
| 生成 | 一次请求输出简短 Idea 和完整 Code |
| 算子概率 | Refine / Pivot / Fuse = 0.50 / 0.15 / 0.35 |
| 质量基座 | ESS 目标 min(N, max(0.1N, 2))，并处理并列最优下界 |
| 次数修正 | p0 ∝ 质量权重 / sqrt(父节点选择次数 + 1) |
| 父节点 | R/F 用 p0；P 用 0.5 p0 + 0.5 全部上下文可用节点均匀分布 |
| Fuse donor | 排除自身、祖先和后代；可容纳的最高质量 5 个中均匀选一个 |
| 无 donor | 请求 Fuse 转为执行 Refine，保留 requested / executed 信息 |
| 轨迹 | 最近 8 条形成事件，含当前节点形成事件；历史最多 2048 实际 tokens |
| 上下文 | 32768，总输出预留 16384，安全余量 256；服务端精确计数 |
| 初始化与档案 | 8 个有效根；保留所有有限 fitness 节点，包括重复、持平和退化 |
| 模型 | Qwen3.8-27B，temperature=1.0、top_p=0.95、top_k=20，enable_thinking=False |

先选算子，再选父节点；每次只生成一个候选，评价后立即重新分配。Idea 长度为软指引，不对算法结构作额外语义验收。完整候选模块直接进入统一 SecureEvaluator，存档和评价采用同一代码文本，保留装饰器、辅助函数、类和模块级语句。只有 prompt 视图去除注释；不截断当前或 donor 代码。

## 运行

仓库根目录使用 `.venv/bin/python`。单路示例：

```bash
.venv/bin/python -m experiments.traceaad_v10_5.run --task tsp_construct --backend server3 --repeat 1 --seed 0 --run-name example_tsp_v105_rep1
```

提交完整批次，持续等待空闲槽位并恢复可恢复的中断：

```bash
.venv/bin/python -m experiments.traceaad_v10_5.launch --batch UNIQUE_BATCH --session-prefix v105 --watch
```

加 `--dry-run` 只展示当前可启动的命令。使用唯一批次名；已有同批次 manifest 则继续调度该批次。三个重复 seed=0/1/2，尽可能分散到不同服务。仅使用 server1、server3、server3b、本地；逻辑容量分别为 6、9、9、3，扣除现有实验占用。调度器不停止其他实验，也不扩大服务并发。每次从当前空闲容量中分配；不足部分自动排队。

调度器应放入持久 tmux session；每一路实验由它另建 tmux session。恢复固定原 backend、run_dir、seed，不以新目录重开一套预算。每路最多启动 5 次；耗尽重试标记 stopped，未知评价标记 blocked，两者均不假报完成。

## 预算与恢复

- Init、评价失败和超时都计入实际评价预算；解析失败、`finish_reason=length`、仅 LLM 失败不计评价预算。
- 选择、完整响应、评价提交、评价收据分阶段持久化。传输重试保持原父节点、donor、prompt 和选择后 RNG；父节点次数只加一次。已落盘响应不重复生成。
- 评价收据写入后，即使在树 checkpoint 前崩溃，也可恢复结果而不重复评价。若只有提交状态而无可确认收据，标记 unknown reservation 并停止该路，避免免费重评；confirmed 与 unknown 单独报告。
- checkpoint 记录机制参数、关键源文件和任务契约指纹。正式运行期间保持这些源文件稳定；不兼容修改会拒绝恢复。
- 连续 50 次无有效输出停止并记录错误。评价预算用尽而有效根不足时也标记错误。

## 运行产物

`results/batch_BATCH.json` 是批次 manifest，包含每路 backend、session、状态、启动次数。每路目录中：

| 文件 | 内容 |
| --- | --- |
| `run_config.json` | 任务、模型采样与机制配置 |
| `tree_state.json` | 全档案、真实预算、选择次数、RNG、机制指纹 |
| `pending_candidate.json` | 尚未提交为节点的操作阶段和可恢复上下文；完成候选后删除 |
| `llm_calls.jsonl` | 完整 prompt / response、finish reason、usage、采样与耗时 |
| `tokenizer_calls.jsonl` | 辅助计数的类型、文本 hash、tokens、耗时及失败 |
| `evaluations.jsonl` | 唯一候选与评价槽号、真实评价收据 |
| `events.jsonl` | 请求/执行算子、父代与 donor、概率、ESS、裁剪记录和原始增益 |
| `logs/run_summary.json` | 终态、best、confirmed evaluations 与 unknown reservations |

JSONL 日志恢复只修复损坏的末尾记录；中间损坏会报错。批次 smoke 目录以 `smoke_` 开头，预算单列，不进入正式 1000-eval 比较。

## 验证

```bash
.venv/bin/python -m pytest -q tests/method/test_traceaad_v105.py tests/experiments/test_traceaad_v105_launch.py tests/tools/test_openai_api.py tests/method/test_traceaad_v103_schema.py tests/method/test_traceaad_v104.py tests/experiments/test_traceaad_v104_launch.py
```

覆盖概率与回退、事件对齐和上下文裁剪、完整代码执行、失败预算、持久化响应、评价收据及提交断点恢复、未知评价阻断、批次容量与恢复，以及共享 API 和前版机制回归。
