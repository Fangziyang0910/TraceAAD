# TraceAAD V10.16

The mechanism is specified in [the design document](../../docs/01-搜索方法/尝试记录与经验使用.md). The search records each program by normalized code and each generation attempt with its inputs and measured outcome, including failures. Duplicates, known failures and repairs are events linked to programs; only identical code counts as a duplicate. Parents are drawn by training quality times a smoothed improvement rate, `(k + 3r) / (n + 3)` over the attempts started from it. Refine, Explore and Crossover see the attempts that started from the current algorithm with their measured outcomes; Explore also sees how the search best improved. Every evaluation records the calls to the target function and the time inside it, also when it times out. A finalist that fails on the selection set is replaced by the next program in the training ranking (at most five replacements).

```bash
uv run python -m experiments.traceaad_v10_16.run --task tsp_construct --run-name trial_1 --budget 1000 --dry-run
uv run python -m experiments.traceaad_v10_16.run --task tsp_construct --run-name trial_1 --budget 1000
uv run python -m experiments.traceaad_v10_16.heldout --run-dir experiments_result/traceaad_v10_16/tsp_construct/trial_1
```

Training, selection and held-out limits and splits are the same as for [V10.15](../traceaad_v10_15/README.md). `launch_server3.py --batch <name> [--launch]` starts the 15-run server3 batch arranged like V10.15-6 (five tasks x three repeats, seeds 0-2, two Qwen services).

## 2026-10-06 实现整理

本版本的机制保留在版本目录中，生成、评价、记录、实验入口与离线诊断改为[共用实现](../../traceaad/common/README.md)。结果仍写入 `experiments_result/traceaad_v10_16/`。解析器不再补模板依赖，完整尝试结束后才保存恢复点，允许重做未提交的尝试；revision 标记这些运行条件。历史结果已统一迁移，格式与恢复条件见[实验与结果](../infra/SEARCH_FORMAT.md)。
