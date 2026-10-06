# TraceAAD V10.19

The mechanism is specified in [the design document](../../docs/01-搜索方法/对原算法的改动检验.md); the diagnosis that motivated it is in [the V10.18 diagnosis](../../docs/03-现象与检验/2026-10-06-新设计开发诊断与改动检验.md). Everything of V10.18 is kept (programs and generation events, experience-weighted starting points, measured outcomes, design-anchored Refine/Crossover goals, Explore draws that continue the open exploration before proposing) except how an Explore proposal is judged. A proposal is a change to the current algorithm: the Explore goal adds "Keep unchanged the parts of the current algorithm that the change does not replace." When the first version does not score better than the algorithm the change was made to, up to 3 Develop steps follow; each starts from the best version the change has reached and sees that algorithm in full, the change's own formation from its first version and the other attempts on the change, with the goal "Make the change work in the algorithm it was made to: keep the computation the change introduces and write a version that scores better than that algorithm." The change ends as soon as a version beats that algorithm. Its outcome (the best version reached) is stated in that algorithm's attempt list and counted in its experience. Shares: Refine 0.40, Explore 0.35, Crossover 0.25.

```bash
uv run python -m experiments.traceaad_v10_19.run --task tsp_construct --run-name trial_1 --budget 1000 --dry-run
uv run python -m experiments.traceaad_v10_19.run --task tsp_construct --run-name trial_1 --budget 1000
uv run python -m experiments.traceaad_v10_19.heldout --run-dir experiments_result/traceaad_v10_19/tsp_construct/trial_1
```

`launch_local.py --batch <name> [--launch]` starts the 15-run batch on this machine (five tasks x three repeats, seeds 0-2, the two server3 Qwen services alternated 8/7, four evaluation workers), the same host as V10.18. `heldout_batch.py --batch-manifest <manifest>` evaluates all selected programs. Method tests: `tests/method/test_traceaad_v1019.py`.

Batch `20261006_local_v1019` was launched on 2026-10-06 (manifest `experiments_result/traceaad_v10_19/batch_20261006_local_v1019.json`).

## 2026-10-06 实现整理

本版本的机制保留在版本目录中，生成、评价、记录、实验入口与离线诊断改为[共用实现](../../traceaad/common/README.md)。结果仍写入 `experiments_result/traceaad_v10_19/`。解析器不再补模板依赖，完整尝试结束后才保存恢复点，允许重做未提交的尝试；revision 标记这些运行条件。历史结果已统一迁移，格式与恢复条件见[实验与结果](../infra/SEARCH_FORMAT.md)。
