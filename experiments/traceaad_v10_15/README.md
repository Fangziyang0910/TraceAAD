# TraceAAD V10.15

The frozen mechanism is specified in [the design document](../../docs/01-搜索方法/生成目标与形成路径.md). Search draws parents over programs by training quality with a fixed target ESS of 8 (uniform over the top-scoring programs when at least eight tie) and immediately admits each valid unique candidate; finalists are the five best programs. An equal training score does not make two programs the same algorithm; only identical normalized code is a duplicate. The formation path is shown only in generation prompts.

The 2026-09-30 prompt revision shows complete diffs for every displayed Refine/Crossover history step. History has no separate token or diff-line cap; whole oldest steps are removed only when the full prompt exceeds 24,320 input tokens, reserving 8,192 output tokens within the 32K context. The independent initialization prompt no longer contains the “more than a single formula” instruction. Use a new run name for this revision: saved runs retain their original protocol and cannot resume after source/config changes.

When enabled (`explore_cards`), Explore shows up to four archive Idea+score cards, selected for distinct visible ideas and code diversity; parent links play no part in choosing them or the Crossover reference. Code diversity is a proxy for different ideas. Crossover displays up to four complete history diffs for each of the current and reference programs. If the full prompt is too long, the longer history loses its oldest step while each non-root side retains its latest step; if this still does not fit, Crossover falls back to Refine. Request and attempt records separately identify displayed Explore references and reference-program history edges. A stopped response whose single code block lacks only its closing fence is accepted when the block parses and defines the target once. Evaluation timeouts stay wall-clock (they are load-sensitive; report timeout rates), OBP re-executes the candidate per instance, the VRPTW template states the depot rule, and sampling uses Qwen3.8's non-thinking profile; see the parameters in the [method description](../../docs/01-搜索方法/生成目标与形成路径.md).

```bash
uv run python -m experiments.traceaad_v10_15.run --task tsp_construct --run-name trial_1 --budget 1000 --dry-run
uv run python -m experiments.traceaad_v10_15.run --task tsp_construct --run-name trial_1 --budget 1000
uv run python -m experiments.traceaad_v10_15.heldout --run-dir experiments_result/traceaad_v10_15/tsp_construct/trial_1
```

For ACO held-out evaluation, pass `--split test_50`, `test_100`, or `test_200` (CVRP also supports `test_20`). TSP and VRPTW accept `eval_50`, `eval_100`, and `eval_200`; online bin packing accepts `eval_<items>_<capacity>` for 1,000/5,000/10,000 items and capacity 100/500. Held-out limits default to 3,000 s (TSP), 1,000 s (VRPTW, OBP) and 3,600 s (CVRP, OP) at every size, matching the shared baseline evaluator; `--timeout-seconds` can set an explicit limit; the evaluated protocol and limit are saved with each result. Search, independent selection, and held-out results are recorded separately. A completed run selects from the five programs with the best training scores; selection allows twice the training limit for TSP, VRPTW and OBP. Training limits are 30 s for TSP, VRPTW and OBP, 60 s for OP and 120 s for CVRP; the effective timeout is recorded in `run_config.json`.

The formal launcher accepts a prior twenty-run batch manifest so the backend assignment is retained. Inspect its plan with `--dry-run` before launch:

```bash
uv run python -m experiments.traceaad_v10_15.launch_batch --from-batch experiments_result/traceaad_v10_14/batch_20260928_v1014.json --batch v1015_trial --dry-run
uv run python -m experiments.traceaad_v10_15.heldout_batch --batch-manifest experiments_result/traceaad_v10_15/batch_v1015_trial.json --dry-run
```
