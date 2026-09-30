# TraceAAD V10.15

The frozen mechanism is specified in [the design document](../../docs/01-主线机制设计/TraceAAD-V10.15-机制设计.md). Search uses the quality-only ESS distribution and immediately admits each valid unique candidate. The formation path is shown only in generation prompts.

The 2026-09-30 prompt revision shows complete diffs for every displayed Refine/Crossover history step. History has no separate token or diff-line cap; whole oldest steps are removed only when the full prompt exceeds 24,320 input tokens, reserving 8,192 output tokens within the 32K context. The independent initialization prompt no longer contains the “more than a single formula” instruction. Use a new run name for this revision: saved runs retain their original protocol and cannot resume after source/config changes.

Explore now replaces its lineage ideas with up to four archive Idea+score cards, selected for distinct visible ideas and code diversity, preferring candidates with no ancestor/descendant relationship to the parent. Code diversity is a proxy for different ideas. Crossover displays up to four complete history diffs for each of the current and reference programs. If the full prompt is too long, the longer history loses its oldest step while each non-root side retains its latest step; if this still does not fit, Crossover falls back to Refine. Request and attempt records separately identify displayed Explore references and reference-program history edges.

```bash
uv run python -m experiments.traceaad_v10_15.run --task tsp_construct --run-name trial_1 --budget 1000 --dry-run
uv run python -m experiments.traceaad_v10_15.run --task tsp_construct --run-name trial_1 --budget 1000
uv run python -m experiments.traceaad_v10_15.heldout --run-dir experiments_result/traceaad_v10_15/tsp_construct/trial_1
```

For ACO held-out evaluation, pass `--split test_50`, `test_100`, or `test_200` (CVRP also supports `test_20`). TSP and VRPTW accept `eval_50`, `eval_100`, and `eval_200`; online bin packing accepts `eval_<items>_<capacity>` for 1,000/5,000/10,000 items and capacity 100/500. `--timeout-seconds` can set an explicit held-out limit; the evaluated protocol and limit are saved with each result. Search, independent selection, and held-out results are recorded separately. A completed run selects from the five highest-quality unique training programs. The CLI freezes the OBP and VRPTW **training** timeout to 30 seconds, matching the evaluators' train configuration; the effective timeout is recorded in `run_config.json`.

The formal launcher accepts a prior twenty-run batch manifest so the backend assignment is retained. Inspect its plan with `--dry-run` before launch:

```bash
uv run python -m experiments.traceaad_v10_15.launch_batch --from-batch experiments_result/traceaad_v10_14_3/batch_20260928_v1014_3_template2.json --batch v1015_trial --dry-run
uv run python -m experiments.traceaad_v10_15.heldout_batch --batch-manifest experiments_result/traceaad_v10_15/batch_v1015_trial.json --dry-run
```
