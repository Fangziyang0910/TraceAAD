# TraceAAD V10.15

The frozen mechanism is specified in [the design document](../../docs/01-主线机制设计/TraceAAD-V10.15-机制设计.md). Search uses the quality-only ESS distribution and immediately admits each valid unique candidate. The formation path is shown only in generation prompts.

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
