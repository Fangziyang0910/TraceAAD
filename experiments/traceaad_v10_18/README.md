# TraceAAD V10.18

The mechanism is specified in [the design document](../../docs/01-搜索方法/新设计的开发过程.md); the diagnosis that motivated it is in [the V10.17 diagnosis](../../docs/03-现象与检验/2026-10-04-短程改写诊断与首版门槛.md). Everything of V10.17 is kept (programs and generation events, experience-weighted starting points, measured outcomes, design-anchored Refine/Crossover goals) except how a new Explore design is treated. Every new program an Explore proposal produces opens an exploration. An Explore draw (share 0.40; Refine 0.35, Crossover 0.25) develops the open exploration if there is one and proposes a new design otherwise. Each Develop step starts from the best version the design has reached and sees the design's own formation from its first version, the source program only as a stated score and Design, and all other development attempts on the design. Development lasts at least 2 steps, ends after 2 steps in a row without a better version, after 8 steps, or as soon as the design's best version ranks among the five best programs of the search (also at proposal, without development).

```bash
uv run python -m experiments.traceaad_v10_18.run --task tsp_construct --run-name trial_1 --budget 1000 --dry-run
uv run python -m experiments.traceaad_v10_18.run --task tsp_construct --run-name trial_1 --budget 1000
uv run python -m experiments.traceaad_v10_18.heldout --run-dir experiments_result/traceaad_v10_18/tsp_construct/trial_1
```

`launch_local.py --batch <name> [--launch]` starts the 15-run batch on this machine (five tasks x three repeats, seeds 0-2, the two server3 Qwen services alternated 8/7, four evaluation workers). Evaluations here run faster than on server3, so comparisons with the server3 batches carry a host difference. `heldout_batch.py --batch-manifest <manifest>` evaluates all selected programs. Method tests: `tests/method/test_traceaad_v1018.py`.
