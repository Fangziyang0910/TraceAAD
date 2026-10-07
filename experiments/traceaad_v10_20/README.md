# TraceAAD V10.20

The mechanism and its reasons are in [the design document](../../docs/01-搜索方法/当前规则引导的搜索.md); the diagnosis that motivated it is in [版本差异与计算使用](../../docs/03-现象与检验/2026-10-06-版本差异与计算使用.md). Everything of V10.17 is kept: experience-weighted starting points, design-anchored Refine/Crossover goals, Explore anchored on the current algorithm, a random eighth of new Explore programs developed with three Refine steps, one repair, final selection with replacement. Three things change:

- **Facts about computation** (all steps): the ACO task texts say the function is called once per instance before the colony starts (they said it "is evaluated many times"); the VRPTW text states that service durations are added to the time but not passed to the function; each measured program states its evaluation time against the limit; the evaluation section states the training instances and the test sets with their limits. Numbers come from `benchmarks/`; the benchmark task files are not edited, so the baselines' prompts do not change.
- **Deepen** (share 0.15): keep the current algorithm's decision rule and use it to guide a search that spends more of the time limit. Its material is what that decision needs: the current algorithm in full with its measured cost against the limit, and the earlier Deepen attempts from it, without the formation path (a path of small edits made the model continue with small edits). Shares: Refine 0.40, Explore 0.25, Crossover 0.20, Deepen 0.15. Deepen attempts count in the experience of their starting program like the other steps.
- **Explore and Deepen proposals both open an exploration**: both bring new computation into the algorithm, so a random eighth of either gets three Refine steps from the best version reached (V10.17's rule).

```bash
uv run python -m experiments.traceaad_v10_20.run --task op_aco --run-name trial_1 --budget 1000 --dry-run
uv run python -m experiments.traceaad_v10_20.run --task op_aco --run-name trial_1 --budget 1000
uv run python -m experiments.traceaad_v10_20.heldout --run-dir experiments_result/traceaad_v10_20/op_aco/trial_1
uv run python -m experiments.traceaad_v10_20.diagnose --run-dir experiments_result/traceaad_v10_20/op_aco/trial_1
```

`launch_local.py --batch <name> [--launch]` starts the 15-run batch on this machine (five tasks × three repeats, seeds 0–2, the two server3 Qwen services alternated, four evaluation workers). Run it on the local host: programs that use more of the time limit score differently on a faster host, so the baseline is V10.18 + V10.19 on the same host. `heldout_batch.py --batch-manifest <manifest>` evaluates the selected programs.

Batch `20261006_local_v1020` was launched on 2026-10-06 at 19:01 (manifest `experiments_result/traceaad_v10_20/batch_20261006_local_v1020.json`; server3 8 runs, server3b 7). The working tree was not committed at launch; the manifest records the commit, `git status` and the hashes of all implementation files. Run held-out on each run soon after it finishes, so that a timeout at the 200 scale shows up early.

[The fixed-parent replay](../diagnosis_v1020_replay/README.md) checked the first draft, whose Deepen used the Refine material: it behaved like Refine, while the same goal without the formation path raised the computation 12–17×. That replay shaped the final material and development rule.

Diagnostics (`diagnostics.json`) add `computation` for every method: the best and top programs' share of the training limit, each step's evaluation time relative to its starting program, timeouts by step, and programs that read the clock; exploration statistics add `by_proposal_action`. Method tests: `tests/method/test_traceaad_v1020.py`.

Implementation: `traceaad/v10_20/` (config, prompts, search class). The search class is V10.17's with `EXPLORING = ("Explore", "Deepen")`; development draws keep V10.17's per-exploration draw. Training sets, test sets and evaluation seeds are unchanged.
