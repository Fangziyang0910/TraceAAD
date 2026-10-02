# TraceAAD V10.16

The mechanism is specified in [the design document](../../docs/01-主线机制设计/TraceAAD-V10.16-机制设计.md). The population is the search's experience: programs (identified by normalized code, failed ones included) and the generation events between them. Duplicates, known failures and repairs are events linked to programs; only identical code counts as a duplicate. Parents are drawn by training quality times each program's experience, `(k + 3r) / (n + 3)` over the attempts started from it. Refine, Explore and Crossover see the attempts that started from the current algorithm with their measured outcomes; Explore also sees how the search best improved. Every evaluation records the calls to the target function and the time inside it, also when it times out. A finalist that fails on the selection set is replaced by the next program in the training ranking (at most five replacements).

```bash
uv run python -m experiments.traceaad_v10_16.run --task tsp_construct --run-name trial_1 --budget 1000 --dry-run
uv run python -m experiments.traceaad_v10_16.run --task tsp_construct --run-name trial_1 --budget 1000
uv run python -m experiments.traceaad_v10_16.heldout --run-dir experiments_result/traceaad_v10_16/tsp_construct/trial_1
```

Training, selection and held-out limits and splits are the same as for [V10.15](../traceaad_v10_15/README.md). `launch_server3.py --batch <name> [--launch]` starts the 15-run server3 batch arranged like V10.15-6 (five tasks x three repeats, seeds 0-2, two Qwen services). Run the method tests (`tests/method/test_traceaad_v1016.py`) and a short smoke run before a formal batch.
