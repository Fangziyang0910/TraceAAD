# V10.20 fixed-parent replay

Does the Deepen step keep the parent's decision rule and add search, and is its first version at least as good as the parent? This checks only the first step of the mechanism in [当前规则引导的搜索](../../docs/01-搜索方法/当前规则引导的搜索.md); the value of sustained development is judged in full searches.

- `replay.py`: for each task, the best training program of one finished local run is the parent (TSP V10.19 rep3, CVRP V10.19 rep2, OP V10.19 rep1, OBP V10.18 rep1, VRPTW V10.19 rep3). The context is that run's facts before the last Refine attempt from the parent, so every arm sees an actual decision state. Arms: R0 V10.17 Refine, R1 V10.20 Refine (new facts), D V10.20 Deepen. Children are evaluated with the search evaluator (training set and limit, seed 730241) on this host; the parent is evaluated again in the same pool before and after.
- `analyze.py`: per task and arm, valid children, children better than the parent, median relative change, evaluation time relative to the parent, and code similarity to the parent.

```bash
PYTHONPATH=. uv run python -m experiments.diagnosis_v1020_replay.replay --samples 8 --workers 6
PYTHONPATH=. uv run python -m experiments.diagnosis_v1020_replay.replay --samples 8 --tasks op_aco cvrp_aco tsp_construct vrptw_construct --arms Dx Dmin
PYTHONPATH=. uv run python -m experiments.diagnosis_v1020_replay.rescore
PYTHONPATH=. uv run python -m experiments.diagnosis_v1020_replay.analyze
```

Decision rule fixed before the run: keep Deepen if, on OP and CVRP, D clearly adds computation and its share of better children and median change are not below R1; drop Deepen if R1 already adds computation with the same gains; if D mostly rewrites the rule and loses, revisit the goal before the batch.

`rescore.py` re-evaluates the OP and CVRP children and parents at eight seeds: the parent is the best of its run at the search seed and is 1%–2% lucky there, so single-seed comparisons are biased against every child.

Result (2026-10-06): with the full context, D adds no computation (median function time 0.9–1.3× the parent, like R0/R1); at eight seeds R0, R1 and D are indistinguishable and close to the parent. Dmin and Dx raise the function time to 12–17× on OP and CVRP, but their first versions are not better (median −0.1% to −1.0% at eight seeds); TSP Dmin loses 3.5%. The batch is not launched. See [the write-up](../../docs/03-现象与检验/2026-10-06-V10.20重放与ACO评价噪声.md).

Data: `experiments_result/diagnosis_v1020_replay/` (`prompts.json`, `parents.json`, `results.jsonl`, `summary.json`).
