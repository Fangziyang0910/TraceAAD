# V10.15-5 diagnosis studies

Supporting experiments for [the V10.15-5 diagnosis](../../docs/03-现象与检验/2026-10-01-生成目标与计算限制.md).

`explore_study.py` is a fixed-parent paired comparison of Explore variants. For each V10.15-5 run it truncates the archive at attempt 600 (`EXPLORE_STUDY_CUT`), takes the best distinct score classes as parents, and builds every arm's prompt from the same parent and archive; arms differ only in the Explore instruction, the `[Evaluation]` time wording and the sampling profile. Children are evaluated on the training evaluator with a 120 s limit and their wall time is recorded, so the analysis can apply the 30 s search limit. Results are appended to `experiments_result/diagnosis_v1015_5/explore_study/<task>.jsonl`; reruns skip finished jobs.

```bash
PYTHONPATH=. uv run python -m experiments.diagnosis_v1015_5.explore_study tsp_construct 4 5 cur cur_time restr_time v1_time
PYTHONPATH=. uv run python -m experiments.diagnosis_v1015_5.analyze_explore tsp_construct 30
```

The main outcome is whether a child beats the run's best score at the cut, not whether it beats its parent: structural rewrites are worse in the median and pay off in the tail.
