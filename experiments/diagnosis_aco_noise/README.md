# ACO training-score noise

How much of an OP or CVRP training score is ACO-seed noise, and how much of the late search progress is real?

- `audit.py`: for every finished OP and CVRP run of V10.17–V10.19, the programs that set a new training best from candidate 50 on, and ten random valid programs per run, are evaluated on the training set under the search condition at eight seeds (730241 and 1–7).
- `analyze.py`: per-program seed spread; "luck" (search-seed score minus the mean of the other seeds) for frontier and random programs; per run, the archived versus seed-mean change of the best program after budget 250 and 500.

```bash
PYTHONPATH=. uv run python -m experiments.diagnosis_aco_noise.audit --workers 6
PYTHONPATH=. uv run python -m experiments.diagnosis_aco_noise.analyze
```

Result (2026-10-06): OP spread 1.14% per program, frontier luck +1.60% (random +0.22%); after budget 250 the archived best rose 1.32% (median) but the seed mean only 0.42%, and 3 of 8 runs made no real progress. CVRP spread 0.53%, luck +0.65%; late progress is real in every run. See [the write-up](../../docs/03-现象与检验/2026-10-06-V10.20重放与ACO评价噪声.md). Data: `experiments_result/diagnosis_aco_noise/`.
