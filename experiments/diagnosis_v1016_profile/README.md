# V10.16 profile replay

Does telling Refine where the current algorithm spends its time help it improve TSP parents near the 30 s limit?

- `replay.py`: resamples the last Refine prompt of 51 V10.16 TSP parents (≥18 s of the limit) as logged and with a py-spy line profile of the parent added; children and parents are evaluated locally in one pool.
- `analyze.py`: paired summary; a child's time is scaled to the search host by its ratio to the parent.

Result (2026-10-03, 4 samples per arm and prompt): improved within the limit 9.3% (base) vs 7.4% (profile), paired difference −2.0 pp [−7.8, +2.9]; over the limit 12.7% vs 11.3%. The profile did not help. Data: `experiments_result/diagnosis_v1016_profile/`.
