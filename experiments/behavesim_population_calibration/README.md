# BehaveSim population geometry calibration

This experiment calibrates OBP probe scale and ACO random streams, then runs
only population-level distance/fitness sanity checks on a frozen V10.6 archive.

```bash
.venv/bin/python -m experiments.behavesim_population_calibration.run prepare
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 .venv/bin/python -m experiments.behavesim_population_calibration.run profile --workers 12
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m experiments.behavesim_population_calibration.run analyze
```

Raw profiles stay under `experiments/behavesim_population_calibration/raw/`. The reviewed protocol and result
summary live under `docs/03-机制验证/03-算法行为几何/2026-09-07-BehaveSim群体几何校准/`.
