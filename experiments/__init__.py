"""Experiment entry points.

numpy's BLAS/OpenMP pools stay single-threaded: evaluations run in their own
processes and many searches share one host, so per-process thread pools
would only oversubscribe the CPUs. This must run before numpy is imported,
which ``python -m experiments....`` guarantees.
"""

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "1")
