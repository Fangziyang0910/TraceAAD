"""Experiment entry points.

Evaluations are limited by CPU seconds (see ``core.evaluate``), so numpy's
BLAS/OpenMP pools stay single-threaded: CPU time then measures one core's
work, and concurrent searches do not oversubscribe the host. This must run
before numpy is imported, which ``python -m experiments....`` guarantees.
"""

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "1")
