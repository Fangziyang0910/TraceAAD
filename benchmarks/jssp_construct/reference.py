"""Fixed construction used to report dataset reference values."""

import numpy as np


def mwkr(processing_times: np.ndarray, machine_order: np.ndarray) -> np.ndarray:
    """Return the start time of every operation, indexed by job and operation."""
    n, m = processing_times.shape
    next_operation = np.zeros(n, dtype=np.int64)
    job_ready = np.zeros(n)
    machine_ready = np.zeros(m)
    starts = np.zeros((n, m))
    for _ in range(n * m):
        jobs = np.flatnonzero(next_operation < m)
        operations = next_operation[jobs]
        machines = machine_order[jobs, operations]
        earliest = np.maximum(job_ready[jobs], machine_ready[machines])
        completion = earliest + processing_times[jobs, operations]
        pivot = int(np.argmin(completion))
        conflict = (machines == machines[pivot]) & (earliest < completion[pivot])
        candidates = np.column_stack((jobs[conflict], operations[conflict]))
        remaining_work = np.array([processing_times[j, k:].sum() for j, k in candidates])
        job, operation = candidates[np.argmax(remaining_work)]
        machine = machine_order[job, operation]
        start = max(job_ready[job], machine_ready[machine])
        end = start + processing_times[job, operation]
        starts[job, operation] = start
        next_operation[job] += 1
        job_ready[job], machine_ready[machine] = end, end
    return starts
