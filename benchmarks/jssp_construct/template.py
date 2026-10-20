"""Whole-solver contract and editable Giffler-Thompson/MWKR starting code."""

template_program = '''import numpy as np

def solve(processing_times: np.ndarray, machine_order: np.ndarray) -> np.ndarray:
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
'''

function_name = 'solve'
task_description = '''Design a complete solver for STATIC Job Shop Scheduling (JSSP).
There are n jobs and m machines. Each job has m operations in a fixed precedence
order and visits every machine exactly once in its own machine order. Operations
are non-preemptive, and a machine can process only one operation at a time.
processing_times[j, k] is the positive duration of job j's kth operation.
machine_order[j, k] is its zero-based machine ID; each row is a permutation of
0..m-1. Both inputs have shape (n, m), with columns indexing OPERATION positions.
Return a finite nonnegative numeric array starts of shape (n, m). starts[j, k]
is the start time of that operation. Each operation must start after its job's
preceding operation finishes. Operations on the same machine must not overlap.
Minimize makespan, max(starts + processing_times). Return start times, not a score.
The supplied Giffler-Thompson/MWKR construction is editable starting code.
You may change or replace the entire algorithm and add helper functions.
solve is called once per instance with the complete problem.
This is deterministic static makespan scheduling, not dynamic arrivals or a flexible-machine assignment problem.'''
