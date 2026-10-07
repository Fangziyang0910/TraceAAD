"""Static job-shop problem and the operation-priority function contract."""

template_program = '''import numpy as np

def score_operations(processing_times: np.ndarray, machine_order: np.ndarray,
                     next_operation: np.ndarray, job_ready: np.ndarray,
                     machine_ready: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Rank the supplied Giffler-Thompson conflict set; larger scores win.

    processing_times: (jobs, machines), positive durations in job operation order.
    machine_order: same shape; each row is a permutation of zero-based machine IDs.
    next_operation: (jobs,), index of each job's next operation; machines means done.
    job_ready: (jobs,), completion time of each job's last scheduled operation.
    machine_ready: (machines,), completion time of the last scheduled operation.
    candidates: (K,2), rows (job ID, operation INDEX within that job), not machine IDs.
    Every row is the next operation of that job. Only these rows may be ranked.
    Return a finite (K,) vector in candidate order. Ties choose the first row.
    """
    return np.array([processing_times[j, k:].sum() for j, k in candidates], dtype=float)
'''

function_name = 'score_operations'
task_description = '''Design a dispatching priority rule for STATIC Job Shop Scheduling (JSSP).
There are n jobs and m machines. Each job has m operations in a fixed precedence order,
and visits every machine exactly once in its own machine order. Each operation has a
positive processing duration. Operations are non-preemptive, and a machine can process
only one operation at a time. Minimize the makespan (last operation completion time).

The fixed Giffler-Thompson construction considers every unfinished job's next operation.
Its earliest start is max(job_ready[j], machine_ready[machine_order[j,k]]).
It chooses the operation with the earliest possible completion (ties: smaller job ID).
Among next operations on THAT machine, those whose earliest start is strictly before
this earliest completion form the conflict set. score_operations ranks only that set.
The chosen operation starts at its earliest feasible time; the outer scheduler updates
job and machine completion times. It repeats until every operation is scheduled.
The returned priority vector does not specify start times or change the machine order.
You receive all processing times and machine routes, current progress and readiness,
and the actual candidate list. You may compute future scheduling consequences to rank it.
The reference schedule is not an input. This is deterministic static makespan scheduling,
not dynamic arrivals, fuzzy durations or a flexible-machine assignment problem.'''
