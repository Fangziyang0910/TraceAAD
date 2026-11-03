"""Whole-solver contract and an editable serial schedule."""

template_program = '''import numpy as np

def solve(processing_times: np.ndarray, machine_order: np.ndarray) -> np.ndarray:
    """Return the start time of every operation, indexed by job and operation."""
    durations = processing_times.ravel()
    return (np.cumsum(durations, dtype=float) - durations).reshape(processing_times.shape)
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
The supplied construction executes all operations serially in job order.
You may change or replace the entire algorithm and add helper functions.
solve is called once per instance with the complete problem.
This is deterministic static makespan scheduling, not dynamic arrivals or a flexible-machine assignment problem.'''
