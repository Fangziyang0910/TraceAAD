"""Giffler-Thompson construction with independent final schedule validation."""
import numpy as np
from .._fixed_evaluation import FixedEvaluation, scores
from . import dataset
from .template import template_program, task_description, function_name


def validate_schedule(times, machines, starts):
    times, machines, starts = np.asarray(times), np.asarray(machines), np.asarray(starts)
    n, m = times.shape
    if starts.shape != times.shape or not np.isfinite(starts).all() or np.any(starts < 0):
        raise ValueError('job-shop start times must be finite, nonnegative and match durations')
    ends = starts + times
    if np.any(starts[:, 1:] < ends[:, :-1]):
        raise ValueError('job-shop schedule violates job precedence')
    for machine in range(m):
        jobs, operations = np.where(machines == machine)
        order = np.argsort(starts[jobs, operations], kind='stable')
        if np.any(starts[jobs[order[1:]], operations[order[1:]]] < ends[jobs[order[:-1]], operations[order[:-1]]]):
            raise ValueError('job-shop schedule overlaps on a machine')
    return float(ends.max())


def solve(data, heuristic):
    times, machines = data['processing_times'], data['machine_order']
    n, m = times.shape
    next_op = np.zeros(n, dtype=np.int64)
    job_ready, machine_ready = np.zeros(n), np.zeros(m)
    starts = np.zeros((n, m))
    for _ in range(n*m):
        jobs = np.flatnonzero(next_op < m)
        operations = next_op[jobs]
        target_machines = machines[jobs, operations]
        earliest = np.maximum(job_ready[jobs], machine_ready[target_machines])
        completion = earliest + times[jobs, operations]
        pivot = int(np.argmin(completion))
        conflict = (target_machines == target_machines[pivot]) & (earliest < completion[pivot])
        candidates = np.column_stack((jobs[conflict], operations[conflict]))
        ranking = heuristic(times.copy(), machines.copy(), next_op.copy(), job_ready.copy(),
                            machine_ready.copy(), candidates.copy())
        job, operation = candidates[int(np.argmax(scores(ranking, len(candidates), function_name)))]
        machine = int(machines[job, operation])
        start = max(job_ready[job], machine_ready[machine])
        end = start + times[job, operation]
        starts[job, operation] = start
        next_op[job] += 1
        job_ready[job], machine_ready[machine] = end, end
    return starts, validate_schedule(times, machines, starts)


class JSSPEvaluation(FixedEvaluation):
    DATASET = dataset
    TEMPLATE = template_program
    DESCRIPTION = task_description
    FUNCTION_NAME = function_name
    SOLVER = staticmethod(solve)
    DEFAULT_SETTINGS = {}
