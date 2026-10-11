"""Evaluate complete job-shop schedules with independent constraint checks."""
import numpy as np
from core.evaluate import InvalidEvaluationResult
from .._fixed_evaluation import FixedEvaluation
from . import dataset
from .template import template_program, task_description, function_name


def validate_schedule(times, machines, starts):
    times, machines = np.asarray(times), np.asarray(machines)
    try:
        starts = np.asarray(starts)
    except (TypeError, ValueError) as exc:
        raise InvalidEvaluationResult('solve must return a numeric start-time matrix') from exc
    n, m = times.shape
    if (starts.shape != times.shape or starts.dtype.kind not in 'iuf'
            or not np.isfinite(starts).all() or np.any(starts < 0)):
        raise InvalidEvaluationResult('job-shop start times must be finite, nonnegative and match durations')
    # Add in floating point so integer overflow cannot hide constraint violations.
    ends = starts.astype(float) + times
    if not np.isfinite(ends).all() or np.any(ends <= starts):
        raise InvalidEvaluationResult('job-shop completion times must be finite and exceed start times')
    if np.any(starts[:, 1:] < ends[:, :-1]):
        raise InvalidEvaluationResult('job-shop schedule violates job precedence')
    for machine in range(m):
        jobs, operations = np.where(machines == machine)
        order = np.argsort(starts[jobs, operations], kind='stable')
        if np.any(starts[jobs[order[1:]], operations[order[1:]]] < ends[jobs[order[:-1]], operations[order[:-1]]]):
            raise InvalidEvaluationResult('job-shop schedule overlaps on a machine')
    return float(ends.max())


def solve(data, solver):
    times, machines = data['processing_times'], data['machine_order']
    starts = solver(times.copy(), machines.copy())
    return starts, validate_schedule(times, machines, starts)


class JSSPEvaluation(FixedEvaluation):
    DATASET = dataset
    TEMPLATE = template_program
    DESCRIPTION = task_description
    FUNCTION_NAME = function_name
    SOLVER = staticmethod(solve)
    DEFAULT_SETTINGS = {}
    INTERFACE = 'full-solver-v1'
