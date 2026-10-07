"""Permutation flow shop: NEH initialization, true local descent and guided perturbations."""

import numpy as np
from numba import njit
from core.evaluate import InvalidEvaluationResult


@njit(cache=True)
def makespan(sequence, times):
    completed = np.zeros(times.shape[1], dtype=np.float64)
    for job in sequence:
        completed[0] += times[job, 0]
        for machine in range(1, times.shape[1]):
            completed[machine] = max(completed[machine], completed[machine - 1]) + times[job, machine]
    return completed[-1]


@njit(cache=True)
def neh(times):
    order = np.argsort(-times.sum(axis=1))
    sequence = order[:1].copy()
    for job in order[1:]:
        best, best_cost = np.empty(len(sequence) + 1, dtype=np.int64), np.inf
        for position in range(len(sequence) + 1):
            candidate = np.empty(len(sequence) + 1, dtype=np.int64)
            candidate[:position] = sequence[:position]
            candidate[position] = job
            candidate[position + 1:] = sequence[position:]
            cost = makespan(candidate, times)
            if cost < best_cost:
                best, best_cost = candidate, cost
        sequence = best
    return sequence


@njit(cache=True)
def best_neighbor(sequence, times, jobs):
    best, best_cost = sequence.copy(), makespan(sequence, times)
    n = len(sequence)
    for job in jobs:
        position = 0
        while sequence[position] != job:
            position += 1
        for destination in range(n):
            if destination == position:
                continue
            candidate = sequence.copy()
            candidate[position], candidate[destination] = candidate[destination], candidate[position]
            cost = makespan(candidate, times)
            if cost < best_cost - 1e-9:
                best, best_cost = candidate, cost
            candidate = sequence.copy()
            if position < destination:
                for k in range(position, destination):
                    candidate[k] = sequence[k + 1]
            else:
                for k in range(position, destination, -1):
                    candidate[k] = sequence[k - 1]
            candidate[destination] = job
            cost = makespan(candidate, times)
            if cost < best_cost - 1e-9:
                best, best_cost = candidate, cost
    return best, best_cost


def solve(data, heuristic, iterations=10, local_passes=3):
    times = np.asarray(data["processing_times"], dtype=np.float64)
    sequence = neh(times)
    all_jobs = np.arange(len(sequence), dtype=np.int64)
    best, best_cost = sequence.copy(), float(makespan(sequence, times))
    for _ in range(iterations):
        for _ in range(local_passes):
            previous = float(makespan(sequence, times))
            sequence, cost = best_neighbor(sequence, times, all_jobs)
            if cost >= previous - 1e-9:
                break
        cost = float(makespan(sequence, times))
        if cost < best_cost:
            best, best_cost = sequence.copy(), cost
        output = heuristic(sequence.copy(), times.copy(), times.shape[1], times.shape[0])
        if not isinstance(output, (tuple, list)) or len(output) != 2:
            raise InvalidEvaluationResult("get_matrix_and_jobs must return (perturbed_times, job_ids)")
        perturbed, jobs = np.asarray(output[0], dtype=float), np.asarray(output[1])
        if perturbed.shape != times.shape or not np.isfinite(perturbed).all() or np.any(perturbed < 0):
            raise InvalidEvaluationResult("perturbed_times must be finite, non-negative and shaped (n_jobs, n_machines)")
        if jobs.ndim != 1 or not 1 <= len(jobs) <= min(5, len(sequence)) or jobs.dtype.kind not in "iu":
            raise InvalidEvaluationResult("job_ids must be a 1-D integer array with 1 to min(5, n_jobs) IDs")
        if len(set(jobs.tolist())) != len(jobs) or np.any(jobs < 0) or np.any(jobs >= len(sequence)):
            raise InvalidEvaluationResult("job_ids must be distinct zero-based job IDs, not sequence positions")
        sequence, _ = best_neighbor(sequence, perturbed, jobs.astype(np.int64))
        cost = float(makespan(sequence, times))
        if cost < best_cost:
            best, best_cost = sequence.copy(), cost
    # Independent final checks use the original job IDs and processing times.
    if sorted(best.tolist()) != list(range(len(best))):
        raise ValueError("outer flow-shop search returned an invalid permutation")
    return best, float(makespan(best, times))
