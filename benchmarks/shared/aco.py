"""Execution rules shared by the two fixed ACO frameworks."""

import multiprocessing

from core import set_kill_with_parent


def validate_parameters(n_ants, n_iterations, n_workers):
    for name, value in [("n_ants", n_ants), ("n_iterations", n_iterations), ("n_workers", n_workers)]:
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")


def run_jobs(worker, jobs, n_workers):
    if n_workers <= 1 or len(jobs) <= 1:
        return [worker(job) for job in jobs]
    context = multiprocessing.get_context("spawn")
    with context.Pool(processes=min(n_workers, len(jobs)), initializer=set_kill_with_parent) as pool:
        return pool.map(worker, jobs)
