"""Problem definition, evolved function contract and executable starting heuristic."""

template_program = '''import numpy as np

def get_matrix_and_jobs(current_sequence: np.ndarray, processing_times: np.ndarray, n_machines: int, n_jobs: int) -> tuple:
    """Return (perturbed_times, job_ids) for one guided perturbation.

    current_sequence: shape (n_jobs,), a permutation of zero-based JOB IDs.
    processing_times: shape (n_jobs, n_machines), ORIGINAL job-by-machine times.
    n_machines and n_jobs: the actual dimensions, not search budgets.
    perturbed_times: finite non-negative array of the same shape.
    job_ids: distinct integer JOB IDs, length 1..min(5, n_jobs), not positions.
    The outer solver tests swaps and insertions involving these jobs on the
    perturbed matrix. True quality always uses the original processing times.
    """
    jobs = np.argsort(-processing_times.sum(axis=1), kind="stable")[:min(5, n_jobs)]
    perturbed = processing_times.astype(float).copy()
    perturbed[jobs] *= 1.25
    return perturbed, jobs
'''

task_description = '''Design a guided-local-search perturbation for the permutation flow-shop scheduling problem.
Every job visits all machines in the SAME machine order; all machines use the SAME job permutation.
Each machine handles one job at a time and operations cannot be interrupted. Minimize the final
makespan (completion time of the last job on the last machine), measured with the ORIGINAL times.
The fixed solver starts with NEH. Each round performs up to local_passes improving swap/insertion
moves on original times, then calls get_matrix_and_jobs once. It takes the best improving swap or
insertion involving the returned job IDs under perturbed_times, retaining the original sequence
when there is no improving perturbed move. It retains the best TRUE solution throughout all rounds.
Evolve only this perturbation function. Perturbed processing times guide moves, never change the
physical job durations or the reported objective. IDs index rows of the matrix, not sequence positions.'''

function_name = 'get_matrix_and_jobs'
