"""Problem definition, evolved function contract and executable starting heuristic."""

template_program = '''import numpy as np

def get_matrix_and_jobs(current_sequence: np.ndarray, processing_times: np.ndarray, n_machines: int, n_jobs: int) -> tuple:
    """Return (perturbed_times, job_ids) for one guided perturbation.

    current_sequence: shape (n_jobs,), a permutation of zero-based JOB IDs.
    processing_times: shape (n_jobs, n_machines), ORIGINAL job-by-machine times.
    n_machines and n_jobs: the actual problem dimensions.
    perturbed_times: finite non-negative array of the same shape.
    job_ids: distinct integer JOB IDs indexing matrix rows, length 1..min(5, n_jobs).
    The outer solver tests swaps and insertions involving these jobs on the
    perturbed matrix. True quality always uses the original processing times.
    """
    return processing_times.copy(), current_sequence[:1].copy()
'''

task_description = '''Design a guided-local-search perturbation for the permutation flow-shop scheduling problem.
Every job visits all machines in the SAME machine order; all machines use the SAME job permutation.
Each machine handles one job at a time and each operation runs to completion. Minimize the final
makespan (completion time of the last job on the last machine), measured with the ORIGINAL times.
The fixed solver starts with NEH. Each round performs improving swap/insertion
moves on original times, then calls get_matrix_and_jobs once. It takes the best improving swap or
insertion involving the returned job IDs under perturbed_times, retaining the original sequence
when there is no improving perturbed move. It retains the best TRUE solution throughout all rounds.
Evolve this perturbation function. Perturbed processing times guide move selection; physical job
durations and the reported objective use the original times. Each returned ID identifies a matrix row.'''

function_name = 'get_matrix_and_jobs'
