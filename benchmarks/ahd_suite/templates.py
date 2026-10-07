"""Executable seed functions and exact contracts of the four evolved components."""

FSSP_TEMPLATE = '''import numpy as np

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

MDMKP_TEMPLATE = '''import numpy as np

def score_moves(profits: np.ndarray, upper_coefficients: np.ndarray, upper_limits: np.ndarray, lower_coefficients: np.ndarray, lower_limits: np.ndarray, selected: np.ndarray, moves: np.ndarray) -> np.ndarray:
    """Rank already feasible moves; return one finite score per row of moves.

    profits: shape (n,), possibly negative item profits.
    upper_coefficients/upper_limits: shapes (m,n)/(m,), constraints A @ x <= b.
    lower_coefficients/lower_limits: shapes (q,n)/(q,), constraints G @ x >= d.
    selected: shape (n,), current feasible binary selection.
    moves: shape (K,2), rows (remove_item, add_item); -1 means no item.
    Each move flips the specified selected item off and unselected item on.
    All supplied moves satisfy EVERY upper and lower constraint after the move.
    Larger scores are preferred; ties choose the first supplied row.
    """
    padded = np.r_[profits, 0.0]
    return padded[moves[:, 1]] - padded[moves[:, 0]]
'''

GRAPH_TEMPLATE = '''import numpy as np

def score_coloring_moves(adjacency: np.ndarray, colors: np.ndarray, moves: np.ndarray, phase: str) -> np.ndarray:
    """Return one finite score per candidate (vertex_id, target_color) row.

    adjacency: symmetric boolean (n,n) matrix, zero diagonal, zero-based IDs.
    colors: integer (n,) array; -1 means uncolored during construction.
    moves: integer (K,2) candidate rows; IDs and colors are zero-based.
    phase="construct": one candidate per uncolored vertex, using its smallest
    conflict-free color. Rank vertices; the outer solver assigns that color.
    phase="repair": recolor conflicting vertices at a fixed smaller color
    budget; candidates can still leave conflicts. Tabu filtering is external.
    Larger scores are preferred; ties choose the first candidate row.
    """
    nodes, targets = moves[:, 0], moves[:, 1]
    if phase == "construct":
        saturation = np.array([len(set(colors[adjacency[v] & (colors >= 0)].tolist())) for v in nodes])
        return saturation * (len(colors) + 1) + adjacency[nodes].sum(axis=1)
    k = int(colors.max()) + 1
    counts = adjacency.astype(np.int32) @ (colors[:, None] == np.arange(k)[None, :]).astype(np.int32)
    return (counts[nodes, colors[nodes]] - counts[nodes, targets]).astype(float)
'''

SET_COVER_TEMPLATE = '''import numpy as np

def score_sets(costs: np.ndarray, coverage: np.ndarray, selected: np.ndarray, uncovered: np.ndarray) -> np.ndarray:
    """Return finite scores for ALL sets, shape (n_sets,).

    costs: positive costs, shape (n_sets,).
    coverage: boolean (n_elements,n_sets); True means the column covers the row.
    selected: boolean (n_sets,), current partial cover.
    uncovered: boolean (n_elements,), elements not yet covered.
    The outer solver masks selected sets and sets adding no uncovered element,
    then chooses the largest score (ties choose the lowest set ID).
    This function also completes partial covers during remove-and-repair.
    """
    gain = coverage[uncovered].sum(axis=0)
    return gain / costs
'''

TEMPLATES = {"fssp_gls": FSSP_TEMPLATE, "mdmkp_search": MDMKP_TEMPLATE,
             "graph_colouring": GRAPH_TEMPLATE, "set_cover_construct": SET_COVER_TEMPLATE}

DESCRIPTIONS = {
    "fssp_gls": """Design a guided-local-search perturbation for the permutation flow-shop scheduling problem.
Every job visits all machines in the SAME machine order; all machines use the SAME job permutation.
Each machine handles one job at a time and operations cannot be interrupted. Minimize the final
makespan (completion time of the last job on the last machine), measured with the ORIGINAL times.
The fixed solver starts with NEH. Each round performs up to local_passes improving swap/insertion
moves on original times, then calls get_matrix_and_jobs once. It takes the best improving swap or
insertion involving the returned job IDs under perturbed_times, retaining the original sequence
when there is no improving perturbed move. It retains the best TRUE solution throughout all rounds.
Evolve only this perturbation function. Perturbed processing times guide moves, never change the
physical job durations or the reported objective. IDs index rows of the matrix, not sequence positions.""",
    "mdmkp_search": """Design a move-priority heuristic for the multi-demand multidimensional knapsack problem.
Choose a binary item subset to MAXIMIZE total profit. All resource upper bounds A @ x <= b and
all demand lower bounds G @ x >= d must hold simultaneously. Profits may be negative. The empty
subset is usually infeasible. The fixed search begins with a prepared feasible binary solution,
prepared independently of profits: generated instances use a planted feasible witness, and
standard instances use an offline ZERO-objective feasibility solve. It enumerates
feasible single additions, removals and one-out/one-in exchanges. A five-step item tabu filter
removes recently changed items when any non-tabu moves exist. score_moves ranks the supplied moves.
The search always takes a highest-scoring move, even if true profit decreases, and retains the
highest-profit feasible solution encountered. Evolve this ranking, not the feasibility checker.
The function sees all coefficients, bounds, profits, the current subset and the actual candidate list.
No reference objective or reference solution is passed to it.""",
    "graph_colouring": """Design a construction and conflict-repair priority for graph coloring.
Assign every vertex one non-negative integer color so adjacent vertices have different colors.
Minimize the number of distinct colors in the FINAL VALID coloring. The graph is undirected,
without self-loops. During construction the fixed solver proposes, for every uncolored vertex,
its smallest currently conflict-free color; score_coloring_moves ranks these assignments.
It then attempts to eliminate the highest color class, remapping its vertices into fewer colors.
During repair it proposes recolorings of currently conflicting vertices within the reduced color
budget. A seven-step reverse-move tabu filter is external (relaxed if all candidates are tabu).
The returned priorities may guide temporarily conflicting moves. A reduced coloring is accepted
only if ALL conflicts disappear. If repair fails, the best previous valid coloring is returned.
Construction and repair share ONE evolved function; phase states which decision is being made.""",
    "set_cover_construct": """Design a set-priority heuristic for the weighted set-covering problem.
Choose sets whose union covers EVERY element, minimizing the total selected-set cost. Coverage
has rows for elements and columns for sets. Costs are positive. The fixed solver repeatedly
calls score_sets and picks the highest-scoring unselected set covering at least one uncovered
element. After construction it deletes redundant selected sets, examining expensive sets first.
It then attempts removals of expensive selected sets; every partial cover is completed with the
SAME evolved score_sets function and redundant sets are removed again. A repaired cover replaces
the best cover only if its TRUE total cost decreases. Evolve the set priority, not coverage
validation or cost calculation. Return a finite score for EVERY set; eligibility is masked
externally. Inputs expose full coverage relations, costs, selected sets and uncovered elements.""",
}

FUNCTION_NAMES = {"fssp_gls": "get_matrix_and_jobs", "mdmkp_search": "score_moves",
                  "graph_colouring": "score_coloring_moves", "set_cover_construct": "score_sets"}
