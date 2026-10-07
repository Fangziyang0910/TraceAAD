"""Problem definition, evolved function contract and executable starting heuristic."""

template_program = '''import numpy as np

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

task_description = '''Design a move-priority heuristic for the multi-demand multidimensional knapsack problem.
Choose a binary item subset to MAXIMIZE total profit. All resource upper bounds A @ x <= b and
all demand lower bounds G @ x >= d must hold simultaneously. Profits may be negative. The empty
subset is usually infeasible. The fixed search begins with a planted feasible binary solution,
prepared independently of profits. It enumerates
feasible single additions, removals and one-out/one-in exchanges. A five-step item tabu filter
removes recently changed items when any non-tabu moves exist. score_moves ranks the supplied moves.
The search always takes a highest-scoring move, even if true profit decreases, and retains the
highest-profit feasible solution encountered. Evolve this ranking, not the feasibility checker.
The function sees all coefficients, bounds, profits, the current subset and the actual candidate list.
No reference objective or reference solution is passed to it.'''

function_name = 'score_moves'
