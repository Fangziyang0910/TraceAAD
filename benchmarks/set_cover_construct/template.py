"""Problem definition, evolved function contract and executable starting heuristic."""

template_program = '''import numpy as np

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

task_description = '''Design a set-priority heuristic for the weighted set-covering problem.
Choose sets whose union covers EVERY element, minimizing the total selected-set cost. Coverage
has rows for elements and columns for sets. Costs are positive. The fixed solver repeatedly
calls score_sets and picks the highest-scoring unselected set covering at least one uncovered
element. After construction it deletes redundant selected sets, examining expensive sets first.
It then attempts removals of expensive selected sets; every partial cover is completed with the
SAME evolved score_sets function and redundant sets are removed again. A repaired cover replaces
the best cover only if its TRUE total cost decreases. Evolve the set priority, not coverage
validation or cost calculation. Return a finite score for EVERY set; eligibility is masked
externally. Inputs expose full coverage relations, costs, selected sets and uncovered elements.'''

function_name = 'score_sets'
