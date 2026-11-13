template_program = '''
import numpy as np


def heuristics(
        prize: np.ndarray,
        distance: np.ndarray,
        maxlen: float,
) -> np.ndarray:
    """Return edge desirability values for OP ant colony optimization.

    Args:
        prize: Node prizes with shape (n,). Node 0 is the depot.
        distance: Pairwise Euclidean distances with shape (n, n).
            Diagonal entries are large sentinels so self-loops are unused.
        maxlen: Maximum allowed tour length (return-to-depot constrained).

    Returns:
        An (n, n) edge-prior matrix. Larger values make an edge more likely
        to be sampled. Values at or below zero are treated as 1e-9.
    """
    return prize[np.newaxis, :] / distance
'''

task_description = """
Design an edge-prior heuristic for Ant Colony Optimization (ACO) on the
Orienteering Problem (OP). A tour starts at depot node 0, collects prizes at
visited nodes, and returns to the depot with total travel length at most
``maxlen``. The objective is to maximize the total collected prize.

ACO combines the returned heuristic matrix with its pheromone matrix to sample
feasible moves. The function receives node prizes, the pairwise distance
matrix, and ``maxlen``. It must return a finite matrix with the same shape as
the distance matrix. Larger entries indicate more promising directed edges.
The function is called once per instance, before the ant colony starts.
Its returned matrix is reused across all ants and iterations.
At each construction step, ACO multiplies this fixed prior and pheromone by a
dynamic feasibility mask based on the distance already travelled, the next
edge, and the return distance to the depot. Candidates exceeding ``maxlen``
are masked out before sampling. The prior supplies relative weights among
feasible moves, and ACO constructs the route through masked sampling. Design how the instance
information expresses the value of visiting nodes and connecting visits within the tour-length constraint.
""".strip()
