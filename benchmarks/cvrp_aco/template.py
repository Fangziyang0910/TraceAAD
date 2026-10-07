template_program = '''
import numpy as np


def heuristics(
        distance_matrix: np.ndarray,
        coordinates: np.ndarray,
        demands: np.ndarray,
        capacity: int,
) -> np.ndarray:
    """Return edge desirability values for CVRP ant colony optimization.

    Args:
        distance_matrix: Pairwise Euclidean distances with shape (n, n),
            where n includes the depot. The diagonal is replaced with 1.0.
        coordinates: Node coordinates with shape (n, 2). Node 0 is the depot.
        demands: Node demands with shape (n,). The depot demand is zero.
        capacity: Capacity shared by all vehicles.

    Returns:
        An (n, n) edge-prior matrix. Larger values make an edge more likely
        to be sampled. The solver applies maximum(value + 1e-9, 1e-9).
        This function is called once per instance, before the ant colony starts.
    """
    return 1.0 / distance_matrix
'''

task_description = """
Design an edge-prior heuristic for Ant Colony Optimization (ACO) on the
Capacitated Vehicle Routing Problem (CVRP). A solution consists of routes that
start and end at depot node 0, visit every customer exactly once, and never
exceed vehicle capacity. ACO combines the returned heuristic matrix with its
pheromone matrix to sample feasible moves. The objective is to minimize the
best total route length found across all ants and iterations.

The function receives the pairwise distance matrix, node coordinates, customer
demands, and vehicle capacity. It must return a finite matrix with the same
shape as the distance matrix. Larger entries indicate more promising directed
edges. The function is called once per instance, before the ant colony starts.
ACO samples a move in proportion to pheromone times the returned prior,
after masking visited customers and customers exceeding remaining capacity.
The distance matrix diagonal is 1.0. The solver adds 1e-9 to each prior value
and clips it below at 1e-9. Within the evaluation time limit, the function
may perform any computation on the complete instance before returning its prior.
""".strip()
