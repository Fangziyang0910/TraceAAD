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
        distance_matrix: Pairwise Euclidean distances with shape (n, n).
        coordinates: Node coordinates with shape (n, 2). Node 0 is the depot.
        demands: Node demands with shape (n,). The depot demand is zero.
        capacity: Capacity shared by all vehicles.

    Returns:
        An (n, n) edge-prior matrix. Larger values make an edge more likely
        to be sampled. Values at or below zero are treated as 1e-9.
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
edges. Use efficient NumPy operations because the function is evaluated many
times.
""".strip()
