template_program = '''
import numpy as np
def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    """
    Design a novel algorithm to select the next node in each step.

    Args:
    current_node: Zero-based ID of the current node in the full graph.
    destination_node: Zero-based ID of the starting node, which the tour returns to.
    unvisited_nodes: 1-D integer array of unvisited full-graph node IDs,
        sorted by increasing distance from current_node.
    distance_matrix: Full symmetric Euclidean (n,n) matrix, with a zero diagonal.

    Return:
    Select an entry from unvisited_nodes and return that full-graph node ID.
    The outer solver appends the final remaining node and return-to-start edge.
    """
    return int(unvisited_nodes[0])
'''

task_description = (
    "The Traveling Salesman Problem asks for a shortest tour that visits each node once and returns to the start. "
    "The graph is Euclidean, with pairwise distances derived from node coordinates. "
    "At each step it receives the current node id, the destination/start node id, an array of unvisited candidate "
    "node ids, and the pairwise distance matrix, and must return the id of the next node to visit. "
    "The fixed outer solver repeatedly calls this function to construct the tour. "
    "The returned node commits the next move; design this computation to improve the length of the completed tour."
)
