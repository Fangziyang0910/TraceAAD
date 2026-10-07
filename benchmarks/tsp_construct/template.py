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
    A full-graph node ID contained in unvisited_nodes, not an array position.
    The outer solver appends the final remaining node and return-to-start edge.
    """
    next_node = unvisited_nodes[0]

    return next_node
'''

task_description = (
    "The Traveling Salesman Problem asks for a shortest tour that visits each node once and returns to the start. "
    "Instances are generated from node coordinates, but the constructive heuristic does not receive coordinates. "
    "At each step it receives the current node id, the destination/start node id, an array of unvisited candidate "
    "node ids, and the pairwise distance matrix, and must return the id of the next node to visit. "
    "Help me design a novel algorithm to select the next node in each step."
)
