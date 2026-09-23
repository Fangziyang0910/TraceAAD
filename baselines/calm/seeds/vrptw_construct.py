# The idea of the algorithm is to select the next node by trading off travel distance, waiting time, and time-window urgency.

SEED_CODE = '''
import numpy as np

def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: float, current_time: float,
                        demands: np.ndarray, distance_matrix: np.ndarray, time_windows: np.ndarray) -> int:
    c1, c2, c3 = 0.6, 0.3, 0.1
    scores = {}

    for node in unvisited_nodes:
        node = int(node)
        arrival = current_time + distance_matrix[current_node][node]
        wait = max(0.0, time_windows[node][0] - arrival)
        slack = max(0.0, time_windows[node][1] - arrival)

        score = (
            c1 * distance_matrix[current_node][node]
            + c2 * wait
            + c3 / (slack + 1.0)
        )
        scores[node] = score

    next_node = min(scores, key=scores.get)
    return next_node
'''.strip()

SEED_IDEA = 'The idea of the algorithm is to select the next node by trading off travel distance, waiting time, and time-window urgency.'
