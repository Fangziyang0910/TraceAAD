# The idea of the algorithm is to return the element-wise reciprocal of the distance matrix

SEED_CODE = '''
import numpy as np

def heuristics(distance_matrix: np.ndarray, coordinates: np.ndarray, demands: np.ndarray, capacity: int) -> np.ndarray:
    return 1 / distance_matrix
'''.strip()

SEED_IDEA = 'The idea of the algorithm is to return the element-wise reciprocal of the distance matrix'
