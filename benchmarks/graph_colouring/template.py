"""Whole-solver contract and a fully editable DSATUR starting program."""

template_program = '''import numpy as np

def solve(adjacency: np.ndarray) -> np.ndarray:
    """Return a nonnegative integer color for every vertex of the input graph."""
    n = len(adjacency)
    colors = np.full(n, -1, dtype=np.int64)
    degrees = adjacency.sum(axis=1)
    for _ in range(n):
        uncolored = np.flatnonzero(colors < 0)
        saturation = np.array([
            len(set(colors[adjacency[v] & (colors >= 0)].tolist()))
            for v in uncolored
        ])
        node = uncolored[np.argmax(saturation * (n + 1) + degrees[uncolored])]
        used = set(colors[adjacency[node]].tolist())
        color = 0
        while color in used:
            color += 1
        colors[node] = color
    return colors
'''

task_description = '''Design a complete solver for graph coloring.
The input adjacency is a symmetric boolean (n, n) matrix with a zero diagonal.
Vertex IDs are zero-based. Return an integer array of shape (n,) assigning a
nonnegative color to every vertex. Adjacent vertices must have different colors.
Minimize the number of DISTINCT colors in the returned valid coloring; color
labels need not be consecutive. Return the coloring itself, not a score.
The supplied DSATUR construction is editable starting code. You may change or
replace the entire algorithm and add helper functions.
The solver is called once per instance and receives the complete graph.'''

function_name = 'solve'
