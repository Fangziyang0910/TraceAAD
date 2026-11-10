"""Whole-solver contract and a one-color-per-vertex starting program."""

template_program = '''import numpy as np

def solve(adjacency: np.ndarray) -> np.ndarray:
    """Return a nonnegative integer color for every vertex of the input graph."""
    return np.arange(len(adjacency), dtype=np.int64)
'''

task_description = '''Design a complete solver for graph coloring.
The input adjacency is a symmetric boolean (n, n) matrix with a zero diagonal.
Vertex IDs are zero-based. Return an integer array of shape (n,) assigning a
nonnegative color to every vertex. Adjacent vertices must have different colors.
Minimize the number of DISTINCT colors in the returned valid coloring; color
labels may use any nonnegative integers. Return the vertex-to-color array.
The supplied construction assigns one color per vertex. You may change or
replace the entire algorithm and add helper functions.
The solver is called once per instance and receives the complete graph.'''

function_name = 'solve'
