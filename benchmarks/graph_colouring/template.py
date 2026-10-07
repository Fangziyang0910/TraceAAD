"""Problem definition, evolved function contract and executable starting heuristic."""

template_program = '''import numpy as np

def score_coloring_moves(adjacency: np.ndarray, colors: np.ndarray, moves: np.ndarray, phase: str) -> np.ndarray:
    """Return one finite score per candidate (vertex_id, target_color) row.

    adjacency: symmetric boolean (n,n) matrix, zero diagonal, zero-based IDs.
    colors: integer (n,) array; -1 means uncolored during construction.
    moves: integer (K,2) candidate rows; IDs and colors are zero-based.
    phase="construct": one candidate per uncolored vertex, using its smallest
    conflict-free color. Rank vertices; the outer solver assigns that color.
    phase="repair": recolor conflicting vertices at a fixed smaller color
    budget; candidates can still leave conflicts. Tabu filtering is external.
    Larger scores are preferred; ties choose the first candidate row.
    """
    nodes, targets = moves[:, 0], moves[:, 1]
    if phase == "construct":
        saturation = np.array([len(set(colors[adjacency[v] & (colors >= 0)].tolist())) for v in nodes])
        return saturation * (len(colors) + 1) + adjacency[nodes].sum(axis=1)
    k = int(colors.max()) + 1
    counts = adjacency.astype(np.int32) @ (colors[:, None] == np.arange(k)[None, :]).astype(np.int32)
    return (counts[nodes, colors[nodes]] - counts[nodes, targets]).astype(float)
'''

task_description = '''Design a construction and conflict-repair priority for graph coloring.
Assign every vertex one non-negative integer color so adjacent vertices have different colors.
Minimize the number of distinct colors in the FINAL VALID coloring. The graph is undirected,
without self-loops. During construction the fixed solver proposes, for every uncolored vertex,
its smallest currently conflict-free color; score_coloring_moves ranks these assignments.
It then attempts to eliminate the highest color class, remapping its vertices into fewer colors.
During repair it proposes recolorings of currently conflicting vertices within the reduced color
budget. A seven-step reverse-move tabu filter is external (relaxed if all candidates are tabu).
The returned priorities may guide temporarily conflicting moves. A reduced coloring is accepted
only if ALL conflicts disappear. If repair fails, the best previous valid coloring is returned.
Construction and repair share ONE evolved function; phase states which decision is being made.'''

function_name = 'score_coloring_moves'
