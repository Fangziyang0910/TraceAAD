"""Construct a valid coloring, then try fewer colors through tabu conflict repair."""

import numpy as np

from .._fixed_evaluation import FixedEvaluation, scores
from . import dataset
from .template import function_name, task_description, template_program


def conflict_counts(adjacency, colors, k):
    one_hot = colors[:, None] == np.arange(k)[None, :]
    return adjacency.astype(np.int32) @ one_hot.astype(np.int32)


def valid(adjacency, colors):
    return np.all(colors >= 0) and not np.any(adjacency & (colors[:, None] == colors[None, :]))


def solve(data, heuristic, repair_steps=100, reduction_attempts=4, tabu_tenure=7):
    adjacency = data["adjacency"]
    n = len(adjacency)
    colors = np.full(n, -1, dtype=np.int64)
    for _ in range(n):
        uncolored = np.flatnonzero(colors < 0)
        k = max(0, int(colors.max()) + 1)
        counts = conflict_counts(adjacency, colors, k + 1)
        targets = np.argmax(counts[uncolored] == 0, axis=1)
        moves = np.column_stack((uncolored, targets))
        value = heuristic(adjacency.copy(), colors.copy(), moves.copy(), "construct")
        node, color = moves[int(np.argmax(scores(value, len(moves), "score_coloring_moves")))]
        colors[node] = color
    best = colors.copy()
    for _ in range(reduction_attempts):
        k = int(best.max())  # one fewer color than the current valid coloring
        if k < 1:
            break
        trial = best.copy()
        removed = np.flatnonzero(trial == k)
        trial[removed] = removed % k
        tabu = np.zeros((n, k), dtype=np.int64)
        for step in range(repair_steps):
            counts = conflict_counts(adjacency, trial, k)
            conflict_nodes = np.flatnonzero(counts[np.arange(n), trial] > 0)
            if not len(conflict_nodes):
                break
            moves = np.array([(int(v), c) for v in conflict_nodes for c in range(k) if c != trial[v]], dtype=np.int64)
            if not len(moves):
                break
            allowed = tabu[moves[:, 0], moves[:, 1]] <= step
            if allowed.any():
                moves = moves[allowed]
            value = heuristic(adjacency.copy(), trial.copy(), moves.copy(), "repair")
            node, color = moves[int(np.argmax(scores(value, len(moves), "score_coloring_moves")))]
            old = trial[node]
            trial[node] = color
            tabu[node, old] = step + tabu_tenure + 1
        if not valid(adjacency, trial):
            break
        # Compress labels so that every future reduction removes an actual color.
        _, best = np.unique(trial, return_inverse=True)
        best = best.astype(np.int64)
    if not valid(adjacency, best):
        raise ValueError("outer graph search returned a conflicting coloring")
    return best, int(len(np.unique(best)))


class GraphColouringEvaluation(FixedEvaluation):
    DATASET = dataset
    TEMPLATE = template_program
    DESCRIPTION = task_description
    FUNCTION_NAME = function_name
    SOLVER = staticmethod(solve)
    DEFAULT_SETTINGS = {'repair_steps': 100, 'reduction_attempts': 4}
