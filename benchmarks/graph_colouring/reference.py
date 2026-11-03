"""Fixed construction used to report dataset reference values."""

import numpy as np


def dsatur(adjacency: np.ndarray) -> np.ndarray:
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
