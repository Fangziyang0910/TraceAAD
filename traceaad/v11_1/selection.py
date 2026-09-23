"""Fixed V11.1 sampling rules for parents and archive references."""

import math
from bisect import bisect_left, bisect_right

OPERATORS = ("Refine", "Tune", "Pivot", "Fuse")
OPERATOR_PROBABILITIES = {operator: 0.25 for operator in OPERATORS}
EXPLORATION_WEIGHT = 0.1
PARENT_TEMPERATURE = 0.2
REFERENCE_TEMPERATURE = 8.0
DEFAULT_REFERENCE_COUNT = 8


def quality_percentiles(fitness_values):
    """Midrank percentile in [0, 1] across valid nodes; 0.5 for one node."""
    total = len(fitness_values)
    if total == 1:
        return [0.5]
    ascending = sorted(fitness_values)
    percentiles = []
    for score in fitness_values:
        low, high = bisect_left(ascending, score), bisect_right(ascending, score)
        lower, equal = low, high - low
        percentiles.append((lower + (equal - 1) / 2) / (total - 1))
    return percentiles


def score_nodes(nodes):
    """Return nodes in id order, their quality percentiles, and scores."""
    nodes = sorted(nodes, key=lambda node: node.id)
    percentiles = quality_percentiles([node.fitness for node in nodes])
    total_attempts = sum(node.attempts for node in nodes)
    scores = [percentile + EXPLORATION_WEIGHT * math.sqrt(
        math.log1p(total_attempts) / (1 + node.attempts))
        for node, percentile in zip(nodes, percentiles)]
    return nodes, percentiles, scores

def sample_parent(nodes, rng):
    """Sample every valid node using the fixed-temperature score softmax."""
    nodes, percentiles, scores = score_nodes(nodes)
    if not nodes:
        raise RuntimeError("cannot select a parent without a valid node")
    maximum = max(scores)
    weights = [math.exp((score - maximum) / PARENT_TEMPERATURE)
               for score in scores]
    total = sum(weights)
    parent = rng.choices(nodes, weights=weights)[0]
    index = nodes.index(parent)
    selection = {
        "parent_id": parent.id,
        "percentile": percentiles[index],
        "attempts": parent.attempts,
        "score": scores[index],
        "probability": weights[index] / total,
        "node_count": len(nodes),
    }
    return parent, selection

def reference_pool(nodes, parent):
    """Archive nodes with ideas, excluding only the current node."""
    return [node for node in nodes
            if node.id != parent.id and node.idea and node.idea.strip()]


def sample_references(nodes, k, rng, *, tau=REFERENCE_TEMPERATURE):
    """Draw up to ``k`` archive nodes without replacement by rank softmax."""
    ordered = sorted(nodes, key=lambda node: (-node.fitness, node.id))
    pool = [(node, math.exp(-(rank - 1) / tau))
            for rank, node in enumerate(ordered, 1)]
    chosen = []
    while len(chosen) < k and pool:
        total = sum(weight for _, weight in pool)
        pick = rng.random() * total
        cumulative = 0.0
        for index, (_, weight) in enumerate(pool):
            cumulative += weight
            if pick < cumulative:
                chosen.append(pool.pop(index)[0])
                break
    return chosen
