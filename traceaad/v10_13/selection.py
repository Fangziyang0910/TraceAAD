"""Simple evolutionary opportunity and archive selection for V10.13.

The archive is the search population. Quality controls the amount of parent
attention through an ESS-calibrated distribution; the operators provide the
creative moves.
"""

from __future__ import annotations

import math

OPERATORS = ("Refine", "Tune", "Pivot", "Fuse")
OPERATOR_PROBABILITIES = {operator: 0.25 for operator in OPERATORS}
QUALITY_ESS_TARGET = 8.0
PARENT_UNIFORM_PROBABILITY = 0.125


def ess(probabilities):
    return 1.0 / sum(p * p for p in probabilities) if probabilities else 0.0


def softmax(scores, beta):
    if not scores:
        return []
    maximum = max(scores)
    weights = [math.exp(beta * (score - maximum)) for score in scores]
    total = sum(weights)
    return [weight / total for weight in weights]


def mix_uniform(probabilities, mass):
    if not probabilities:
        return []
    return [(1.0 - mass) * value + mass / len(probabilities)
            for value in probabilities]


def calibrate_beta(scores, target):
    """Find the inverse temperature whose softmax ESS is near ``target``."""
    if len(scores) <= 1:
        return 0.0, len(scores), float(len(scores))
    target = min(len(scores), max(float(target), sum(score == max(scores) for score in scores)))
    if target >= len(scores) or max(scores) == min(scores):
        return 0.0, target, float(len(scores))
    high = 1.0
    for _ in range(60):
        if ess(softmax(scores, high)) <= target:
            break
        high *= 10.0
    low = 0.0
    for _ in range(80):
        middle = (low + high) / 2.0
        if ess(softmax(scores, middle)) > target:
            low = middle
        else:
            high = middle
    return high, target, ess(softmax(scores, high))


def quality_distribution(nodes):
    """Return an ESS-calibrated quality distribution over evaluated nodes."""
    if not nodes:
        raise RuntimeError("cannot allocate an empty archive")
    scores = [node.fitness for node in nodes]
    beta, _, _ = calibrate_beta(scores, QUALITY_ESS_TARGET)
    return softmax(scores, beta)


def sample_parent(nodes, rng):
    """Sample from quality with a small uniform exploration mass."""
    probabilities = quality_distribution(nodes)
    probabilities = mix_uniform(probabilities, PARENT_UNIFORM_PROBABILITY)
    return rng.choices(nodes, weights=probabilities)[0]


def sample_reference(nodes, parent, rng):
    """Sample one different implementation for Fuse."""
    pool = [node for node in nodes if node.code != parent.code]
    if not pool:
        return None
    if rng.random() < 0.5:
        return rng.choices(pool, weights=quality_distribution(pool))[0]
    return rng.choice(pool)
