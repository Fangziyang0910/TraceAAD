"""Simple evolutionary opportunity and archive selection for V10.13.

The archive is the search population. Quality controls the amount of parent
attention through an ESS-calibrated distribution; the operators provide the
creative moves. Parent attempts are recorded for analysis only and never become
a deterministic novelty or lineage score.
"""

from __future__ import annotations

import ast
import math
from functools import lru_cache

OPERATORS = ("Refine", "Tune", "Pivot", "Fuse")
OPERATOR_PROBABILITIES = {operator: 0.25 for operator in OPERATORS}
QUALITY_ESS_TARGET = 8.0
PARENT_UNIFORM_PROBABILITY = 0.125
REFERENCE_COUNT = 3


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
    """Return an ESS-calibrated quality distribution over valid nodes."""
    if not nodes:
        raise RuntimeError("cannot allocate an empty archive")
    scores = [node.fitness for node in nodes]
    beta, target, quality_ess = calibrate_beta(scores, QUALITY_ESS_TARGET)
    probabilities = softmax(scores, beta)
    return probabilities, {
        "beta": beta,
        "quality_ess": ess(probabilities),
        "ess_target": target,
        "population_size": len(nodes),
    }


def sample_parent(nodes, operator, rng):
    """Keep the previous aggregate exploration mass, independent of operator."""
    probabilities, stats = quality_distribution(nodes)
    probabilities = mix_uniform(probabilities, PARENT_UNIFORM_PROBABILITY)
    index = rng.choices(range(len(nodes)), weights=probabilities)[0]
    parent = nodes[index]
    selection = {
        "parent_probability": probabilities[index],
        "parent_ess": ess(probabilities),
        "parent_ess_target": stats["ess_target"],
        "parent_count_before": parent.attempts,
        "population_size": len(nodes),
        "uniform_mass": PARENT_UNIFORM_PROBABILITY,
    }
    return parent, selection


@lru_cache(maxsize=8192)
def code_key(code):
    return ast.dump(ast.parse(code), include_attributes=False)


def reference_shortlist(nodes, parent, rng, count=REFERENCE_COUNT):
    """Expose distinct implementations, not purported semantic categories.

    One quality draw, one uniform draw, and one inverse-exposure draw supply
    different opportunities. The generating model judges complementarity and
    can request one full program or ignore every card. Exposure is not quality.
    """
    parent_key = code_key(parent.code)
    by_code, exposures = {}, {}
    for node in sorted(nodes, key=lambda item: item.id):
        key = code_key(node.code)
        if key != parent_key:
            by_code[key] = node  # latest measured representative, not best replicate
            exposures[key] = exposures.get(key, 0) + node.reference_uses
    pool = list(by_code.values())
    selected, sources = [], {}
    for source in ("quality", "uniform", "underexposed")[:count]:
        if not pool:
            break
        if source == "quality":
            weights, _ = quality_distribution(pool)
        elif source == "uniform":
            weights = [1.0] * len(pool)
        else:
            weights = [1.0 / (1 + exposures[code_key(node.code)]) for node in pool]
        index = rng.choices(range(len(pool)), weights=weights)[0]
        node = pool.pop(index)
        selected.append(node)
        sources[str(node.id)] = source
    rng.shuffle(selected)  # Do not present rank order as a preferred answer.
    return selected, {"reference_sources": sources, "distinct_reference_pool": len(by_code)}
