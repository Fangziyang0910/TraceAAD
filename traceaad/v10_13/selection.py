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
PIVOT_UNIFORM_PROBABILITY = 0.50
DONOR_UNIFORM_PROBABILITY = 0.50


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
    """Select a parent with V10.10 quality allocation."""
    probabilities, stats = quality_distribution(nodes)
    if operator == "Pivot":
        probabilities = mix_uniform(probabilities, PIVOT_UNIFORM_PROBABILITY)
    index = rng.choices(range(len(nodes)), weights=probabilities)[0]
    parent = nodes[index]
    selection = {
        "parent_probability": probabilities[index],
        "parent_ess": ess(probabilities),
        "parent_ess_target": stats["ess_target"],
        "parent_count_before": parent.attempts,
        "population_size": len(nodes),
    }
    return parent, selection


@lru_cache(maxsize=8192)
def code_key(code):
    return ast.dump(ast.parse(code), include_attributes=False)


def select_donor(nodes, parent, rng):
    """Select one executable archive reference for Pivot or Fuse.

    Exact code copies are excluded. The 50% uniform component keeps the LLM's
    reference material broad without pretending that archive position is a
    semantic category.
    """
    parent_key = code_key(parent.code)
    candidates = [node for node in nodes
                  if node.id != parent.id and code_key(node.code) != parent_key]
    if not candidates:
        return None, {"donor_pool_size": 0}
    probabilities, stats = quality_distribution(candidates)
    probabilities = mix_uniform(probabilities, DONOR_UNIFORM_PROBABILITY)
    index = rng.choices(range(len(candidates)), weights=probabilities)[0]
    donor = candidates[index]
    return donor, {
        "donor_probability": probabilities[index],
        "donor_ess": ess(probabilities),
        "donor_ess_target": stats["ess_target"],
        "donor_pool_size": len(candidates),
    }
