"""Fixed V10.12 sampling rules: ESS-calibrated quality softmax."""

import ast
import math
from functools import lru_cache

OPERATORS = ("Refine", "Tune", "Pivot", "Fuse")
OPERATOR_PROBABILITIES = {operator: 0.25 for operator in OPERATORS}
QUALITY_ESS_TARGET = 8.0
PIVOT_UNIFORM_PROBABILITY = 0.5
DONOR_UNIFORM_PROBABILITY = 0.5


@lru_cache(maxsize=8192)
def code_key(code):
    return ast.dump(ast.parse(code), include_attributes=False)


def mix_uniform(probabilities, mass):
    n = len(probabilities)
    return [(1.0 - mass) * value + mass / n for value in probabilities]


def ess(probabilities):
    return 1.0 / sum(p * p for p in probabilities)


def softmax(scores, beta):
    maximum = max(scores)
    weights = [math.exp(beta * (score - maximum)) for score in scores]
    total = sum(weights)
    return [weight / total for weight in weights]


def _ess(beta, scores):
    return ess(softmax(scores, beta))


def calibrate_beta(scores, target):
    target = min(len(scores), max(float(target), sum(score == max(scores) for score in scores)))
    if len(scores) <= 1 or _ess(0.0, scores) <= target:
        return 0.0, target, _ess(0.0, scores)
    high = 1.0
    for _ in range(60):
        if _ess(high, scores) <= target:
            break
        high *= 10.0
    low = 0.0
    for _ in range(80):
        middle = (low + high) / 2
        if _ess(middle, scores) > target:
            low = middle
        else:
            high = middle
    return high, target, _ess(high, scores)
