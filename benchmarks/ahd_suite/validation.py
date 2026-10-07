"""The common contract of a candidate-ranking heuristic."""

import numpy as np
from core.evaluate import InvalidEvaluationResult


def scores(value, length, name):
    try:
        value = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InvalidEvaluationResult(f"{name} must return numeric candidate scores") from exc
    if value.shape != (length,) or not np.isfinite(value).all():
        raise InvalidEvaluationResult(f"{name} must return {length} finite scores in candidate order")
    return value
