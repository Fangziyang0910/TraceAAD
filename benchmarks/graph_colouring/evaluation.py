"""Evaluate complete colorings against the original graph."""

import numpy as np

from core.evaluate import InvalidEvaluationResult
from .._fixed_evaluation import FixedEvaluation
from . import dataset
from .template import function_name, task_description, template_program


def validate_coloring(adjacency, colors):
    try:
        colors = np.asarray(colors)
    except (TypeError, ValueError) as exc:
        raise InvalidEvaluationResult('solve must return an integer color array') from exc
    if colors.shape != (len(adjacency),) or colors.dtype.kind not in 'iu' or np.any(colors < 0):
        raise InvalidEvaluationResult('solve must return one nonnegative integer color per vertex')
    if np.any(adjacency & (colors[:, None] == colors[None, :])):
        raise InvalidEvaluationResult('coloring assigns the same color to adjacent vertices')
    return int(len(np.unique(colors)))


def solve(data, solver):
    adjacency = data['adjacency']
    colors = solver(adjacency.copy())
    return colors, validate_coloring(adjacency, colors)


class GraphColouringEvaluation(FixedEvaluation):
    DATASET = dataset
    TEMPLATE = template_program
    DESCRIPTION = task_description
    FUNCTION_NAME = function_name
    SOLVER = staticmethod(solve)
    DEFAULT_SETTINGS = {}
    INTERFACE = 'full-solver-v1'
