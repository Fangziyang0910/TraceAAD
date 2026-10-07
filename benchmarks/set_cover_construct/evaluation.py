"""Evolved greedy completion, redundant-set deletion, and remove-and-repair improvement."""

import numpy as np

from .._fixed_evaluation import FixedEvaluation
from . import dataset
from .template import function_name, task_description, template_program

from .._fixed_evaluation import scores


def complete(coverage, costs, selected, heuristic):
    selected = selected.copy()
    while True:
        uncovered = ~coverage[:, selected].any(axis=1)
        if not uncovered.any():
            return selected
        gain = coverage[uncovered].sum(axis=0)
        eligible = (~selected) & (gain > 0)
        if not eligible.any():
            raise ValueError("set-cover instance cannot be covered")
        value = heuristic(costs.copy(), coverage.copy(), selected.copy(), uncovered.copy())
        ranking = scores(value, len(costs), "score_sets")
        choices = np.flatnonzero(eligible)
        selected[choices[int(np.argmax(ranking[choices]))]] = True


def delete_redundant(coverage, costs, selected):
    selected = selected.copy()
    counts = coverage[:, selected].sum(axis=1)
    order = np.flatnonzero(selected)
    order = order[np.argsort(-costs[order], kind="stable")]
    for column in order:
        rows = coverage[:, column]
        if np.all(counts[rows] > 1):
            selected[column] = False
            counts[rows] -= 1
    return selected


def solve(data, heuristic, improvement_steps=16):
    coverage, costs = data["coverage"], data["costs"]
    selected = complete(coverage, costs, np.zeros(len(costs), dtype=bool), heuristic)
    best = delete_redundant(coverage, costs, selected)
    best_cost = float(costs[best].sum())
    choices = np.flatnonzero(best)
    choices = choices[np.argsort(-costs[choices], kind="stable")][:improvement_steps]
    for column in choices:
        if not best[column]:
            continue
        start = best.copy()
        start[column] = False
        trial = delete_redundant(coverage, costs, complete(coverage, costs, start, heuristic))
        cost = float(costs[trial].sum())
        if cost < best_cost:
            best, best_cost = trial, cost
    if not np.all(coverage[:, best].any(axis=1)):
        raise ValueError("outer set-cover search returned an incomplete cover")
    return best, float(costs[best].sum())


class SetCoverEvaluation(FixedEvaluation):
    DATASET = dataset
    TEMPLATE = template_program
    DESCRIPTION = task_description
    FUNCTION_NAME = function_name
    SOLVER = staticmethod(solve)
    DEFAULT_SETTINGS = {'improvement_steps': 16}
