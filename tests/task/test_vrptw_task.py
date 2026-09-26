"""Regression tests for VRPTW construction behavior."""
from __future__ import annotations

import numpy as np
import pytest

from core import InvalidEvaluationResult


def test_vrptw_problem_size_counts_customers_excluding_depot():
    from benchmarks.vrptw_construct.evaluation import VRPTWEvaluation

    evaluation = VRPTWEvaluation(problem_size=3, n_instance=1)
    coordinates, distances, demands, _, service, windows = evaluation._datasets[0]
    assert coordinates.shape == (4, 2)
    assert distances.shape == (4, 4)
    assert demands.shape == service.shape == (4,)
    assert windows.shape == (4, 2)


def test_vrptw_only_offers_feasible_customers_to_heuristic():
    from benchmarks.vrptw_construct.evaluation import VRPTWEvaluation

    evaluation = VRPTWEvaluation.__new__(VRPTWEvaluation)
    evaluation.problem_size = 2
    evaluation.n_instance = 1
    distances = np.array(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    demands = np.array([0.0, 1.0, 10.0])
    service = np.zeros(3)
    windows = np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])
    evaluation._datasets = [
        (np.zeros((3, 2)), distances, demands, 5.0, service, windows)
    ]
    offered = []

    def choose_first(
        current_node,
        depot,
        feasible_nodes,
        rest_capacity,
        current_time,
        demands_arg,
        distances_arg,
        windows_arg,
    ):
        offered.append(feasible_nodes.copy())
        return int(feasible_nodes[0])

    with pytest.raises(InvalidEvaluationResult, match="no feasible customer"):
        evaluation.evaluate_program("", choose_first)
    assert evaluation.evaluate(choose_first) is None
    assert offered
    assert all(2 not in nodes for nodes in offered)


def test_vrptw_cost_includes_final_return_to_depot():
    from benchmarks.vrptw_construct.evaluation import VRPTWEvaluation

    evaluation = VRPTWEvaluation.__new__(VRPTWEvaluation)
    evaluation.problem_size = 1
    evaluation.n_instance = 1
    distances = np.array([[0.0, 2.0], [2.0, 0.0]])
    evaluation._datasets = [
        (
            np.zeros((2, 2)),
            distances,
            np.array([0.0, 1.0]),
            5.0,
            np.zeros(2),
            np.array([[0.0, 10.0], [0.0, 10.0]]),
        )
    ]

    def choose_only_customer(*args):
        return int(args[2][0])

    assert evaluation.evaluate(choose_only_customer) == -4.0
