"""Regression tests for the five main-experiment tasks."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from core import InvalidEvaluationResult


TASKS = (
    "tsp_construct",
    "cvrp_aco",
    "op_aco",
    "online_bin_packing",
    "vrptw_construct",
)


def test_no_template_body_uses_undefined_kwargs():
    root = Path("benchmarks")
    offenders = []
    for relative in TASKS:
        template_path = root / relative / "template.py"
        tree = ast.parse(template_path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "template_program":
                    value = node.value
                    if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
                        prog = ast.literal_eval(value.func.value)
                    else:
                        prog = ast.literal_eval(value)
                    ptree = ast.parse(prog)
                    funcs = [item for item in ptree.body if isinstance(item, ast.FunctionDef)]
                    assert len(funcs) == 1
                    func = funcs[0]
                    if func.args.kwarg is not None:
                        continue
                    body = ast.unparse(func)
                    if "kwargs[" in body or "kwargs.get" in body:
                        offenders.append(str(template_path))
    assert offenders == []


def test_tsp_description_does_not_claim_coordinate_inputs():
    from benchmarks.tsp_construct import template as tsp_t

    assert "does not receive coordinates" in tsp_t.task_description.lower()
    assert "distance matrix" in tsp_t.task_description.lower()


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
