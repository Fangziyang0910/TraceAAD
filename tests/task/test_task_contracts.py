"""Each published task template must be executable by its own evaluator."""

import numpy as np
import pytest

from benchmarks.cvrp_aco import CVRPACOEvaluation
from benchmarks.online_bin_packing import OBPEvaluation
from benchmarks.op_aco import OPACOEvaluation
from benchmarks.tsp_construct.evaluation import TSPEvaluation
from benchmarks.vrptw_construct.evaluation import VRPTWEvaluation
from core import SecureEvaluator


@pytest.mark.parametrize(
    ("evaluation_type", "kwargs", "minimize"),
    [
        (TSPEvaluation, {"n_instance": 1, "problem_size": 5}, True),
        (CVRPACOEvaluation, {"n_ants": 2, "n_iterations": 1}, True),
        (OPACOEvaluation, {"n_ants": 2, "n_iterations": 1}, False),
        (OBPEvaluation, {"n_instances": 1, "n_items": 20}, True),
        (VRPTWEvaluation, {"n_instance": 1, "problem_size": 3}, True),
    ],
    ids=["tsp", "cvrp", "op", "bin-packing", "vrptw"],
)
def test_template_produces_a_valid_score(evaluation_type, kwargs, minimize):
    evaluator = evaluation_type(**kwargs)
    evaluator.safe_evaluate = False
    if isinstance(evaluator, (CVRPACOEvaluation, OPACOEvaluation)):
        evaluator._datasets = evaluator._datasets[:1]

    assert evaluator.task_description.strip()
    outcome = SecureEvaluator(evaluator).evaluate_program_with_details(
        evaluator.template_program
    )

    assert outcome.failure_kind is None, outcome.error
    assert np.isfinite(outcome.result)
    if minimize:
        assert outcome.result < 0
    else:
        assert outcome.result >= 0
