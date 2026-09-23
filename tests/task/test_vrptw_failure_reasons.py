"""VRPTW invalid constructions report the violated condition, not a bare None."""
import numpy as np

from core import SecureEvaluator
from benchmarks.vrptw_construct import VRPTWEvaluation


def program(return_value):
    return ('import numpy as np\n\n\n'
            'def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray,'
            ' rest_capacity: float, current_time: float, demands: np.ndarray,'
            ' distance_matrix: np.ndarray, time_windows: np.ndarray) -> int:\n'
            f'    return {return_value}\n')


def evaluation(capacity):
    task = VRPTWEvaluation(timeout_seconds=30, problem_size=2, n_instance=1, seed=7)
    size = task.problem_size + 1
    distances = np.ones((size, size))
    np.fill_diagonal(distances, 0.0)
    task._datasets = [(None, distances, np.array([0, 5, 5]), capacity,
                       np.zeros(size), np.array([[0, 100.0]] * size))]
    return task


def outcome(capacity, return_value):
    evaluator = SecureEvaluator(evaluation(capacity))
    return evaluator.evaluate_program_with_details(program(return_value))


def test_depot_with_no_feasible_customer_names_the_condition():
    result = outcome(capacity=1, return_value=1)
    assert result.failure_kind == 'invalid_result'
    assert 'no feasible customer' in result.error and 'depot' in result.error


def test_returning_the_depot_while_at_the_depot_names_the_condition():
    result = outcome(capacity=10, return_value=0)
    assert result.failure_kind == 'invalid_result'
    assert 'returned the depot' in result.error


def test_node_outside_the_feasible_set_names_the_condition():
    result = outcome(capacity=10, return_value=99)
    assert result.failure_kind == 'invalid_result'
    assert 'outside the feasible candidate set' in result.error


def test_a_completing_heuristic_keeps_its_score():
    result = outcome(capacity=10, return_value='int(unvisited_nodes[0])')
    assert result.result == -3.0
