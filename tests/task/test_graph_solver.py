"""Full graph solvers may use any labels, but must color the original graph."""
import numpy as np
import pytest
from benchmarks.graph_colouring.evaluation import solve, GraphColouringEvaluation
from core.evaluate import InvalidEvaluationResult


def path_graph():
    return {'adjacency': np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)}


def test_arbitrary_color_labels_are_counted_by_distinct_values():
    assert solve(path_graph(), lambda a: [10, 90, 10])[1] == 2
    assert solve(path_graph(), lambda a: [10, 90, 30])[1] == 3


@pytest.mark.parametrize('output', [None, [0, 0, 1], [0, 1], [-1, 0, 1],
    [0., 1., 0.], [0, np.nan, 1], [[0], [1], [0]], [[0, 1], [0]],
    [False, True, False], ['0', '1', '0']])
def test_invalid_colorings_are_rejected(output):
    with pytest.raises(InvalidEvaluationResult):
        solve(path_graph(), lambda a: output)


def test_modifying_input_cannot_remove_edges_from_validation():
    data = path_graph()
    before = data['adjacency'].copy()
    def cheat(adjacency):
        adjacency[:] = False
        return np.zeros(3, dtype=int)
    with pytest.raises(InvalidEvaluationResult, match='adjacent'):
        solve(data, cheat)
    assert np.array_equal(data['adjacency'], before)


def test_dsatur_starter_matches_reference_without_fixed_repair():
    evaluation = GraphColouringEvaluation(limit=1, safe_evaluate=False)
    # The starter is the reference construction, so its raw objective is the reference mean.
    assert evaluation.evaluate_program(evaluation.TEMPLATE, None) == pytest.approx(
        np.mean([row['reference'] for row in evaluation._rows]))
    assert evaluation.outer_settings == {}
