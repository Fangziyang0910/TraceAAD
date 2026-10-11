"""Whole schedules are independently checked against the original instance."""
import numpy as np
import pytest
from benchmarks.jssp_construct.evaluation import solve, validate_schedule
from benchmarks.jssp_construct import JSSPEvaluation
from benchmarks._seeded_data import generate_dataset, digest_arrays
from benchmarks.jssp_construct import dataset
from core.evaluate import InvalidEvaluationResult


def tiny():
    return {'processing_times': np.array([[3, 2], [2, 4]]),
            'machine_order': np.array([[0, 1], [1, 0]])}


def test_complete_solver_can_choose_any_valid_schedule():
    data = tiny()
    starts, cost = solve(data, lambda t, m: np.array([[6, 9], [0, 2]]))
    assert np.array_equal(starts, [[6, 9], [0, 2]]) and cost == 11
    namespace = {}
    exec(JSSPEvaluation.TEMPLATE, namespace)
    better, best_cost = solve(data, namespace['solve'])
    assert np.array_equal(better, [[0, 3], [0, 3]])
    assert best_cost == 7  # machine 0 requires exactly seven time units
    # Deliberate idle time is legal; the evaluator must not run GT over the output.
    assert solve(data, lambda t, m: better + 10)[1] == 17


@pytest.mark.parametrize('output', [None, np.array([np.nan]), np.array([[1.]]), np.ones(99),
    [[0, 3], [0]], np.full((2, 2), np.inf), np.full((2, 2), -1),
    np.full((2, 2), '0'), np.zeros((2, 2), dtype=bool), np.full((2, 2), 1e308)])
def test_solver_rejects_invalid_start_times(output):
    with pytest.raises(InvalidEvaluationResult):
        solve(tiny(), lambda *args: output)


def test_independent_validator_rejects_precedence_and_machine_overlap():
    d = tiny()
    with pytest.raises(InvalidEvaluationResult, match='precedence'):
        validate_schedule(d['processing_times'], d['machine_order'], np.array([[0, 1], [0, 3]]))
    with pytest.raises(InvalidEvaluationResult, match='overlaps'):
        validate_schedule(d['processing_times'], d['machine_order'], np.array([[0, 3], [0, 2]]))


def test_input_mutation_cannot_change_feasibility_or_objective():
    d = tiny()
    before = digest_arrays(d)
    def mutate(t, m):
        t[:] = 0
        m[:] = 0
        return np.array([[0, 3], [0, 3]])
    assert solve(d, mutate)[1] == 7
    assert digest_arrays(d) == before
    def cheat(t, m):
        t[:] = 0
        return np.zeros_like(t)
    with pytest.raises(InvalidEvaluationResult):
        solve(d, cheat)


def test_seeded_instance_reconstruction_and_split_independence():
    a, train = generate_dataset(dataset, 'train', limit=2)
    b, repeated = generate_dataset(dataset, 'train', limit=2)
    _, test = generate_dataset(dataset, 'test', limit=2)
    assert a == b
    assert [digest_arrays(x) for x in train] == [digest_arrays(x) for x in repeated]
    assert not ({digest_arrays(x) for x in train} & {digest_arrays(x) for x in test})
    assert all(np.array_equal(np.sort(x['machine_order'], axis=1), np.tile(np.arange(20), (20, 1))) for x in train)


def test_real_evaluation_recomputes_objective_and_has_zero_template_deviation():
    evaluation = JSSPEvaluation(limit=2, safe_evaluate=False)
    assert evaluation.evaluate_program(evaluation.TEMPLATE, None) == 0
    assert evaluation.n_instance == 2 and evaluation.outer_settings == {}
