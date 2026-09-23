from __future__ import annotations

import numpy as np
import pytest

from benchmarks.op_aco import (
    OPACOEvaluation,
    get_max_len,
    get_split_spec,
    load_split_instances,
)
from benchmarks.op_aco.dataset import gen_distance_matrix, gen_prizes
from benchmarks.op_aco.evaluation import ACO
from benchmarks.op_aco.template import design_notes


def prize_over_distance(prize, distance, maxlen):
    return prize[np.newaxis, :] / distance


def test_train_split_matches_published_protocol_and_is_reproducible():
    first, metadata = load_split_instances("train")
    second, _ = load_split_instances("train")

    assert metadata["problem_size"] == 50
    assert metadata["n_instances"] == 5
    assert metadata["seed"] == 1234
    assert metadata["max_len"] == 3.0
    assert first.shape == (5, 50, 2)
    assert np.array_equal(first, second)


def test_task_exposes_evaluator_semantics_as_design_notes():
    notes = ' '.join(design_notes.split())
    assert 'Node 0 is masked as a candidate' in notes
    assert 'Return-to-depot distance still affects feasibility' in notes
    assert 'static prior computed before ACO' in notes
    assert OPACOEvaluation().design_notes == design_notes


@pytest.mark.parametrize(
    ("split", "size", "count", "seed", "maxlen"),
    [
        ("val_50", 50, 64, 3456, 3.0),
        ("val_100", 100, 64, 3456, 4.0),
        ("val_200", 200, 64, 3456, 5.0),
        ("test_50", 50, 64, 4567, 3.0),
        ("test_100", 100, 64, 4567, 4.0),
        ("test_200", 200, 64, 4567, 5.0),
    ],
)
def test_fixed_evaluation_split_protocol(split, size, count, seed, maxlen):
    instances, metadata = load_split_instances(split)
    assert instances.shape == (count, size, 2)
    assert metadata["problem_size"] == size
    assert metadata["n_instances"] == count
    assert metadata["seed"] == seed
    assert metadata["max_len"] == maxlen
    assert get_split_spec(split).seed == seed
    assert get_max_len(size) == maxlen


def test_train_and_test_use_independent_seeds():
    train, _ = load_split_instances("train")
    test, _ = load_split_instances("test_50")
    assert not np.array_equal(train[0], test[0])


def test_prize_and_distance_helpers():
    coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    prizes = gen_prizes(coordinates)
    distances = gen_distance_matrix(coordinates)
    assert prizes.shape == (3,)
    assert np.isclose(prizes.max(), 1.0)
    assert distances.shape == (3, 3)
    assert distances[0, 0] >= 1e9


def test_prize_over_distance_score_is_deterministic():
    evaluator = OPACOEvaluation(split="train", n_ants=3, n_iterations=1)
    first = evaluator.evaluate(prize_over_distance)
    second = evaluator.evaluate(prize_over_distance)
    assert first == second
    assert first is not None and first > 0


def test_aco_collects_finite_prize_and_respects_budget():
    coordinates = np.random.RandomState(0).rand(10, 2)
    prizes = gen_prizes(coordinates)
    distances = gen_distance_matrix(coordinates)
    heuristic = np.maximum(prizes[np.newaxis, :] / distances, 1e-9)
    aco = ACO(
        prizes=prizes,
        distances=distances,
        max_len=3.0,
        heuristic=heuristic,
        n_ants=4,
        rng=np.random.default_rng(7),
    )
    obj = aco.run(2)
    assert np.isfinite(obj)
    assert obj >= 0


def test_invalid_worker_counts_raise():
    with pytest.raises(ValueError):
        OPACOEvaluation(n_ants=0)
    with pytest.raises(ValueError):
        OPACOEvaluation(n_workers=0)


def test_serial_and_parallel_scores_match():
    serial = OPACOEvaluation(
        split="train", n_ants=2, n_iterations=1, n_workers=1, aco_seed=7
    )
    parallel = OPACOEvaluation(
        split="train", n_ants=2, n_iterations=1, n_workers=2, aco_seed=7
    )
    assert serial.evaluate(prize_over_distance) == parallel.evaluate(prize_over_distance)
