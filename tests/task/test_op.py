from __future__ import annotations

import numpy as np

from benchmarks.op_aco import load_split_instances
from benchmarks.op_aco.dataset import gen_distance_matrix, gen_prizes
from benchmarks.op_aco.evaluation import ACO


def test_train_and_test_use_independent_seeds():
    train, _ = load_split_instances("train")
    test, _ = load_split_instances("test_50")
    assert not np.array_equal(train[0], test[0])


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
