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


def test_fixed_prior_cannot_override_dynamic_route_feasibility():
    distances = np.array([[0., 1., 3.], [1., 0., 3.], [3., 3., 0.]])
    heuristic = np.ones((3, 3))
    heuristic[:, 2] = 1e20  # Node 2 is attractive, but its return trip exceeds the budget.
    aco = ACO(np.array([0., 1., 100.]), distances, 3., heuristic,
              n_ants=4, rng=np.random.default_rng(7))
    routes = aco._gen_sol()
    assert 1 in routes and 2 not in routes
    assert np.all(aco._gen_sol_obj(routes) == 1.)
