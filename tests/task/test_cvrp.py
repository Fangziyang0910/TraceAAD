from __future__ import annotations

import numpy as np

from benchmarks.cvrp_aco import load_split_instances
from benchmarks.cvrp_aco.evaluation import ACO


def test_splits_use_independent_seeded_instances():
    train, _ = load_split_instances("train")
    test, _ = load_split_instances("test_50")
    assert not np.array_equal(train[0], test[0])


def test_aco_routes_respect_capacity_visit_each_customer_and_close_at_depot():
    distances = np.ones((4, 4), dtype=float)
    np.fill_diagonal(distances, 1.0)
    aco = ACO(
        distances=distances,
        demands=np.array([0.0, 2.0, 2.0, 2.0]),
        heuristic=np.ones((4, 4), dtype=float),
        capacity=3,
        n_ants=1,
        rng=np.random.default_rng(7),
    )

    paths = aco._generate_paths()
    route = paths[:, 0].tolist()
    assert route[0] == route[-1] == 0
    assert sorted(node for node in route if node != 0) == [1, 2, 3]
    assert all(
        sum(aco.demands[node] for node in leg) <= aco.capacity for leg in _customer_legs(route)
    )
    assert aco._path_costs(paths).tolist() == [6.0]

    original = aco.pheromone.copy()
    costs = aco._path_costs(paths)
    aco._update_pheromone(paths, costs)
    assert np.any(aco.pheromone > original * aco.decay)


def _customer_legs(route):
    leg = []
    for node in route[1:]:
        if node == 0:
            yield leg
            leg = []
        else:
            leg.append(node)
