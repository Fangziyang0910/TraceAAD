from __future__ import annotations

import numpy as np
import pytest

from benchmarks.cvrp_aco import (
    CVRPACOEvaluation,
    get_split_spec,
    load_split_instances,
)
from benchmarks.cvrp_aco.evaluation import ACO


def inverse_distance(distance_matrix, coordinates, demands, capacity):
    return 1.0 / distance_matrix


def test_train_split_matches_published_protocol_and_is_reproducible():
    first, metadata = load_split_instances("train")
    second, _ = load_split_instances("train")

    assert metadata["problem_size"] == 50
    assert metadata["n_instances"] == 10
    assert metadata["capacity"] == 50
    assert metadata["seed"] == 1234
    assert first.shape == (10, 51, 3)
    assert np.array_equal(first, second)
    assert np.all(first[:, 0, 0] == 0)
    assert np.all(first[:, 0, 1:] == [0.5, 0.5])
    assert np.all((first[:, 1:, 0] >= 1) & (first[:, 1:, 0] <= 9))
    assert np.allclose(first[0, 1], [6.0, 0.1915194503788923, 0.6221087710398319])


@pytest.mark.parametrize(
    ("split", "size", "count"),
    [
        ("val_20", 20, 64),
        ("val_50", 50, 64),
        ("val_100", 100, 64),
        ("test_20", 20, 64),
        ("test_50", 50, 64),
        ("test_100", 100, 64),
        ("test_200", 200, 64),
        ("paper_test_50", 50, 250),
        ("paper_test_100", 100, 250),
    ],
)
def test_fixed_evaluation_split_protocol(split, size, count):
    instances, metadata = load_split_instances(split)
    assert instances.shape == (count, size + 1, 3)
    assert metadata["problem_size"] == size
    assert metadata["n_instances"] == count


def test_splits_use_independent_seeded_instances():
    train, _ = load_split_instances("train")
    test, _ = load_split_instances("test_50")
    assert not np.array_equal(train[0], test[0])


def test_inverse_distance_score_is_deterministic():
    evaluator = CVRPACOEvaluation(split="train", n_ants=3, n_iterations=1)
    evaluator._datasets = evaluator._datasets[:2]
    first = evaluator.evaluate(inverse_distance)
    second = evaluator.evaluate(inverse_distance)
    assert first == second
    assert first is not None and first < 0


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
        sum(aco.demands[node] for node in leg) <= aco.capacity
        for leg in _customer_legs(route)
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


@pytest.mark.parametrize(
    "bad_prior",
    [
        lambda d, c, q, cap: np.ones(len(d)),
        lambda d, c, q, cap: np.full_like(d, np.nan),
    ],
)
def test_invalid_heuristic_matrix_is_rejected(bad_prior):
    evaluator = CVRPACOEvaluation(split="train", n_ants=1, n_iterations=1)
    evaluator._datasets = evaluator._datasets[:1]
    assert evaluator.evaluate(bad_prior) is None


def test_unknown_split_and_invalid_aco_settings_are_rejected():
    with pytest.raises(ValueError, match="Unknown CVRP-ACO split"):
        get_split_spec("missing")
    for kwargs in ({"n_ants": 0}, {"n_ants": 0.5}, {"n_iterations": 0.5}):
        with pytest.raises(ValueError, match="positive integers"):
            CVRPACOEvaluation(**kwargs)
    for kwargs in ({"n_workers": 0}, {"n_workers": 0.5}, {"n_workers": True}):
        with pytest.raises(ValueError, match="n_workers"):
            CVRPACOEvaluation(**kwargs)


def test_parallel_workers_match_serial_scores():
    """Instance-parallel ACO must be numerically identical to serial evaluate."""
    serial = CVRPACOEvaluation(
        split="train", n_ants=3, n_iterations=2, n_workers=1
    )
    serial._datasets = serial._datasets[:4]
    parallel = CVRPACOEvaluation(
        split="train", n_ants=3, n_iterations=2, n_workers=4
    )
    parallel._datasets = parallel._datasets[:4]

    serial_score = serial.evaluate(inverse_distance)
    parallel_score = parallel.evaluate(inverse_distance)
    assert serial_score is not None and parallel_score is not None
    assert serial_score == parallel_score
    assert serial_score < 0
