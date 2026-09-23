from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

DEFAULT_SPLIT = "train"
TRAIN_SEED = 1234
VAL_SEED = 3456
TEST_SEED = 4567


@dataclass(frozen=True)
class OPACODataset:
    role: str
    problem_size: int
    n_instances: int
    seed: int


# Protocol matches ReEvo / HSEvo / PathWise / CALM OP-ACO:
# train: 5 × OP50 (seed 1234); val/test: 64 × {50,100,200} (seeds 3456 / 4567).
SPLIT_SPECS = {
    "train": OPACODataset("train", 50, 5, TRAIN_SEED),
    "val_50": OPACODataset("validation", 50, 64, VAL_SEED),
    "val_100": OPACODataset("validation", 100, 64, VAL_SEED),
    "val_200": OPACODataset("validation", 200, 64, VAL_SEED),
    "test_50": OPACODataset("test", 50, 64, TEST_SEED),
    "test_100": OPACODataset("test", 100, 64, TEST_SEED),
    "test_200": OPACODataset("test", 200, 64, TEST_SEED),
}


def get_max_len(n: int) -> float:
    """Travel budget used by ReEvo/DeepACO OP-ACO for standard sizes."""
    thresholds = (50, 100, 200, 300)
    budgets = (3.0, 4.0, 5.0, 6.0)
    for threshold, budget in zip(thresholds, budgets):
        if n <= threshold:
            return budget
    return 7.0


def get_split_spec(split: str = DEFAULT_SPLIT) -> OPACODataset:
    try:
        return SPLIT_SPECS[split]
    except KeyError as exc:
        known = ", ".join(SPLIT_SPECS)
        raise ValueError(f"Unknown OP-ACO split {split!r}. Known splits: {known}") from exc


def gen_prizes(coordinates: np.ndarray) -> np.ndarray:
    """Kool-style prizes normalized by distance-to-depot (node 0)."""
    depot = coordinates[0]
    distances = np.linalg.norm(coordinates - depot, axis=1)
    max_distance = float(distances.max())
    if max_distance <= 0:
        prizes = np.ones(len(coordinates), dtype=np.float64)
    else:
        prizes = 1.0 + np.floor(99.0 * distances / max_distance)
    prizes = prizes / prizes.max()
    return prizes.astype(np.float64)


def gen_distance_matrix(coordinates: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(
        coordinates[:, np.newaxis] - coordinates[np.newaxis, :], axis=2
    )
    n = len(coordinates)
    distances[np.arange(n), np.arange(n)] = 1e9
    return distances.astype(np.float64)


def generate_instances(spec: OPACODataset) -> np.ndarray:
    """Generate fixed coordinate arrays with shape (n_instances, n, 2).

    Node 0 is the depot (random location in the unit square), matching ReEvo.
    """
    rng = np.random.RandomState(spec.seed)
    return rng.rand(spec.n_instances, spec.problem_size, 2).astype(np.float64)


def load_split_instances(split: str = DEFAULT_SPLIT):
    spec = get_split_spec(split)
    coordinates = generate_instances(spec)
    metadata = {
        "dataset_id": "op_aco_reevo_v1",
        "split": split,
        **asdict(spec),
        "max_len": get_max_len(spec.problem_size),
        "protocol": (
            "ReEvo/HSEvo/PathWise/CALM OP-ACO: train seed=1234 (5×OP50); "
            "val seed=3456 / test seed=4567 (64×OP50/100/200); "
            "maxlen={3,4,5}; ACO 20 ants × 50 iterations"
        ),
    }
    return coordinates, metadata
