from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

CAPACITY = 50
DEPOT = (0.5, 0.5)
DEMAND_LOW = 1
DEMAND_HIGH = 9
DEFAULT_SPLIT = "train"


@dataclass(frozen=True)
class CVRPACODataset:
    role: str
    problem_size: int
    n_instances: int
    seed: int
    capacity: int = CAPACITY


# The 10/64-instance protocol follows MCTS-AHD and ReEvo. The 250-instance
# paper_test splits match PathWise's final reporting protocol. test_200 was
# appended to the canonical stream so train/test_50/test_100 stay byte-identical.
SPLIT_SPECS = {
    "train": CVRPACODataset("train", 50, 10, 1234),
    "val_20": CVRPACODataset("validation", 20, 64, 1234),
    "val_50": CVRPACODataset("validation", 50, 64, 1234),
    "val_100": CVRPACODataset("validation", 100, 64, 1234),
    "test_20": CVRPACODataset("test", 20, 64, 3200),
    "test_50": CVRPACODataset("test", 50, 64, 1234),
    "test_100": CVRPACODataset("test", 100, 64, 1234),
    "test_200": CVRPACODataset("test", 200, 64, 1234),
    "paper_test_50": CVRPACODataset("test", 50, 250, 4500),
    "paper_test_100": CVRPACODataset("test", 100, 250, 4100),
}

CANONICAL_SPLIT_ORDER = (
    "train",
    "val_20",
    "val_50",
    "val_100",
    "test_50",
    "test_100",
    "test_200",
)


def get_split_spec(split: str = DEFAULT_SPLIT) -> CVRPACODataset:
    try:
        return SPLIT_SPECS[split]
    except KeyError as exc:
        known = ", ".join(SPLIT_SPECS)
        raise ValueError(f"Unknown CVRP-ACO split {split!r}. Known splits: {known}") from exc


def _generate_batch(
    spec: CVRPACODataset, rng: np.random.RandomState
) -> np.ndarray:
    instances = np.empty(
        (spec.n_instances, spec.problem_size + 1, 3), dtype=np.float64
    )
    for instance in instances:
        coordinates = rng.rand(spec.problem_size, 2)
        demands = rng.randint(
            DEMAND_LOW, DEMAND_HIGH + 1, size=spec.problem_size
        )
        instance[0] = (0.0, *DEPOT)
        instance[1:, 0] = demands
        instance[1:, 1:] = coordinates
    return instances


def generate_instances(spec: CVRPACODataset) -> np.ndarray:
    """Generate fixed [demand, x, y] arrays with a centered depot."""
    split = next(
        (name for name, value in SPLIT_SPECS.items() if value == spec), None
    )
    if split in CANONICAL_SPLIT_ORDER:
        # The reference generator seeds once, then emits all seven splits in
        # this order. Replaying preceding draws makes every split byte-identical.
        rng = np.random.RandomState(1234)
        for name in CANONICAL_SPLIT_ORDER:
            instances = _generate_batch(SPLIT_SPECS[name], rng)
            if name == split:
                return instances
        raise AssertionError("unreachable canonical CVRP-ACO split")
    return _generate_batch(spec, np.random.RandomState(spec.seed))


def load_split_instances(split: str = DEFAULT_SPLIT):
    spec = get_split_spec(split)
    metadata = {
        "dataset_id": "cvrp_aco_seeded_v1",
        "split": split,
        **asdict(spec),
        "depot": list(DEPOT),
        "demand_range": [DEMAND_LOW, DEMAND_HIGH],
    }
    return generate_instances(spec), metadata
