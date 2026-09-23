import numpy as np


def _generate_sizes(rng, num_items: int, max_item_size: int) -> np.ndarray:
    samples = rng.weibull(3, num_items) * 45
    samples = np.clip(samples, 1, max_item_size)
    return np.round(samples).astype(int)


def generate_weibull_dataset(
    num_instances,
    num_items,
    capacity_limit,
    seed=2024,
    max_item_size: int = 100,
):
    """Generate Weibull OBP instances.

    Item sizes are clipped at ``max_item_size`` (paper default 100), independent
    of bin ``capacity_limit``. This matches EoH / ReEvo / MCTS-AHD / PathWise,
    where capacity-500 tests still use items clipped at 100.
    """
    rng = np.random.RandomState(seed)

    dataset = {}

    for i in range(num_instances):
        sizes = _generate_sizes(rng, num_items, max_item_size)
        instance = {
            'capacity': capacity_limit,
            'num_items': num_items,
            'items': sizes,
        }
        dataset[f'instance_{i}'] = instance

    return dataset


def generate_weibull_multiscale_dataset(
    dataset_specs,
    seed=2024,
    max_item_size: int = 100,
):
    """Generate a fixed OBP dataset spanning multiple sizes and capacities.

    For each generated item sequence, every configured capacity receives the
    same items. This follows the MCTS-AHD protocol where capacity changes the
    packing problem without changing the underlying item sequence.
    """
    dataset = {}

    for spec in dataset_specs:
        num_instances = int(spec["n_instances"])
        num_items = int(spec["n_items"])
        capacities = tuple(int(value) for value in spec["capacities"])
        size_label = f"{num_items // 1000}k" if num_items % 1000 == 0 else str(num_items)
        scale_seed = np.random.SeedSequence([int(seed), num_items]).generate_state(1)[0]
        rng = np.random.RandomState(scale_seed)

        for instance_index in range(num_instances):
            sizes = _generate_sizes(rng, num_items, max_item_size)
            for capacity in capacities:
                key = f"{size_label}_{capacity}_instance_{instance_index}"
                if key in dataset:
                    raise ValueError(f"duplicate OBP dataset configuration: {key}")
                dataset[key] = {
                    "capacity": capacity,
                    "num_items": num_items,
                    "items": sizes.copy(),
                }

    return dataset
