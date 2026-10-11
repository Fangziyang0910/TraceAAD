"""Fixed generated-data settings for training and evaluation."""

DATA_CONFIGS = {
    'train': {
        'timeout_seconds': 30,
        'dataset_specs': [
            {'n_instances': 1, 'n_items': 1000, 'capacities': [100, 500]},
            {'n_instances': 1, 'n_items': 5000, 'capacities': [100, 500]},
        ],
        'seed': 2024,
    },
    'eval': {
        'timeout_seconds': 30,
        'dataset_specs': [
            {'n_instances': 5, 'n_items': 1000, 'capacities': [100, 500]},
            {'n_instances': 5, 'n_items': 5000, 'capacities': [100, 500]},
            {'n_instances': 5, 'n_items': 10000, 'capacities': [100, 500]},
        ],
        'seed': 2025,
    },
}
