"""Fixed generated-data settings for training and evaluation."""

DATA_CONFIGS = {
    'train': {'timeout_seconds': 30, 'n_instance': 16, 'problem_size': 50, 'seed': 2024},
    'eval': {'timeout_seconds': 30, 'n_instance': 16, 'problem_size': 50, 'seed': 2025},
}
