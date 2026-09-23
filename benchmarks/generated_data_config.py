from __future__ import annotations

from copy import deepcopy
from typing import Any

TRAIN_SEED = 2024
EVAL_SEED = 2025
MAX_CUT_TRAIN_SEED = 2026
MAX_CUT_EVAL_SEED = 2027

GENERATED_TASK_CONFIGS: dict[str, dict[str, dict[str, Any]]] = {
    "admissible_set": {
        "train": {
            "timeout_seconds": 60,
            "dimension": 15,
            "weight": 10,
        },
        "eval": {
            "timeout_seconds": 60,
            "dimension": 15,
            "weight": 10,
        },
    },
    "bp_1d_construct": {
        "train": {
            "timeout_seconds": 60,
            "n_bins": 500,
            "n_instance": 8,
            "n_items": 500,
            "bin_capacity": 100,
            "seed": TRAIN_SEED,
        },
        "eval": {
            "timeout_seconds": 60,
            "n_bins": 500,
            "n_instance": 8,
            "n_items": 500,
            "bin_capacity": 100,
            "seed": EVAL_SEED,
        },
    },
    "bp_2d_construct": {
        "train": {
            "timeout_seconds": 120,
            "n_bins": 100,
            "n_instance": 8,
            "n_items": 100,
            "bin_width": 100,
            "bin_height": 100,
            "seed": TRAIN_SEED,
        },
        "eval": {
            "timeout_seconds": 120,
            "n_bins": 100,
            "n_instance": 8,
            "n_items": 100,
            "bin_width": 100,
            "bin_height": 100,
            "seed": EVAL_SEED,
        },
    },
    "cvrp_construct": {
        "train": {
            "timeout_seconds": 20,
            "n_instance": 16,
            "problem_size": 50,
            "capacity": 40,
            "seed": TRAIN_SEED,
        },
        "eval": {
            "timeout_seconds": 20,
            "n_instance": 16,
            "problem_size": 50,
            "capacity": 40,
            "seed": EVAL_SEED,
        },
    },
    "knapsack_construct": {
        "train": {
            "timeout_seconds": 20,
            "n_instance": 32,
            "n_items": 100,
            "knapsack_capacity": 100,
            "seed": TRAIN_SEED,
        },
        "eval": {
            "timeout_seconds": 20,
            "n_instance": 32,
            "n_items": 100,
            "knapsack_capacity": 100,
            "seed": EVAL_SEED,
        },
    },
    "max_cut": {
        "train": {
            "timeout_seconds": 30,
            "n_instance": 16,
            "n_nodes": 100,
            "edge_probability": 0.5,
            "seed": MAX_CUT_TRAIN_SEED,
        },
        "eval": {
            "timeout_seconds": 30,
            "n_instance": 16,
            "n_nodes": 100,
            "edge_probability": 0.5,
            "seed": MAX_CUT_EVAL_SEED,
        },
    },
    "online_bin_packing": {
        "train": {
            "timeout_seconds": 30,
            "dataset_specs": [
                {
                    "n_instances": 1,
                    "n_items": 1000,
                    "capacities": [100, 500],
                },
                {
                    "n_instances": 1,
                    "n_items": 5000,
                    "capacities": [100, 500],
                },
            ],
            "seed": TRAIN_SEED,
        },
        "eval": {
            "timeout_seconds": 30,
            "dataset_specs": [
                {
                    "n_instances": 5,
                    "n_items": 1000,
                    "capacities": [100, 500],
                },
                {
                    "n_instances": 5,
                    "n_items": 5000,
                    "capacities": [100, 500],
                },
                {
                    "n_instances": 5,
                    "n_items": 10000,
                    "capacities": [100, 500],
                },
            ],
            "seed": EVAL_SEED,
        },
    },
    "online_bin_packing_2O": {
        "train": {
            "timeout_seconds": 60,
            "n_instances": 5,
            "n_items": 5000,
            "capacity": 100,
            "seed": TRAIN_SEED,
        },
        "eval": {
            "timeout_seconds": 60,
            "n_instances": 5,
            "n_items": 5000,
            "capacity": 100,
            "seed": EVAL_SEED,
        },
    },
    "ovrp_construct": {
        "train": {
            "timeout_seconds": 20,
            "problem_size": 50,
            "n_instance": 16,
            "seed": TRAIN_SEED,
        },
        "eval": {
            "timeout_seconds": 20,
            "problem_size": 50,
            "n_instance": 16,
            "seed": EVAL_SEED,
        },
    },
    "pymoo_moead": {
        "train": {
            "timeout_seconds": 100,
            "n_var": 10,
            "n_obj": 3,
            "n_partitions": 12,
            "pop_size": 100,
            "n_gen": 100,
            "seed": TRAIN_SEED,
        },
        "eval": {
            "timeout_seconds": 100,
            "n_var": 10,
            "n_obj": 3,
            "n_partitions": 12,
            "pop_size": 100,
            "n_gen": 100,
            "seed": EVAL_SEED,
        },
    },
    "qap_construct": {
        "train": {
            "timeout_seconds": 60,
            "n_facilities": 20,
            "n_instance": 8,
            "seed": TRAIN_SEED,
        },
        "eval": {
            "timeout_seconds": 60,
            "n_facilities": 20,
            "n_instance": 8,
            "seed": EVAL_SEED,
        },
    },
    "tsp_construct": {
        "train": {
            "timeout_seconds": 20,
            "n_instance": 16,
            "problem_size": 50,
            "seed": TRAIN_SEED,
        },
        "eval": {
            "timeout_seconds": 20,
            "n_instance": 16,
            "problem_size": 50,
            "seed": EVAL_SEED,
        },
    },
    "tsp_gls_2O": {
        "train": {
            "timeout_seconds": 120,
            "n_instance": 16,
            "problem_size": 200,
            "seed": TRAIN_SEED,
        },
        "eval": {
            "timeout_seconds": 120,
            "n_instance": 16,
            "problem_size": 200,
            "seed": EVAL_SEED,
        },
    },
    "vrptw_construct": {
        "train": {
            "timeout_seconds": 30,
            "problem_size": 50,
            "n_instance": 16,
            "seed": TRAIN_SEED,
        },
        "eval": {
            "timeout_seconds": 30,
            "problem_size": 50,
            "n_instance": 16,
            "seed": EVAL_SEED,
        },
    },
}


def get_generated_task_kwargs(task_name: str, split: str = "train") -> dict[str, Any]:
    try:
        return deepcopy(GENERATED_TASK_CONFIGS[task_name][split])
    except KeyError as exc:
        known_tasks = ", ".join(sorted(GENERATED_TASK_CONFIGS))
        raise KeyError(
            f"Unknown generated optimization task/split: "
            f"{task_name!r}/{split!r}. Known tasks: {known_tasks}"
        ) from exc
