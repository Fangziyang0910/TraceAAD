"""CALM seed heuristics adapted to LLM4AD template signatures."""

from __future__ import annotations

from importlib import import_module


_TASK_MODULES = {
    'online_bin_packing': 'online_bin_packing',
    'tsp_construct': 'tsp_construct',
    'op_aco': 'op_aco',
    'cvrp_aco': 'cvrp_aco',
    'vrptw_construct': 'vrptw_construct',
}


def load_seed(task_key: str) -> tuple[str, str]:
    """Return (seed_code, seed_idea) for a mapped LLM4AD task key."""
    if task_key not in _TASK_MODULES:
        raise KeyError(f'No CALM seed for task {task_key!r}')
    mod = import_module(f'.{_TASK_MODULES[task_key]}', package=__name__)
    return mod.SEED_CODE, mod.SEED_IDEA
