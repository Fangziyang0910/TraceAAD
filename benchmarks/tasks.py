"""Registered tasks, explicit six-task/legacy suites, and fixed evaluation conditions."""

from copy import deepcopy
import math

from .generated_data_config import get_generated_task_kwargs
from .tsp_construct import TSPEvaluation
from .vrptw_construct import VRPTWEvaluation
from .online_bin_packing import OBPEvaluation
from .cvrp_aco import CVRPACOEvaluation
from .op_aco import OPACOEvaluation
from .fssp_gls import FSSPGLSEvaluation
from .graph_colouring import GraphColouringEvaluation
from .jssp_construct import JSSPEvaluation

FIXED_TASKS = ('fssp_gls', 'graph_colouring', 'jssp_construct')
TASKS = ('tsp_construct', 'cvrp_aco', 'op_aco', 'online_bin_packing', 'vrptw_construct')
ALL_TASKS = TASKS + FIXED_TASKS
CO_TASKS = ('tsp_construct', 'cvrp_aco', 'fssp_gls', 'graph_colouring', 'jssp_construct', 'op_aco')
SUITES = {'legacy': TASKS, 'co6': CO_TASKS}
TASK_SHORT = dict(zip(TASKS, ('tsp', 'cvrp', 'op', 'obp', 'vrptw')))
TASK_SHORT.update(dict(zip(FIXED_TASKS, ('fssp', 'gcol', 'jssp'))))
NATIVE_MINIMIZE = set(ALL_TASKS) - {'op_aco'}
MINIMIZE = set(ALL_TASKS)  # Every evaluator returns a minimized scalar objective.
CLASSES = dict(zip(ALL_TASKS, (TSPEvaluation, CVRPACOEvaluation, OPACOEvaluation, OBPEvaluation, VRPTWEvaluation,
                                    FSSPGLSEvaluation, GraphColouringEvaluation, JSSPEvaluation)))
SELECTION_SEED = 20260927
DEFAULT_WORKERS = 4
# Six-task protocol: each instance gets the same wall-clock budget, solver
# included. Instances run one after another, so a split's limit is its
# instance count times INSTANCE_SECONDS, on training and same-scale test alike.
INSTANCE_SECONDS = 10
TRAIN_INSTANCES, TEST_INSTANCES = 16, 50
TRAIN_TIMEOUT = {'online_bin_packing': 30, 'vrptw_construct': 30}
TRAIN_TIMEOUT.update({task: TRAIN_INSTANCES * INSTANCE_SECONDS for task in CO_TASKS})
HELDOUT_TIMEOUT = {'vrptw_construct': 1000, 'online_bin_packing': 1000}
HELDOUT_TIMEOUT.update({task: TEST_INSTANCES * INSTANCE_SECONDS for task in CO_TASKS})
# Cross-scale checks lie outside the six-task protocol and keep their old limits.
CROSS_SCALE_TIMEOUT = {'tsp_construct': 3000, 'cvrp_aco': 3600, 'op_aco': 3600}
SCALES = {'tsp_construct': (50, 100, 200), 'vrptw_construct': (50, 100, 200),
          'cvrp_aco': (20, 50, 100, 200), 'op_aco': (50, 100, 200),
          'online_bin_packing': ('1k_100', '1k_500', '5k_100', '5k_500', '10k_100', '10k_500')}
SCALES.update({task: (CLASSES[task].DATASET.SCALE,) for task in FIXED_TASKS})
TEST_SCALES = {task: {50} for task in TASKS}
TEST_SCALES['online_bin_packing'] = {'1k_100', '1k_500', '5k_100', '5k_500'}
TEST_SCALES.update({task: {CLASSES[task].DATASET.SCALE} for task in FIXED_TASKS})


def split_of_scale(task, scale):
    if task in FIXED_TASKS:
        return 'test_' + str(scale)
    if task == 'online_bin_packing':
        items, capacity = str(scale).split('k_')
        return f'eval_{int(items)*1000}_{capacity}'
    return ('test_' if task in {'cvrp_aco', 'op_aco'} else 'eval_') + str(scale)


def scale_of_split(task, split):
    if task in FIXED_TASKS:
        if split == 'eval':
            return CLASSES[task].DATASET.SCALE
        if split not in SPLITS[task]:
            raise ValueError(f'unknown {task} held-out split: {split}')
        return int(split.split('_')[-1])
    if split not in SPLITS[task] and not (split == 'eval' and task not in {'cvrp_aco', 'op_aco'}):
        raise ValueError(f'unknown {task} held-out split: {split}')
    if task == 'online_bin_packing':
        if split == 'eval':
            return 'all'
        _, items, capacity = split.split('_')
        return f'{int(items)//1000}k_{capacity}'
    return 50 if split == 'eval' else int(split.split('_')[-1])


SPLITS = {task: tuple(split_of_scale(task, scale) for scale in SCALES[task]) for task in ALL_TASKS}
PRIMARY_SPLITS = {task: (split_of_scale(task, next(iter(TEST_SCALES[task]))),) for task in CO_TASKS}


def obp_scale(kwargs, n_items, capacity):
    kwargs = deepcopy(kwargs)
    specs = kwargs['dataset_specs']
    matches = [spec['n_instances'] for spec in specs
               if spec['n_items'] == n_items and capacity in spec['capacities']]
    if not matches:
        raise ValueError(f'unknown OBP scale: {n_items} items, capacity {capacity}')
    count = matches[0]
    kwargs['dataset_specs'] = [{'n_instances': count, 'n_items': n_items, 'capacities': [capacity]}]
    return kwargs


def training_task(task, workers=None, *, condition='shared'):
    if task in FIXED_TASKS:
        kwargs = dict(split='train', timeout_seconds=TRAIN_TIMEOUT[task])
        return CLASSES[task](**kwargs), kwargs
    if task in {'cvrp_aco', 'op_aco'}:
        cvrp = task == 'cvrp_aco'
        # One worker: the per-instance budget assumes instances run in turn.
        kwargs = dict(split='train', timeout_seconds=TRAIN_TIMEOUT[task],
                      n_ants=30 if cvrp else 20, n_iterations=100 if cvrp else 50,
                      aco_seed=1234, n_workers=1)
    else:
        kwargs = get_generated_task_kwargs(task, 'train')
        if task in CO_TASKS or condition == 'traceaad':
            kwargs['timeout_seconds'] = TRAIN_TIMEOUT.get(task, kwargs['timeout_seconds'])
    return CLASSES[task](**kwargs), {'split': 'train', **kwargs}


def selection_task(task, search):
    if task in FIXED_TASKS:
        raise ValueError('the six-task AHD data have no validation split; choose the training-best program')
    if task in {'cvrp_aco', 'op_aco'}:
        selected = CLASSES[task](split='val_50', timeout_seconds=search.timeout_seconds,
            n_ants=search.n_ants, n_iterations=search.n_iterations,
            aco_seed=search.aco_seed, n_workers=search.n_workers)
        if search.timeout_seconds is not None:
            selected.timeout_seconds *= max(1., selected.n_instance/search.n_instance)
        return selected
    kwargs = get_generated_task_kwargs(task, 'train')
    kwargs.update(seed=SELECTION_SEED,
                  timeout_seconds=None if search.timeout_seconds is None else 2*search.timeout_seconds)
    return CLASSES[task](**kwargs)


def heldout_task(task, split, workers=DEFAULT_WORKERS, timeout_seconds=None):
    primary = task in CO_TASKS and (split == 'eval' or split in PRIMARY_SPLITS[task])
    if timeout_seconds is not None:
        timeout = timeout_seconds
    else:
        timeout = HELDOUT_TIMEOUT[task] if primary else CROSS_SCALE_TIMEOUT.get(task, HELDOUT_TIMEOUT[task])
    if workers < 1 or timeout <= 0 or not math.isfinite(timeout):
        raise ValueError('workers and timeout must be positive')
    if task in FIXED_TASKS:
        if split != 'eval' and split not in SPLITS[task]:
            raise ValueError(f'unknown {task} held-out split: {split}')
        actual_split = 'test' if split == 'eval' else split
        return CLASSES[task](split=actual_split, timeout_seconds=timeout)
    if split not in SPLITS[task] and not (split == 'eval' and task not in {'cvrp_aco', 'op_aco'}):
        raise ValueError(f'unknown {task} held-out split: {split}')
    if task in {'cvrp_aco', 'op_aco'}:
        cvrp = task == 'cvrp_aco'
        return CLASSES[task](split=split, timeout_seconds=timeout, n_workers=1 if primary else workers,
            n_ants=30 if cvrp else 20, n_iterations=100 if cvrp else 50, aco_seed=1234)
    kwargs = get_generated_task_kwargs(task, 'eval')
    if task == 'online_bin_packing' and split != 'eval':
        _, items, capacity = split.split('_')
        kwargs = obp_scale(kwargs, int(items), int(capacity))
    elif task != 'online_bin_packing':
        kwargs['problem_size'] = scale_of_split(task, split)
    if primary:
        kwargs['n_instance'] = TEST_INSTANCES
    kwargs['timeout_seconds'] = timeout
    return CLASSES[task](**kwargs)


def evaluation_limits(tasks=TASKS, *, final_selection='training'):
    output = {}
    for task in tasks:
        train, _ = training_task(task, condition='traceaad')
        selected = selection_task(task, train) if final_selection == 'validation' else None
        output[task] = {'search': train.timeout_seconds,
                        'selection': selected.timeout_seconds if selected else None,
                        'heldout': HELDOUT_TIMEOUT[task]}
    return output
