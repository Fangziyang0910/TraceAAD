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
from .mdmkp_search import MDMKPEvaluation
from .graph_colouring import GraphColouringEvaluation
from .set_cover_construct import SetCoverEvaluation
from ._prepared_data import read_records

PREPARED_TASKS = ('fssp_gls', 'mdmkp_search', 'graph_colouring', 'set_cover_construct')
TASKS = ('tsp_construct', 'cvrp_aco', 'op_aco', 'online_bin_packing', 'vrptw_construct')
ALL_TASKS = TASKS + PREPARED_TASKS
CO_TASKS = ('tsp_construct', 'cvrp_aco') + PREPARED_TASKS
SUITES = {'legacy': TASKS, 'co6': CO_TASKS}
TASK_SHORT = dict(zip(TASKS, ('tsp', 'cvrp', 'op', 'obp', 'vrptw')))
TASK_SHORT.update(dict(zip(PREPARED_TASKS, ('fssp', 'mdmkp', 'gcol', 'scp'))))
NATIVE_MINIMIZE = set(ALL_TASKS) - {'op_aco', 'mdmkp_search'}
MINIMIZE = set(ALL_TASKS)  # Every evaluator returns a minimized scalar objective.
CLASSES = dict(zip(ALL_TASKS, (TSPEvaluation, CVRPACOEvaluation, OPACOEvaluation, OBPEvaluation, VRPTWEvaluation,
                                    FSSPGLSEvaluation, MDMKPEvaluation, GraphColouringEvaluation, SetCoverEvaluation)))
SELECTION_SEED = 20260927
DEFAULT_WORKERS = 4
TRAIN_TIMEOUT = {'online_bin_packing': 30, 'vrptw_construct': 30}
HELDOUT_TIMEOUT = {'tsp_construct': 3000, 'vrptw_construct': 1000, 'online_bin_packing': 1000,
                   'cvrp_aco': 3600, 'op_aco': 3600}
HELDOUT_TIMEOUT.update({task: 60 * CLASSES[task].DATASET.COUNTS['test'] / CLASSES[task].DATASET.COUNTS['train'] for task in PREPARED_TASKS})
SCALES = {'tsp_construct': (50, 100, 200), 'vrptw_construct': (50, 100, 200),
          'cvrp_aco': (20, 50, 100, 200), 'op_aco': (50, 100, 200),
          'online_bin_packing': ('1k_100', '1k_500', '5k_100', '5k_500', '10k_100', '10k_500')}
SCALES.update({task: (CLASSES[task].DATASET.SCALE,) for task in PREPARED_TASKS})
TEST_SCALES = {task: {50} for task in TASKS}
TEST_SCALES['online_bin_packing'] = {'1k_100', '1k_500', '5k_100', '5k_500'}
TEST_SCALES.update({task: {CLASSES[task].DATASET.SCALE} for task in PREPARED_TASKS})


def split_of_scale(task, scale):
    if task in PREPARED_TASKS:
        return 'test_' + str(scale)
    if task == 'online_bin_packing':
        items, capacity = str(scale).split('k_')
        return f'eval_{int(items)*1000}_{capacity}'
    return ('test_' if task in {'cvrp_aco', 'op_aco'} else 'eval_') + str(scale)


def scale_of_split(task, split):
    if task in PREPARED_TASKS:
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
    if task in PREPARED_TASKS:
        kwargs = dict(split='train', timeout_seconds=60)
        return CLASSES[task](**kwargs), kwargs
    if task in {'cvrp_aco', 'op_aco'}:
        cvrp = task == 'cvrp_aco'
        kwargs = dict(split='train', timeout_seconds=120 if cvrp else 60,
                      n_ants=30 if cvrp else 20, n_iterations=100 if cvrp else 50,
                      aco_seed=1234, n_workers=workers or DEFAULT_WORKERS)
    else:
        kwargs = get_generated_task_kwargs(task, 'train')
        if condition == 'traceaad':
            kwargs['timeout_seconds'] = TRAIN_TIMEOUT.get(task, kwargs['timeout_seconds'])
    return CLASSES[task](**kwargs), {'split': 'train', **kwargs}


def selection_task(task, search):
    if task in PREPARED_TASKS:
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
    timeout = HELDOUT_TIMEOUT[task] if timeout_seconds is None else timeout_seconds
    if workers < 1 or timeout <= 0 or not math.isfinite(timeout):
        raise ValueError('workers and timeout must be positive')
    if task in PREPARED_TASKS:
        if split != 'eval' and split not in SPLITS[task]:
            raise ValueError(f'unknown {task} held-out split: {split}')
        actual_split = 'test' if split == 'eval' else split
        if timeout_seconds is None:
            timeout = 60 * len(read_records(CLASSES[task].DATASET.DATA_ROOT, actual_split, task=task)) / CLASSES[task].DATASET.COUNTS['train']
        return CLASSES[task](split=actual_split, timeout_seconds=timeout)
    if split not in SPLITS[task] and not (split == 'eval' and task not in {'cvrp_aco', 'op_aco'}):
        raise ValueError(f'unknown {task} held-out split: {split}')
    if task in {'cvrp_aco', 'op_aco'}:
        cvrp = task == 'cvrp_aco'
        return CLASSES[task](split=split, timeout_seconds=timeout, n_workers=workers,
            n_ants=30 if cvrp else 20, n_iterations=100 if cvrp else 50, aco_seed=1234)
    kwargs = get_generated_task_kwargs(task, 'eval')
    if task == 'online_bin_packing' and split != 'eval':
        _, items, capacity = split.split('_')
        kwargs = obp_scale(kwargs, int(items), int(capacity))
    elif task != 'online_bin_packing':
        kwargs['problem_size'] = scale_of_split(task, split)
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
