"""Prepare independent primary data and separate standard supplementary data."""

from pathlib import Path

import numpy as np

from .. import _fixed_evaluation, _prepared_data
from .._prepared_data import prepare_cli, rng_for, source_cases, standard_metadata
from . import dataset
from .evaluation import solve
from .template import function_name, template_program


def generated_instances(split):
    namespace = {}
    exec(template_program, namespace)
    heuristic = namespace[function_name]
    for i in range(dataset.COUNTS[split]):
        rng, entropy = rng_for(3, split, i)
        group = f'generated_{split}_base{i:03d}'
        coverage = np.zeros((200, 2000), dtype=bool)
        for row in coverage:
            row[rng.choice(2000, 40, replace=False)] = True
        arrays = {'coverage': coverage, 'costs': rng.integers(1, 101, 2000).astype(float)}
        _, reference = solve(arrays, heuristic, improvement_steps=0)
        yield arrays, {'id': group, 'group': group, 'seed_entropy': entropy, 'source_kind': 'generated',
            'scale': 2000, 'dimensions': {'elements': 200, 'sets': 2000},
            'density': float(coverage.mean())}, float(reference), 'greedy feasible upper bound'


def standard_instances(source, hashes):
    folder = Path(source) / 'Set covering'
    for name, index, case, reference in source_cases(folder, [f'scp5{i}.txt' for i in range(1, 11)], hashes):
        coverage = np.zeros((200, 2000), dtype=bool)
        for row, columns in enumerate(case['row_cover']):
            coverage[row, np.asarray(columns)-1] = True
        arrays = {'coverage': coverage, 'costs': np.asarray(case['costs'], dtype=float)}
        meta = standard_metadata(name, index, int(case['n']))
        meta.update(dimensions={'elements': 200, 'sets': 2000}, density=float(coverage.mean()))
        yield arrays, meta, float(reference), 'CO-Bench published reference'


if __name__ == '__main__':
    prepare_cli(dataset, generated_instances, standard_instances,
                [Path(__file__), Path(dataset.__file__), Path(__file__).with_name('evaluation.py'),
                 Path(__file__).with_name('template.py'), Path(_prepared_data.__file__),
                 Path(_fixed_evaluation.__file__)])
