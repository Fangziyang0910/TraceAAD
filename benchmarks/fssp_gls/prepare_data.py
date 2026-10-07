"""Prepare independent primary data and separate standard supplementary data."""

from pathlib import Path

import numpy as np

from .. import _fixed_evaluation, _prepared_data
from .._prepared_data import prepare_cli, rng_for, source_cases, standard_metadata
from . import dataset
from .evaluation import makespan, neh


def generated_instances(split):
    for i in range(dataset.COUNTS[split]):
        rng, entropy = rng_for(0, split, i)
        group = f'generated_{split}_base{i:03d}'
        times = rng.integers(1, 100, (50, 20)).astype(float)
        yield {'processing_times': times}, {
            'id': group, 'group': group, 'seed_entropy': entropy, 'source_kind': 'generated',
            'scale': 50, 'dimensions': {'jobs': 50, 'machines': 20}}, float(makespan(neh(times), times)), 'NEH feasible upper bound'


def standard_instances(source, hashes):
    folder = Path(source) / 'Flow shop scheduling'
    for name, index, case, _ in source_cases(folder, ['tai50_20.txt'], hashes):
        arrays = {'processing_times': np.asarray(case['matrix'], dtype=float)}
        meta = standard_metadata(name, index, int(case['n']))
        meta['dimensions'] = {'jobs': 50, 'machines': 20}
        yield arrays, meta, float(case['upper_bound']), 'published Taillard upper bound'


if __name__ == '__main__':
    prepare_cli(dataset, generated_instances, standard_instances,
                [Path(__file__), Path(dataset.__file__), Path(__file__).with_name('evaluation.py'),
                 Path(__file__).with_name('template.py'), Path(_prepared_data.__file__),
                 Path(_fixed_evaluation.__file__)])
