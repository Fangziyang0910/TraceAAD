"""Prepare independent training and test data from the fixed task distribution."""

from pathlib import Path

import numpy as np

from .. import _fixed_evaluation, _prepared_data
from .._prepared_data import prepare_cli, rng_for
from . import dataset
from .evaluation import solve
from .template import function_name, template_program


def generated_instances(split):
    namespace = {}
    exec(template_program, namespace)
    heuristic = namespace[function_name]
    for i in range(dataset.COUNTS[split]):
        rng, entropy = rng_for(2, split, i)
        group = f'generated_{split}_base{i:03d}'
        upper = np.triu(rng.random((300, 300)) < 0.5, k=1)
        arrays = {'adjacency': upper | upper.T}
        _, reference = solve(arrays, heuristic, reduction_attempts=0)
        yield arrays, {'id': group, 'group': group, 'seed_entropy': entropy, 'source_kind': 'generated',
            'scale': 300, 'dimensions': {'vertices': 300},
            'density': float(arrays['adjacency'].sum() / (300*299))}, float(reference), 'DSATUR feasible upper bound'


if __name__ == '__main__':
    prepare_cli(dataset, generated_instances,
                [Path(__file__), Path(dataset.__file__), Path(__file__).with_name('evaluation.py'),
                 Path(__file__).with_name('template.py'), Path(_prepared_data.__file__),
                 Path(_fixed_evaluation.__file__)])
