"""Prepare independent training and test data from the fixed task distribution."""

from pathlib import Path

import numpy as np

from .. import _fixed_evaluation, _prepared_data
from .._prepared_data import prepare_cli, rng_for
from . import dataset
from scipy.optimize import linprog


def constrained_rows(rng, x, alpha, count, upper):
    rows, limits = [], []
    while len(rows) < count:
        row = rng.integers(1, 1001, len(x), dtype=np.int64)
        limit = int(alpha * int(row.sum()))
        used = int(row @ x)
        accepted = used <= limit if upper else used >= limit
        if accepted:
            rows.append(row)
            limits.append(limit)
    return np.array(rows), np.array(limits)


def lp_bound(data):
    result = linprog(-data["cost_vector"].astype(float),
                     A_ub=np.vstack((data["A_leq"], -data["A_geq"])),
                     b_ub=np.r_[data["b_leq"], -data["b_geq"]], bounds=(0, 1), method="highs")
    if not result.success or -result.fun <= 0:
        raise ValueError(f"could not compute positive MDMKP LP reference: {result.message}")
    return float(-result.fun)


def generated_instances(split):
    for i in range(dataset.COUNTS[split] // 2):
        rng, entropy = rng_for(1, split, i)
        group = f'generated_{split}_base{i:03d}'
        alpha = (0.25, 0.5, 0.75)[i % 3]
        x = np.zeros(100, dtype=np.int8)
        x[rng.choice(100, int(100*alpha), replace=False)] = 1
        a, b = constrained_rows(rng, x, alpha, 10, True)
        g, d = constrained_rows(rng, x, alpha, 5, False)
        base = {'A_leq': a, 'b_leq': b, 'A_geq': g, 'b_geq': d, 'initial_solution': x}
        dim = {'items': 100, 'upper_constraints': 10, 'lower_constraints': 5}
        for cost_type, low in (('positive', 1), ('mixed', -500)):
            arrays = {**base, 'cost_vector': rng.integers(low, 1001, 100, dtype=np.int64)}
            yield arrays, {'id': f'{group}_{cost_type}', 'group': group, 'seed_entropy': entropy,
                'source_kind': 'generated', 'scale': 100, 'dimensions': dim,
                'tightness': alpha, 'cost_type': cost_type}, lp_bound(arrays), 'LP relaxation upper bound'


if __name__ == '__main__':
    prepare_cli(dataset, generated_instances,
                [Path(__file__), Path(dataset.__file__), Path(__file__).with_name('evaluation.py'),
                 Path(__file__).with_name('template.py'), Path(_prepared_data.__file__),
                 Path(_fixed_evaluation.__file__)])
