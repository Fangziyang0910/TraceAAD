"""Fixed scale, data distribution and instance description for set_cover_construct."""

import numpy as np

from .._seeded_data import canonical_split, rng_for

TASK = 'set_cover_construct'
SCALE = 2000
COUNTS = {'train': 16, 'test': 100}
DIMENSIONS = {'elements': 200, 'sets': 2000}
DISTRIBUTION = '200 elements x 2000 sets; each element chooses 40 distinct sets uniformly, independently of other elements; costs independently uniform integers 1..100; reference is gain/cost construction plus redundant-set deletion'


def describe(split, count=None):
    phase = canonical_split(split, SCALE)
    count = COUNTS[phase] if count is None else count
    dim = DIMENSIONS
    size = f"{dim['elements']} elements and {dim['sets']} sets (coverage density about 0.02)"
    return f"{count} fixed generated {phase} instances with {size}"


def generate_instances(split):
    from .evaluation import solve
    from .template import function_name, template_program
    split = canonical_split(split, SCALE)

    namespace = {}
    exec(template_program, namespace)
    heuristic = namespace[function_name]
    for i in range(COUNTS[split]):
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
