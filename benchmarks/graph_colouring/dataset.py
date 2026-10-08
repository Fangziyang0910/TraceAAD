"""Fixed scale, data distribution and instance description for graph_colouring."""

import numpy as np

from .._seeded_data import canonical_split, rng_for

TASK = 'graph_colouring'
SCALE = 300
COUNTS = {'train': 16, 'test': 50}
DIMENSIONS = {'vertices': 300}
DISTRIBUTION = 'G(300,0.5): each undirected edge independently present with probability 0.5; reference is deterministic DSATUR construction'


def describe(split, count=None):
    phase = canonical_split(split, SCALE)
    count = COUNTS[phase] if count is None else count
    dim = DIMENSIONS
    size = f"{dim['vertices']} vertices (edge density about 0.5)"
    return f"{count} fixed generated {phase} instances with {size}"


def generate_instances(split):
    from .evaluation import solve
    from .template import function_name, template_program
    split = canonical_split(split, SCALE)

    namespace = {}
    exec(template_program, namespace)
    heuristic = namespace[function_name]
    for i in range(COUNTS[split]):
        rng, entropy = rng_for(2, split, i)
        group = f'generated_{split}_base{i:03d}'
        upper = np.triu(rng.random((300, 300)) < 0.5, k=1)
        arrays = {'adjacency': upper | upper.T}
        _, reference = solve(arrays, heuristic, reduction_attempts=0)
        yield arrays, {'id': group, 'group': group, 'seed_entropy': entropy, 'source_kind': 'generated',
            'scale': 300, 'dimensions': {'vertices': 300},
            'density': float(arrays['adjacency'].sum() / (300*299))}, float(reference), 'DSATUR feasible upper bound'
