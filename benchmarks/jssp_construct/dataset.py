"""Independent seeded static 20-job/20-machine instances, no stored arrays."""
import numpy as np
from .._seeded_data import canonical_split, rng_for

TASK = 'jssp_construct'
SCALE = 20
COUNTS = {'train': 16, 'test': 50}
DIMENSIONS = {'jobs': 20, 'machines': 20, 'operations': 400}
DISTRIBUTION = ('Static 20 jobs x 20 machines. Each job independently visits a uniformly '
                'random permutation of all 20 machines; durations are independent integers '
                '1..99. Reference is Giffler-Thompson construction with most remaining work '
                '(MWKR). This is a project-defined Taillard-style distribution, not the '
                'published Taillard instances or random generator.')


def describe(split, count=None):
    phase = canonical_split(split, SCALE)
    count = COUNTS[phase] if count is None else count
    return f'{count} fixed generated {phase} instances with 20 jobs, 20 machines and 400 operations'


def generate_instances(split):
    from .evaluation import solve
    from .template import template_program, function_name
    split = canonical_split(split, SCALE)
    namespace = {}
    exec(template_program, namespace)
    for i in range(COUNTS[split]):
        rng, entropy = rng_for(4, split, i)
        arrays = {'processing_times': rng.integers(1, 100, (20, 20), dtype=np.int64),
                  'machine_order': np.array([rng.permutation(20) for _ in range(20)], dtype=np.int64)}
        _, reference = solve(arrays, namespace[function_name])
        group = f'generated_{split}_base{i:03d}'
        yield arrays, {'id': group, 'group': group, 'seed_entropy': entropy,
                      'source_kind': 'generated', 'scale': SCALE, 'dimensions': DIMENSIONS}, reference, 'MWKR feasible makespan'
