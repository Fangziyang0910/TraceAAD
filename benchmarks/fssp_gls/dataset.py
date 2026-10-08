"""Fixed scale, data distribution and instance description for fssp_gls."""

from .._seeded_data import canonical_split, rng_for

TASK = 'fssp_gls'
SCALE = 50
COUNTS = {'train': 16, 'test': 50}
DIMENSIONS = {'jobs': 50, 'machines': 20}
DISTRIBUTION = '50 jobs x 20 machines; independent integer processing times uniform in 1..99; reference is NEH makespan'


def describe(split, count=None):
    phase = canonical_split(split, SCALE)
    count = COUNTS[phase] if count is None else count
    dim = DIMENSIONS
    size = f"{dim['jobs']} jobs and {dim['machines']} machines"
    return f"{count} fixed generated {phase} instances with {size}"


def generate_instances(split):
    from .evaluation import makespan, neh
    split = canonical_split(split, SCALE)

    for i in range(COUNTS[split]):
        rng, entropy = rng_for(0, split, i)
        group = f'generated_{split}_base{i:03d}'
        times = rng.integers(1, 100, (50, 20)).astype(float)
        yield {'processing_times': times}, {
            'id': group, 'group': group, 'seed_entropy': entropy, 'source_kind': 'generated',
            'scale': 50, 'dimensions': {'jobs': 50, 'machines': 20}}, float(makespan(neh(times), times)), 'NEH feasible upper bound'
