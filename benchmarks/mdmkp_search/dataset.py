"""Fixed scale, data distribution and instance description for mdmkp_search."""

import numpy as np

from .._seeded_data import canonical_split, rng_for

TASK = 'mdmkp_search'
SCALE = 100
COUNTS = {'train': 18, 'test': 108}
DIMENSIONS = {'items': 100, 'upper_constraints': 10, 'lower_constraints': 5}
DISTRIBUTION = '100 items, 10 upper and 5 lower constraints. Fractions alpha=0.25/0.50/0.75 balanced by base group. A uniformly chosen alpha*100-item witness is planted independently of profits. Each coefficient row is sampled from independent uniform integers 1..1000, conditioned on the witness satisfying its bound floor(alpha*row_sum). Positive profits uniform 1..1000; mixed profits uniform -500..1000. Both profit variants share the base constraints and witness. Reference is a continuous LP relaxation upper bound; it is not an integer optimum.'


def describe(split, count=None):
    phase = canonical_split(split, SCALE)
    count = COUNTS[phase] if count is None else count
    dim = DIMENSIONS
    size = f"{dim['items']} items, {dim['upper_constraints']} upper and {dim['lower_constraints']} lower constraints"
    return f"{count} fixed generated {phase} instances with {size}; positive/mixed profits and constraint-tightness fractions 0.25/0.50/0.75 are balanced by base group"


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
    import warnings
    from scipy.optimize import linprog, OptimizeWarning

    # SciPy forwards HiGHS' threads option; use one thread before later evaluator forks.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=OptimizeWarning,
                                message='Unrecognized options detected:.*threads')
        result = linprog(-data["cost_vector"].astype(float),
                         A_ub=np.vstack((data["A_leq"], -data["A_geq"])),
                         b_ub=np.r_[data["b_leq"], -data["b_geq"]], bounds=(0, 1), method="highs",
                         options={'threads': 1})
    if not result.success or -result.fun <= 0:
        raise ValueError(f"could not compute positive MDMKP LP reference: {result.message}")
    return float(-result.fun)


def generate_instances(split):
    split = canonical_split(split, SCALE)

    for i in range(COUNTS[split] // 2):
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
