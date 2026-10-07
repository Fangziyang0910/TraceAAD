"""Fixed scale, data distribution and instance description for mdmkp_search."""

from pathlib import Path

TASK = 'mdmkp_search'
SCALE = 100
COUNTS = {'train': 18, 'test': 108}
DATA_ROOT = Path(__file__).with_name('data')
DISTRIBUTION = '100 items, 10 upper and 5 lower constraints. Fractions alpha=0.25/0.50/0.75 balanced by base group. A uniformly chosen alpha*100-item witness is planted independently of profits. Each coefficient row is sampled from independent uniform integers 1..1000, conditioned on the witness satisfying its bound floor(alpha*row_sum). Positive profits uniform 1..1000; mixed profits uniform -500..1000. Both profit variants share the base constraints and witness. Reference is a continuous LP relaxation upper bound; it is not an integer optimum.'


def describe(rows):
    dim = rows[0]['dimensions']
    size = f"{dim['items']} items, {dim['upper_constraints']} upper and {dim['lower_constraints']} lower constraints"
    phase = rows[0]['split']
    return f"{len(rows)} fixed generated {phase} instances with {size}; positive/mixed profits and constraint-tightness fractions 0.25/0.50/0.75 are balanced by base group"
