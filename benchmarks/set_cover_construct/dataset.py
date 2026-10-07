"""Fixed scale, data distribution and instance description for set_cover_construct."""

from pathlib import Path

TASK = 'set_cover_construct'
SCALE = 2000
COUNTS = {'train': 16, 'test': 100}
DATA_ROOT = Path(__file__).with_name('data')
DISTRIBUTION = '200 elements x 2000 sets; each element chooses 40 distinct sets uniformly, independently of other elements; costs independently uniform integers 1..100; reference is gain/cost construction plus redundant-set deletion'


def describe(rows):
    dim = rows[0]['dimensions']
    size = f"{dim['elements']} elements and {dim['sets']} sets (coverage density about 0.02)"
    phase = rows[0]['split']
    return f"{len(rows)} fixed generated {phase} instances with {size}"
