"""Fixed scale, data distribution and instance description for graph_colouring."""

from pathlib import Path

TASK = 'graph_colouring'
SCALE = 300
COUNTS = {'train': 16, 'test': 100, 'standard': 10}
DATA_ROOT = Path(__file__).with_name('data')
DISTRIBUTION = 'G(300,0.5): each undirected edge independently present with probability 0.5; reference is deterministic DSATUR construction'


def describe(rows):
    dim = rows[0]['dimensions']
    size = f"{dim['vertices']} vertices (edge density about 0.5)"
    phase = rows[0]['split']
    provenance = 'standard supplementary' if phase == 'standard' else 'generated'
    return f"{len(rows)} fixed {provenance} {phase} instances with {size}"
