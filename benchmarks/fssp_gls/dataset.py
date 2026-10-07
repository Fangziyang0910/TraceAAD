"""Fixed scale, data distribution and instance description for fssp_gls."""

from pathlib import Path

TASK = 'fssp_gls'
SCALE = 50
COUNTS = {'train': 16, 'test': 100, 'standard': 10}
DATA_ROOT = Path(__file__).with_name('data')
DISTRIBUTION = '50 jobs x 20 machines; independent integer processing times uniform in 1..99; reference is NEH makespan'


def describe(rows):
    dim = rows[0]['dimensions']
    size = f"{dim['jobs']} jobs and {dim['machines']} machines"
    phase = rows[0]['split']
    provenance = 'standard supplementary' if phase == 'standard' else 'generated'
    return f"{len(rows)} fixed {provenance} {phase} instances with {size}"
