"""Freeze and continuously fill all 27 slots with the Idea+full-code ablation."""

import subprocess
import sys
from pathlib import Path

from .freeze import freeze


BATCH = '20260915_v1012_idea_code'
PREFIX = 'v1012ic'


def main():
    root = Path(__file__).resolve().parents[2]
    runtime = root / 'experiments/traceaad_v10_12/results' / f'runtime_{BATCH}'
    if not runtime.exists():
        freeze(BATCH, PREFIX)
    subprocess.run([
        sys.executable, '-m', 'experiments.traceaad_v10_12.launch',
        '--batch', BATCH, '--session-prefix', PREFIX,
        '--history-code', '--traj-gens', '8', '--repeats', '3',
        '--cvrp-last',
        '--cvrp-barrier-batches',
        '20260915_v1012_generic,20260915_v1012_no_traj_idea',
        '--backends', 'local,server1,server3,server3b',
        '--direct', '--watch', '--interval', '30',
    ], cwd=runtime, check=True)


if __name__ == '__main__':
    main()
