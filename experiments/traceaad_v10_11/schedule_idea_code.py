"""Continuously fill all 27 slots with the Idea+full-code ablation."""

import subprocess
from pathlib import Path


BATCH = '20260915_v1011_idea_code'
PREFIX = 'v1011ic'


def main():
    root = Path(__file__).resolve().parents[2]
    subprocess.run([
        'uv', 'run', 'python', '-m', 'experiments.traceaad_v10_11.launch',
        '--batch', BATCH, '--session-prefix', PREFIX,
        '--history-code', '--traj-gens', '8', '--repeats', '3',
        '--cvrp-last',
        '--cvrp-barrier-batches',
        '20260915_v1011_generic,20260915_v1011_no_traj_idea',
        '--backends', 'local,server1,server3,server3b',
        '--direct', '--watch', '--interval', '30',
    ], cwd=root, check=True)


if __name__ == '__main__':
    main()
