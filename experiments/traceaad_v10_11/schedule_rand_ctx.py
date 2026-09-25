"""Continuously fill all 27 slots with the random-reference ablation.

Context ablation arm: the formation-history block is replaced by eight archive
algorithms (idea + measured fitness) drawn per request by rank softmax. CVRP
repeats queue last; the idea-code barrier was lifted on 2026-09-16, so CVRP
starts as soon as slots free up.
"""

import subprocess
from pathlib import Path


BATCH = '20260916_v1011_rand_ctx'
PREFIX = 'v1011rc'


def main():
    root = Path(__file__).resolve().parents[2]
    subprocess.run([
        'uv', 'run', 'python', '-m', 'experiments.traceaad_v10_11.launch',
        '--batch', BATCH, '--session-prefix', PREFIX,
        '--rand-context', '--n-references', '8', '--repeats', '3',
        '--cvrp-last',
        '--backends', 'local,server1,server3,server3b',
        '--direct', '--watch', '--interval', '30',
    ], cwd=root, check=True)


if __name__ == '__main__':
    main()
