"""Run a lower-priority command after all prerequisite batches finish."""

import argparse
import json
import os
from pathlib import Path
import time


def dependencies_finished(paths):
    """Missing, blocked and stopped prerequisites must not release the queue."""
    for path in paths:
        if not path.is_file():
            return False
        data = json.loads(path.read_text())
        plan = data.get('plan', [])
        if not plan or any(row.get('status') != 'finished' for row in plan):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--after-batch', type=Path, action='append', required=True)
    parser.add_argument('--interval', type=int, default=30)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.interval < 1:
        parser.error('interval must be positive')
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if command[:3] != ['uv', 'run', 'python']:
        parser.error('command must start with: uv run python')
    print('Waiting for priority batches: ' + ', '.join(str(p) for p in args.after_batch), flush=True)
    while not dependencies_finished(args.after_batch):
        time.sleep(args.interval)
    print('Priority batches finished; starting queued command.', flush=True)
    os.chdir(Path(__file__).resolve().parents[2])
    os.execvp(command[0], command)


if __name__ == '__main__':
    main()
