"""Create a local reviewed-source runtime without copying credentials or run archives."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


def freeze(batch, prefix='v1011'):
    if not all(re.fullmatch(r'[A-Za-z0-9_-]+', s) for s in (batch, prefix)):
        raise ValueError('invalid batch or prefix')
    root = Path(__file__).resolve().parents[2]
    results = root / 'experiments/traceaad_v10_11/results'
    runtime = results / f'runtime_{batch}'
    runtime.mkdir(parents=True, exist_ok=False)
    # Import closure of experiments.traceaad_v10_11.run only: retired method
    # packages and unrelated task families are deliberately not frozen.
    sources = [
        root / 'core',
        root / 'traceaad/v10_11',
        root / 'traceaad/rand_ctx',
        root / 'experiments/infra',
        root / 'experiments/traceaad_v10_11',
    ]
    sources += [root / 'benchmarks' / task for task in (
        'tsp_construct', 'cvrp_aco', 'op_aco', 'online_bin_packing', 'vrptw_construct')]
    files = [
        root / 'experiments/__init__.py',
        root / 'core/__init__.py',
        root / 'traceaad/__init__.py',
        root / 'benchmarks/__init__.py',
        root / 'benchmarks/generated_data_config.py',
    ]
    for directory in sources:
        for folder, dirs, names in os.walk(directory):
            dirs[:] = sorted(d for d in dirs if d not in ('results', '__pycache__', 'data'))
            files.extend(Path(folder) / name for name in sorted(names)
                         if Path(name).suffix in ('.py', '.yaml', '.model'))
    hashes = {}
    for path in files:
        relative = path.relative_to(root)
        target = runtime / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    # Reuse private local credentials without copying them into the source manifest.
    if (root / '.env').is_file():
        (runtime / '.env').symlink_to(root / '.env')
    (runtime / 'experiments/traceaad_v10_11/results').symlink_to(results, target_is_directory=True)
    command = [sys.executable, '-m', 'experiments.traceaad_v10_11.launch', '--batch', batch,
               '--session-prefix', prefix, '--watch']
    payload = dict(batch=batch, runtime=str(runtime), python=sys.executable, python_version=sys.version,
                   git_base=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                   files=hashes, launch_command=command)
    (runtime / 'runtime_manifest.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True)
    parser.add_argument('--session-prefix', default='v1011')
    args = parser.parse_args()
    result = freeze(args.batch, args.session_prefix)
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    main()
