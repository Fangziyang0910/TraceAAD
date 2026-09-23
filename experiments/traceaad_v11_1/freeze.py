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


def runtime_environment(runtime):
    environment = dict(os.environ)
    environment['PYTHONPATH'] = str(Path(runtime).resolve())
    return environment


def verify_runtime(runtime, *, preflight=True):
    runtime = Path(runtime).resolve()
    payload = json.loads((runtime / 'runtime_manifest.json').read_text())
    for relative, expected in payload['files'].items():
        source = runtime / relative
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f'frozen source changed or missing: {relative}')
        if source.suffix == '.py':
            compile(source.read_text(), str(source), 'exec')
    if preflight:
        probe = """
import __main__, multiprocessing, os
from pathlib import Path
import experiments.traceaad_v11_1.run as entry
assert Path(entry.__file__).resolve().is_relative_to(Path.cwd().resolve())
__main__.__spec__ = entry.__spec__
worker = multiprocessing.get_context('spawn').Process(target=os.getpid)
worker.start(); worker.join(30)
if worker.is_alive(): worker.terminate(); worker.join(); raise RuntimeError('spawn bootstrap timed out')
assert worker.exitcode == 0, worker.exitcode
"""
        subprocess.run([sys.executable, '-c', probe], cwd=runtime,
                       env=runtime_environment(runtime), check=True, timeout=45)
    return hashlib.sha256((runtime / 'runtime_manifest.json').read_bytes()).hexdigest()


def freeze(batch, prefix='v111'):
    if not all(re.fullmatch(r'[A-Za-z0-9_-]+', s) for s in (batch, prefix)):
        raise ValueError('invalid batch or prefix')
    root = Path(__file__).resolve().parents[2]
    results = root / 'experiments/traceaad_v11_1/results'
    runtime = results / f'runtime_{batch}'
    runtime.mkdir(parents=True, exist_ok=False)
    # Import closure of experiments.traceaad_v11_1.run only: retired method
    # packages and unrelated task families are deliberately not frozen.
    sources = [
        root / 'core',
        root / 'traceaad/v11_1',
        root / 'experiments/infra',
        root / 'experiments/traceaad_v11_1',
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
    (runtime / 'experiments/traceaad_v11_1/results').symlink_to(results, target_is_directory=True)
    payload = dict(batch=batch, runtime=str(runtime), python=sys.executable, python_version=sys.version,
                   git_base=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                   files=hashes)
    (runtime / 'runtime_manifest.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True)
    parser.add_argument('--session-prefix', default='v111')
    args = parser.parse_args()
    result = freeze(args.batch, args.session_prefix)
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    main()
