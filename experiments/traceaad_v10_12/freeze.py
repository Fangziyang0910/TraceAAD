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


def freeze(batch, prefix='v1012'):
    if not all(re.fullmatch(r'[A-Za-z0-9_-]+', s) for s in (batch, prefix)):
        raise ValueError('invalid batch or prefix')
    root = Path(__file__).resolve().parents[2]
    results = root / 'experiments/traceaad_v10_12/results'
    runtime = results / f'runtime_{batch}'
    runtime.mkdir(parents=True, exist_ok=False)
    # Import closure of experiments.traceaad_v10_12.run only: retired method
    # packages and unrelated task families are deliberately not frozen.
    sources = [
        root / 'llm4ad/base',
        root / 'llm4ad/tools',
        root / 'traceaad/v10_12',
        root / 'experiments/infra',
        root / 'experiments/traceaad_v10_12',
    ]
    sources += [root / 'benchmarks' / task for task in (
        'tsp_construct', 'cvrp_aco', 'op_aco', 'online_bin_packing', 'vrptw_construct')]
    files = [
        root / 'experiments/__init__.py',
        root / 'llm4ad/__init__.py',
        root / 'traceaad/__init__.py',
        root / 'benchmarks/__init__.py',
        root / 'benchmarks/generated_data_config.py',
    ]
    for directory in sources:
        if not directory.is_dir():
            raise FileNotFoundError(f'missing runtime source directory: {directory}')
        for folder, dirs, names in os.walk(directory):
            dirs[:] = sorted(d for d in dirs if d not in ('__pycache__', 'data')
                             and not d.startswith(('results', 'backup_', 'analysis_')))
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
    (runtime / 'experiments/traceaad_v10_12/results').symlink_to(results, target_is_directory=True)
    command = [sys.executable, '-m', 'experiments.traceaad_v10_12.launch', '--batch', batch,
               '--session-prefix', prefix, '--watch']
    payload = dict(batch=batch, runtime=str(runtime), python=sys.executable, python_version=sys.version,
                   git_base=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                   files=hashes, launch_command=command)
    (runtime / 'runtime_manifest.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    return payload


def runtime_environment(runtime):
    """Ensure the parent and spawned workers import the same frozen source."""
    environment = dict(os.environ)
    environment['PYTHONPATH'] = str(Path(runtime).resolve())
    return environment


def verify_runtime(runtime, *, preflight=True):
    runtime = Path(runtime).resolve()
    manifest = runtime / 'runtime_manifest.json'
    payload = json.loads(manifest.read_text())
    for relative, expected in payload['files'].items():
        source = runtime / relative
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f'frozen source changed or missing: {relative}')
        if source.suffix == '.py':
            compile(source.read_text(), str(source), 'exec')
    if preflight:
        # -m search workers re-import their entry point when ACO starts a spawn
        # pool. Check that exact bootstrap before spending any evaluator slots.
        probe = """
import __main__, multiprocessing, os
from pathlib import Path
import experiments.traceaad_v10_12.run as entry
assert Path(entry.__file__).resolve().is_relative_to(Path.cwd().resolve())
__main__.__spec__ = entry.__spec__
worker = multiprocessing.get_context('spawn').Process(target=os.getpid)
worker.start()
worker.join(30)
if worker.is_alive():
    worker.terminate()
    worker.join()
    raise RuntimeError('frozen worker bootstrap timed out')
assert worker.exitcode == 0, worker.exitcode
"""
        subprocess.run([sys.executable, '-c', probe], cwd=runtime,
                       env=runtime_environment(runtime), check=True, timeout=45)
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True)
    parser.add_argument('--session-prefix', default='v1012')
    args = parser.parse_args()
    result = freeze(args.batch, args.session_prefix)
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    main()
