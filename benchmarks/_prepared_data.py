"""Portable dataset loading and preparation; problem rules live in task directories."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import tempfile

import numpy as np

PROTOCOL = 'ahd-six-tasks-v3-training-selection'
SEED = 20261007


def read_manifest(root):
    path = Path(root) / 'manifest.json'
    if not path.exists():
        raise FileNotFoundError(f'Missing prepared task data: {path}')
    result = json.loads(path.read_text())
    if result.get('protocol') != PROTOCOL:
        raise ValueError(f'Obsolete data protocol: {path}')
    return result


def read_records(root, split, *, task=None):
    if split in {'train', 'test'}:
        phase, scale = split, None
    elif split.startswith('test_') and split[5:].isdigit():
        phase, scale = 'test', int(split[5:])
    else:
        raise ValueError(f'unknown prepared-data split: {split}')
    data = read_manifest(root)
    if task is not None and data['task'] != task:
        raise ValueError(f'dataset belongs to {data["task"]}, expected {task}')
    rows = [r for r in data['instances'] if r['split'] == phase and (scale is None or r['scale'] == scale)]
    if not rows:
        raise ValueError(f'empty split: {root}/{split}')
    if len({tuple(sorted(r['dimensions'].items())) for r in rows}) != 1:
        raise ValueError(f'mixed dimensions in fixed-scale split: {root}/{split}')
    return rows


def load_instance(record, root):
    path = Path(root) / record['file']
    if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
        raise ValueError(f'task instance changed: {path}')
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def digest_arrays(arrays):
    h = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        value = np.asarray(value)
        h.update(key.encode())
        h.update(str((value.shape, value.dtype.str)).encode())
        h.update(value.tobytes())
    return h.hexdigest()


def rng_for(stream, split, index):
    entropy = [SEED, stream, {'train': 0, 'test': 2}[split], index]
    return np.random.default_rng(np.random.SeedSequence(entropy)), entropy


def prepare(dataset, generated, destination, implementation_files):
    """Write a complete task dataset, then publish its manifest and remove obsolete records."""
    import numba
    import scipy
    root = Path(destination)
    root.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads((root / 'manifest.json').read_text()) if (root / 'manifest.json').exists() else {}
    repository = Path(__file__).resolve().parent.parent
    hashes = {str(Path(path).resolve().relative_to(repository)): hashlib.sha256(Path(path).read_bytes()).hexdigest()
              for path in implementation_files}
    report = {'protocol': PROTOCOL, 'task': dataset.TASK, 'seed': SEED, 'final_selection': 'training',
              'primary_source': 'TraceAAD explicit generated distributions', 'distribution': dataset.DISTRIBUTION,
              'split_policy': 'Independent train/test seed streams; base-group variants stay together.',
              'preparation_environment': {'python': platform.python_version(), 'numpy': np.__version__,
                                          'scipy': scipy.__version__, 'numba': numba.__version__},
              'source_files': hashes, 'instances': []}
    with tempfile.TemporaryDirectory(prefix='prepare_', dir=root.parent) as temp:
        staged = Path(temp)
        for split in ('train', 'test'):
            print(f'preparing {dataset.TASK}/{split}', flush=True)
            for arrays, metadata, reference, kind in generated(split):
                if not np.isfinite(reference) or reference <= 0:
                    raise ValueError(f'non-positive reference: {metadata["id"]}')
                relative = Path(split) / (metadata['id'] + '.npz')
                target = staged / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(target, **arrays)
                report['instances'].append({**metadata, 'split': split, 'reference': float(reference), 'reference_kind': kind,
                    'content': digest_arrays(arrays), 'file': str(relative), 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
        for field in ('group', 'content'):
            used = {}
            for row in report['instances']:
                if used.setdefault(row[field], row['split']) != row['split']:
                    raise ValueError(f'split leakage: {row["id"]}')
        report['counts'] = {s: sum(r['split'] == s for r in report['instances']) for s in ('train', 'test')}
        root.mkdir(parents=True, exist_ok=True)
        for row in report['instances']:
            target = root / row['file']
            target.parent.mkdir(parents=True, exist_ok=True)
            (staged / row['file']).replace(target)
        path = staged / 'manifest.json'
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        path.replace(root / 'manifest.json')
        kept = {r['file'] for r in report['instances']}
        for row in previous.get('instances', []):
            if row['file'] not in kept:
                (root / row['file']).unlink(missing_ok=True)
    print(dataset.TASK, report['counts'], flush=True)
    return report


def prepare_cli(dataset, generated, implementation_files):
    parser = argparse.ArgumentParser(description=f'Prepare {dataset.TASK} independent train/test data')
    parser.add_argument('--output', type=Path, default=dataset.DATA_ROOT)
    args = parser.parse_args()
    prepare(dataset, generated, args.output, implementation_files)
