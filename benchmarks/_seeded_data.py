"""Fixed in-memory datasets; task modules own their distributions and references."""

import hashlib
from itertools import islice

import numpy as np

PROTOCOL = 'ahd-six-tasks-v4-seeded-runtime'
SEED = 20261007


def canonical_split(split, scale):
    if split in {'train', 'test'}:
        return split
    if split == f'test_{scale}':
        return 'test'
    raise ValueError(f'unknown generated-data split: {split}')


def rng_for(stream, split, index):
    entropy = [SEED, stream, {'train': 0, 'test': 2}[split], index]
    return np.random.default_rng(np.random.SeedSequence(entropy)), entropy


def digest_arrays(arrays):
    digest = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        value = np.asarray(value)
        digest.update(key.encode())
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def generate_dataset(dataset, split, limit=None):
    """Build inputs and references once, outside candidate execution and timing."""
    phase = canonical_split(split, dataset.SCALE)
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError('limit must be a positive integer')
    count = min(limit, dataset.COUNTS[phase]) if limit is not None else dataset.COUNTS[phase]
    rows, instances = [], []
    for arrays, metadata, reference, kind in islice(dataset.generate_instances(phase), count):
        if not np.isfinite(reference) or reference <= 0:
            raise ValueError(f'non-positive reference: {metadata["id"]}')
        if metadata['scale'] != dataset.SCALE or metadata['dimensions'] != dataset.DIMENSIONS:
            raise ValueError(f'generated dimensions do not match {dataset.TASK}')
        rows.append({**metadata, 'split': phase, 'reference': float(reference),
                     'reference_kind': kind, 'content': digest_arrays(arrays)})
        instances.append(arrays)
    if len(rows) != count or not rows:
        raise ValueError(f'incorrect instance count for {dataset.TASK}/{phase}')
    return rows, instances
