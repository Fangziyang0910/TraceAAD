"""Move explicitly selected inactive results into a verified archive without deleting payloads."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from traceaad.v12_1.storage import write_json


def archive_results(source, destination, paths):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists() or destination.is_relative_to(source):
        raise ValueError('Choose a new archive outside the active results directory')
    if not paths or any(Path(p).is_absolute() or '..' in Path(p).parts for p in paths):
        raise ValueError('Provide nonempty paths relative to the results directory')
    entries = sorted({source / p for p in paths})
    for entry in entries:
        if not entry.is_relative_to(source) or entry == source or not entry.exists():
            raise ValueError(f'Invalid archive entry: {entry}')
        if any(entry != other and entry.is_relative_to(other) for other in entries):
            raise ValueError('Archive entries must not overlap')
        configs = entry.rglob('run_config.json') if entry.is_dir() else []
        names = entry.rglob('*') if entry.is_dir() else [entry]
        if any('deepseek' in p.name.lower() for p in names):
            raise ValueError(f'DeepSeek logs or results are outside this legacy archive: {entry}')
        for config in configs:
            data = json.loads(config.read_text())
            if 'deepseek' in str((data.get('llm') or {}).get('model', '')).lower():
                raise ValueError(f'DeepSeek result is outside this legacy archive: {config}')
    destination.mkdir(parents=True)
    if source.stat().st_dev != destination.stat().st_dev:
        raise ValueError('Archive must be on the same filesystem for lossless renames')
    manifest = {'status': 'archiving', 'created_at': datetime.now(timezone.utc).isoformat(),
                'source': str(source), 'destination': str(destination),
                'paths': [str(p.relative_to(source)) for p in entries], 'moved': [],
                'reason': 'Old local-model experiments retired by the user'}
    write_json(destination / 'archive_manifest.json', manifest)
    records, cache = [], {}
    for entry in entries:
        files = entry.rglob('*') if entry.is_dir() else [entry]
        for path in files:
            if path.is_symlink():
                records.append({'path': str(path.relative_to(source)), 'symlink': os.readlink(path)})
            elif path.is_file():
                stat = path.stat()
                identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
                if identity not in cache:
                    digest = hashlib.sha256()
                    with path.open('rb') as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b''):
                            digest.update(block)
                    cache[identity] = digest.hexdigest()
                records.append({'path': str(path.relative_to(source)), 'bytes': stat.st_size,
                                'sha256': cache[identity], 'identity': list(identity)})
        print(f'inventoried {entry.relative_to(source)}', flush=True)
    with (destination / 'integrity.jsonl').open('w') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    for entry in entries:
        relative = entry.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        entry.rename(target)
        manifest['moved'].append(str(relative))
        write_json(destination / 'archive_manifest.json', manifest)
    for record in records:
        path = destination / record['path']
        if 'symlink' in record:
            assert path.is_symlink() and os.readlink(path) == record['symlink']
        else:
            stat = path.stat()
            # A rename preserves each file's inode, bytes and modification time.
            assert [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns] == record['identity'], path
    checksums = {r['path']: r.get('sha256') for r in records}
    metadata = [p for p in destination.rglob('*.json')
                if p.name in {'run_config.json', 'summary.json', 'manifest.json'} or p.name.startswith('batch_')]
    configs = [p for p in metadata if p.name == 'run_config.json']
    metadata.extend(p.parent / 'summary.json' for p in configs if not (p.parent / 'summary.json').exists())
    for path in metadata:
        original = path.read_bytes() if path.exists() else None
        data = json.loads(original) if original is not None else {}
        if not isinstance(data, dict):
            continue
        if original is not None:
            assert hashlib.sha256(original).hexdigest() == checksums[str(path.relative_to(destination))]
            backup = destination / 'original_metadata' / path.relative_to(destination)
            backup.parent.mkdir(parents=True, exist_ok=True)
            backup.write_bytes(original)
        data.update(archived_from_status=data.get('status'), status='archived',
                    archived_at=manifest['created_at'], archive_reason=manifest['reason'])
        for row in data.get('plan') or []:
            if isinstance(row, dict):
                row.update(archived_from_status=row.get('status'), status='archived')
        write_json(path, data)  # Atomic replacement avoids modifying hard-linked originals.
    manifest.update(status='archived', verified=True, files=len(records),
                    logical_bytes=sum(r.get('bytes', 0) for r in records), runs=len(configs),
                    metadata_files=len(metadata))
    write_json(destination / 'archive_manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--paths-file', type=Path, required=True)
    args = parser.parse_args()
    result = archive_results(args.source, args.destination, json.loads(args.paths_file.read_text()))
    print(json.dumps({k: result[k] for k in ('status', 'files', 'logical_bytes', 'runs', 'verified')}))


if __name__ == '__main__':
    main()
