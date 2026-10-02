"""Deduplicate immutable JSON archives without removing their logical paths.

Use only on completed, read-only archives. Restore independent files with the
manifest before editing or resuming a writer: hard links share future writes.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def save_manifest(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def plan(roots: list[Path]) -> dict:
    files = sorted({p.absolute() for root in roots for p in root.rglob("*.json")
                    if p.is_file() and not p.is_symlink()
                    and not any(parent.is_symlink() for parent in p.parents)})
    groups = {}
    records = []
    for path in files:
        info = path.stat()
        sha = digest(path)
        identity = (info.st_dev, info.st_size, stat.S_IMODE(info.st_mode), sha)
        source = groups.setdefault(identity, path)
        if source == path or (source.stat().st_dev, source.stat().st_ino) == (info.st_dev, info.st_ino):
            continue
        records.append({"path": str(path), "source": str(source), "sha256": sha,
                        "bytes": info.st_size, "mode": stat.S_IMODE(info.st_mode),
                        "atime_ns": info.st_atime_ns, "mtime_ns": info.st_mtime_ns})
    return {"created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "status": "planned", "roots": [str(p.absolute()) for p in roots],
            "n_files_scanned": len(files), "n_duplicates": len(records),
            "logical_duplicate_bytes": sum(r["bytes"] for r in records),
            "records": records,
            "write_contract": "Immutable archives only; restore before editing or resuming writers."}


def apply(manifest: dict, path: Path) -> dict:
    if path.exists():
        raise FileExistsError(f"Choose a new manifest; existing recovery record: {path}")
    # Persist all original metadata before replacing any duplicate.
    save_manifest(path, manifest)
    changed = 0
    for row in manifest["records"]:
        source, target = Path(row["source"]), Path(row["path"])
        if source.is_symlink() or target.is_symlink():
            raise ValueError(f"Archive path became a symlink: {target}")
        if digest(source) != row["sha256"] or digest(target) != row["sha256"]:
            raise ValueError(f"Archive changed since planning: {target}")
        fd, temporary = tempfile.mkstemp(prefix=target.name + ".dedup-", dir=target.parent)
        os.close(fd)
        os.unlink(temporary)
        try:
            os.link(source, temporary)
            if digest(Path(temporary)) != row["sha256"] or digest(target) != row["sha256"]:
                raise ValueError(f"Archive changed during deduplication: {target}")
            os.replace(temporary, target)
            changed += 1
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    for row in manifest["records"]:
        source, target = Path(row["source"]), Path(row["path"])
        assert digest(target) == row["sha256"]
        assert source.stat().st_ino == target.stat().st_ino
    manifest.update(status="deduplicated", n_linked=changed,
                    verified_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save_manifest(path, manifest)
    return manifest


def restore(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    # Preflight all payloads before any replacement, including a partial apply.
    for row in manifest["records"]:
        target = Path(row["path"])
        if target.is_symlink() or digest(target) != row["sha256"]:
            raise ValueError(f"Refusing to overwrite changed archive: {target}")
    for row in manifest["records"]:
        target = Path(row["path"])
        fd, temporary = tempfile.mkstemp(prefix=target.name + ".restore-", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as out, target.open("rb") as inp:
                shutil.copyfileobj(inp, out)
                out.flush()
                os.fsync(out.fileno())
            os.chmod(temporary, row["mode"])
            os.utime(temporary, ns=(row["atime_ns"], row["mtime_ns"]))
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    manifest.update(status="restored", restored_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save_manifest(path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="*", type=Path)
    parser.add_argument("--apply", action="store_true", help="default is a read-only plan")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--restore", type=Path, metavar="MANIFEST")
    args = parser.parse_args()
    if args.restore:
        if args.roots or args.apply or args.manifest:
            parser.error("--restore cannot be combined with planning options")
        result = restore(args.restore)
    else:
        if not args.roots or any(not p.is_dir() or p.is_symlink() for p in args.roots):
            parser.error("provide existing archive directories, without symlinks")
        if args.apply and args.manifest is None:
            parser.error("--apply requires a recovery --manifest")
        result = plan(args.roots)
        if args.apply:
            result = apply(result, args.manifest)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
