"""Small file helpers shared by experiment writers and readers."""

import json
import os
from pathlib import Path
import tempfile
import gzip
import shutil

JOURNAL_NAME = "events.jsonl"


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path, limit=None):
    path = Path(path)
    if not path.exists():
        return
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        for line in handle:
            if not line.endswith(b"\n") or limit is not None and handle.tell() > limit:
                break
            yield json.loads(line)


def committed_size(run_dir, name=JOURNAL_NAME):
    """The checkpoint is the common definition of visible search history."""
    run_dir = Path(run_dir)
    path = run_dir / name
    if not path.exists():
        return 0
    size = path.stat().st_size
    return min(size, read_json(run_dir / "resume.json", {}).get("files", {}).get(name, size))


def committed_rows(run_dir):
    yield from rows(Path(run_dir) / JOURNAL_NAME, committed_size(run_dir))


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
    ).encode()
    detach_link(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "ab") as handle:
        handle.write(payload)
    return path.stat().st_size


def detach_link(path):
    """Completed archives can share an inode; a resumed writer owns its copy."""
    path = Path(path)
    if path.exists() and path.stat().st_nlink > 1:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
        shutil.copy2(path, temporary)
        temporary.replace(path)


def repair_jsonl_tail(path, backup_dir):
    """A stopped writer can leave a partial line; preserve it before appending again."""
    path = Path(path)
    size = path.stat().st_size
    if not size:
        return
    with path.open("rb") as source:
        source.seek(size - 1)
        if source.read(1) == b"\n":
            return
        start, boundary = size, 0
        while start:
            offset = max(0, start - 65536)
            source.seek(offset)
            newline = source.read(start - offset).rfind(b"\n")
            if newline >= 0:
                boundary = offset + newline + 1
                break
            start = offset
        backup = Path(backup_dir)
        backup.mkdir(parents=True, exist_ok=True)
        source.seek(boundary)
        with tempfile.NamedTemporaryFile(
            dir=backup, prefix=path.name + ".", suffix=".partial", delete=False
        ) as handle:
            shutil.copyfileobj(source, handle)
    detach_link(path)
    with path.open("r+b") as handle:
        handle.truncate(boundary)


def seal_calls(run_dir):
    run_dir = Path(run_dir)
    source = run_dir / "calls.jsonl"
    if not source.exists():
        return
    target = run_dir / "calls.jsonl.gz"
    with tempfile.NamedTemporaryFile(dir=run_dir, suffix=".tmp", delete=False) as handle:
        destination = Path(handle.name)
    try:
        if target.exists():
            with target.open("rb") as previous, destination.open("wb") as handle:
                shutil.copyfileobj(previous, handle)
        with (
            source.open("rb") as original,
            gzip.open(destination, "ab", compresslevel=1) as compressed,
        ):
            shutil.copyfileobj(original, compressed)
        destination.replace(target)
    finally:
        destination.unlink(missing_ok=True)
    source.unlink()
    checkpoint = read_json(run_dir / "resume.json")
    if checkpoint and "files" in checkpoint:
        checkpoint["files"].pop("calls.jsonl", None)
        checkpoint["files"]["calls.jsonl.gz"] = target.stat().st_size
        write_json(run_dir / "resume.json", checkpoint)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(data)
    os.replace(handle.name, path)
