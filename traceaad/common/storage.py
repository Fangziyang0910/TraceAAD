"""Small file helpers shared by experiment writers and readers."""

import json
import os
from pathlib import Path
import tempfile
import gzip
import hashlib
import shutil

JOURNAL_NAME = "events.jsonl"
RESULT_FORMAT = "traceaad-results-v1"


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


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


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()
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


class Programs:
    """One source per content hash; index offsets without retaining every source."""

    def __init__(self, run_dir):
        self.path = Path(run_dir) / "programs.jsonl"
        self.offsets, self.offset = {}, 0
        self.identity = None

    def refresh(self):
        if not self.path.exists():
            return
        stat = self.path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity != self.identity or stat.st_size < self.offset:
            self.offsets, self.offset, self.identity = {}, 0, identity
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            while raw := handle.readline():
                start = self.offset
                if not raw.endswith(b"\n"):
                    break
                row = json.loads(raw)
                self.offsets[row["key"]] = start
                self.offset = handle.tell()

    def add(self, code):
        source_key = hashlib.sha256(code.encode()).hexdigest()
        self.refresh()
        if source_key not in self.offsets:
            self.offsets[source_key] = self.offset
            self.offset = append_jsonl(self.path, {"key": source_key, "code": code})
            stat = self.path.stat()
            self.identity = (stat.st_dev, stat.st_ino)
        return source_key

    def get(self, source_key):
        self.refresh()
        offset = self.offsets.get(source_key)
        if offset is None:
            return None
        with self.path.open("rb") as handle:
            handle.seek(offset)
            return json.loads(handle.readline())["code"]


def save_heldout(run_dir, result, variant=""):
    path = Path(run_dir) / "heldout.json"
    records = read_json(path, [])
    result = {**result, "variant": variant, "scale": str(result["scale"])}
    records = [r for r in records if (r["variant"], r["scale"]) != (variant, result["scale"])]
    records.append(result)
    write_json(path, records)


def seal_calls(run_dir):
    run_dir = Path(run_dir)
    source = run_dir / "calls.jsonl"
    if not source.exists():
        return
    target = run_dir / "calls.jsonl.gz"
    destination = target if target.exists() else target.with_suffix(".tmp")
    detach_link(destination)
    mode = "ab" if destination == target else "wb"
    with source.open("rb") as original, gzip.open(destination, mode, compresslevel=1) as compressed:
        shutil.copyfileobj(original, compressed)
    if destination != target:
        destination.replace(target)
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
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
        handle.write(data)
    os.replace(handle.name, path)
