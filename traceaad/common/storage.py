"""Small file helpers shared by experiment writers and readers."""

import json
import os
from pathlib import Path
import tempfile
import gzip
import hashlib
import shutil
from functools import lru_cache

JOURNAL_NAME = "events.jsonl"
RESULT_FORMAT = "traceaad-results-v2"


def normalize_live_record(value, task):
    """Read a legacy writer's snapshot as minimized objectives without rewriting it.

    A writer already in memory must keep its byte offsets. Its three CVRP runs
    are explicitly marked until completion; distances/counts are nonnegative,
    so this also handles the prefix converted before the writer was detected.
    """
    if isinstance(value, list):
        return [normalize_live_record(v, task) for v in value]
    if not isinstance(value, dict):
        return value
    direction = -1 if task == "op_aco" or str(task).startswith("cob_") else 1

    def metric(item):
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            return direction * abs(item)
        if isinstance(item, list):
            return [metric(v) for v in item]
        if isinstance(item, dict):
            return {k: metric(v) for k, v in item.items()}
        return item

    return {k: metric(v) if k in {"fitness", "score", "scores", "selection_fitness",
            "first_score", "start_score", "best_score", "search_best_score"}
            else normalize_live_record(v, task) for k, v in value.items()}


def live_snapshot(run_dir):
    path = Path(run_dir) / ".minimize/live.json"
    return json.loads(path.read_text()) if path.exists() else None


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    value = json.loads(path.read_text(encoding="utf-8"))
    live = live_snapshot(path.parent) if path.name in {"summary.json", "resume.json", "selection.json", "heldout.json"} else None
    return normalize_live_record(value, live["task"]) if live else value


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
    live = live_snapshot(run_dir)
    for row in rows(Path(run_dir) / JOURNAL_NAME, committed_size(run_dir)):
        yield normalize_live_record(row, live["task"]) if live else row


def selected_program(run_dir):
    run_dir = Path(run_dir)
    names = ("run_config.json", "summary.json", "selection.json", "best_program.py", "programs.jsonl")
    signature = tuple((p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ino)
                      if p.exists() else None for p in (run_dir / name for name in names))
    best = _selected_program(run_dir, signature)
    return dict(best) if best else None


@lru_cache(maxsize=1024)
def _selected_program(run_dir, signature):
    summary = read_json(run_dir / "summary.json", {})
    best = summary.get("best")
    if summary.get("status") != "finished" or not best:
        return None
    exported = run_dir / "best_program.py"
    code = exported.read_text(encoding="utf-8") if exported.exists() else Programs(run_dir).get(best["key"])
    selection = read_json(run_dir / "selection.json", {})
    identity_matches = not selection or (selection["selected_node"], selection["selected_key"]) == (best["id"], best["key"])
    return {**best, "code": code, "task": read_json(run_dir / "run_config.json", {}).get("task"),
            "verified": bool(code and identity_matches and hashlib.sha256(code.encode()).hexdigest() == best["key"])}


def heldout_identity(run_dir, result):
    best = selected_program(run_dir)
    task = best["task"] if best else read_json(Path(run_dir) / "run_config.json", {}).get("task")
    if task and task != result["task"]:
        return "task_mismatch"
    if result["verification"] != "verified":
        return result["verification"]
    if not best or not best["verified"]:
        return "program_unverified"
    return ("verified" if (result.get("key"), result.get("node_id")) == (best["key"], best["id"])
            else "program_mismatch")


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
