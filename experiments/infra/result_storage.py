"""Experiment result readers, with sources indexed by program ID."""

import json
from pathlib import Path
from functools import lru_cache
from traceaad.storage import read_json, write_json, append_jsonl


def selected_program(run_dir):
    run_dir = Path(run_dir)
    names = (
        "run_config.json",
        "summary.json",
        "selection.json",
        "best_program.py",
        "programs.jsonl",
    )
    signature = tuple(
        (p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ino) if p.exists() else None
        for p in (run_dir / name for name in names)
    )
    best = _selected_program(run_dir, signature)
    return dict(best) if best else None


@lru_cache(maxsize=1024)
def _selected_program(run_dir, signature):
    summary = read_json(run_dir / "summary.json", {})
    best = summary.get("best")
    status = (
        summary.get("archived_from_status")
        if summary.get("status") == "archived"
        else summary.get("status")
    )
    if status != "finished" or not best:
        return None
    exported = run_dir / "best_program.py"
    code = (
        exported.read_text(encoding="utf-8")
        if exported.exists()
        else Programs(run_dir).get(best["id"])
    )
    selection = read_json(run_dir / "selection.json", {})
    identity_matches = not selection or selection["selected_node"] == best["id"]
    verified = bool(code and identity_matches)
    return {
        **best,
        "code": code,
        "task": read_json(run_dir / "run_config.json", {}).get("task"),
        "verified": verified,
    }


def heldout_identity(run_dir, result):
    best = selected_program(run_dir)
    task = best["task"] if best else read_json(Path(run_dir) / "run_config.json", {}).get("task")
    if task and task != result["task"]:
        return "task_mismatch"
    if result["verification"] != "verified":
        return result["verification"]
    if not best or not best["verified"]:
        return "program_unverified"
    return "verified" if result.get("node_id") == best["id"] else "program_mismatch"


class Programs:
    """Read source records by program ID using file offsets."""

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
                self.offsets[row["id"]] = start
                self.offset = handle.tell()

    def add(self, program_id, code):
        self.refresh()
        self.offsets[program_id] = self.offset
        self.offset = append_jsonl(self.path, {"id": program_id, "code": code})
        stat = self.path.stat()
        self.identity = (stat.st_dev, stat.st_ino)

    def get(self, program_id):
        self.refresh()
        offset = self.offsets.get(program_id)
        if offset is None:
            return None
        with self.path.open("rb") as handle:
            handle.seek(offset)
            return json.loads(handle.readline())["code"]


def stored_heldout(run_dir, scale, variant=None):
    matches = [
        r
        for r in read_json(Path(run_dir) / "heldout.json", [])
        if str(r["scale"]) == str(scale) and (variant is None or r["variant"] == variant)
    ]
    return min(matches, key=lambda r: bool(r["variant"])) if matches else None


def save_heldout(run_dir, result, variant=""):
    path = Path(run_dir) / "heldout.json"
    records = read_json(path, [])
    result = {**result, "variant": variant, "scale": str(result["scale"])}
    records = [r for r in records if (r["variant"], r["scale"]) != (variant, result["scale"])]
    records.append(result)
    write_json(path, records)
