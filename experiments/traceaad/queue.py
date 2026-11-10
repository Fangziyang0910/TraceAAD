"""Persist a cross-version run queue, including already-running searches in its capacity."""

import argparse
from collections import Counter
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo

from benchmarks.tasks import CO_TASKS
from traceaad_base.storage import write_json
from .launch_host import launch_run


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def read(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def key(row):
    return row["batch"], row["run_name"]


def inventory(keys):
    """Count one route even when its evaluator children inherit the same command line."""
    active = set()
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            args = [part.decode(errors="replace") for part in (path / "cmdline").read_bytes().split(b"\0") if part]
            if "--batch" not in args or "--run-name" not in args:
                continue
            identity = args[args.index("--batch") + 1], args[args.index("--run-name") + 1]
            if identity in keys:
                active.add(identity)
        except (OSError, ValueError, IndexError):
            continue
    result = subprocess.run(["tmux", "list-panes", "-a", "-F", "#{session_name}\t#{pane_dead}"],
                            capture_output=True, text=True, timeout=10)
    if result.returncode and not any(text in result.stderr for text in ("no server running", "No such file")):
        raise RuntimeError(result.stderr.strip())
    sessions = {line.split("\t")[0] for line in result.stdout.splitlines() if line.endswith("\t0")}
    return active, sessions


def observe(row, active, sessions):
    directory = Path(row["run_dir"])
    if key(row) in active or row["session"] in sessions:
        return "running", None
    summary = read(directory / "summary.json", {})
    if summary.get("status") == "finished" and read(directory / "heldout.json", []):
        return "finished", None
    if (row["status"] in {"planned", "queued", "starting"}
            and not directory.exists() and not Path(row["startup_log"]).exists()):
        return "queued", None
    return "blocked", summary.get("error") or "Worker exited before search and held-out evaluation completed"


def step(manifests, observed, *, tasks, capacity, state_path, spacing=1):
    """Reconcile disk and live workers before starting any new route."""
    settings = {"manifests": [str(p) for p in manifests], "observed": [str(p) for p in observed],
                "tasks": list(tasks), "capacity": capacity}
    previous = read(state_path, {})
    if previous and previous["settings"] != settings:
        raise ValueError("Queue settings changed; use the original settings or a new state path")
    paths = [*manifests, *observed]
    if len(set(paths)) != len(paths):
        raise ValueError("Queue manifest paths must be distinct")
    batches = {path: read(path) for path in paths}
    managed, external = [], []
    for path, batch in batches.items():
        for row in batch["plan"]:
            if row["task"] not in tasks or row["status"] == "deferred":
                if path in manifests:
                    raise ValueError(f"Queued plan contains an excluded task: {row['run_name']}")
                continue
            item = {**row, "batch": batch["batch"], "manifest": path, "original": row}
            (managed if path in manifests else external).append(item)
    rows = [*external, *managed]
    if (len({key(r) for r in rows}) != len(rows) or len({r["session"] for r in rows}) != len(rows)
            or len({r["run_name"] for r in rows}) != len(rows)):
        raise ValueError("Queue run identities and sessions must be unique")
    active, sessions = inventory({key(r) for r in rows})
    states = {}
    for row in rows:
        status, error = observe(row, active, sessions)
        states[row["run_name"]] = {"batch": row["batch"], "task": row["task"],
                                   "status": status, "error": error}
    running = sum(s["status"] == "running" for s in states.values())
    # Start the first version first, then interleave later versions as positions open.
    groups = [[r for r in managed if r["manifest"] == path] for path in manifests]
    pending = list(groups[0]) if groups else []
    for index in range(max((len(group) for group in groups[1:]), default=0)):
        pending.extend(group[index] for group in groups[1:] if index < len(group))
    changed = set()

    def save_status(row, status, error=None):
        original = row["original"]
        if original["status"] != status or original.get("queue_error") != error:
            original["status"] = status
            if error:
                original["queue_error"] = error
            else:
                original.pop("queue_error", None)
            changed.add(row["manifest"])

    for row in managed:
        state = states[row["run_name"]]
        save_status(row, "launched" if state["status"] == "running" else state["status"], state["error"])
    for row in pending:
        if running >= capacity:
            break
        if states[row["run_name"]]["status"] != "queued":
            continue
        save_status(row, "starting")
        write_json(row["manifest"], batches[row["manifest"]])
        try:
            launch_run(row)
        except (OSError, subprocess.SubprocessError) as error:
            message = f"{type(error).__name__}: {error}"
            save_status(row, "blocked", message)
            states[row["run_name"]].update(status="blocked", error=message)
            break  # Recount before another start if tmux may have partially succeeded.
        else:
            row["original"].setdefault("started_at", now())
            save_status(row, "launched")
            states[row["run_name"]].update(status="running", error=None)
            running += 1
            if spacing:
                time.sleep(spacing)
    for path in changed:
        statuses = {row["status"] for row in batches[path]["plan"]}
        batches[path]["status"] = ("finished" if statuses == {"finished"} else
                                   "completed_with_errors" if statuses <= {"finished", "blocked"} else "queued")
        write_json(path, batches[path])
    events = state_path.with_suffix(".events.jsonl")
    events.parent.mkdir(parents=True, exist_ok=True)
    with events.open("a") as stream:
        for name, value in states.items():
            if previous.get("runs", {}).get(name, {}).get("status") != value["status"]:
                stream.write(json.dumps({"at": now(), "run_name": name, **value}) + "\n")
    counts = dict(Counter(s["status"] for s in states.values()))
    state = {"updated_at": now(), "pid": os.getpid(), "settings": settings, "counts": counts, "runs": states,
             "done": not any(s["status"] in {"running", "queued"} for s in states.values())}
    write_json(state_path, state)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", nargs="+", type=Path, required=True)
    parser.add_argument("--observe", nargs="*", type=Path, default=[])
    parser.add_argument("--tasks", nargs="+", choices=CO_TASKS, default=list(CO_TASKS))
    parser.add_argument("--capacity", type=int, default=36)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--spacing", type=float, default=1)
    args = parser.parse_args(argv)
    if args.capacity < 1 or args.interval <= 0 or args.spacing < 0 or len(set(args.tasks)) != len(args.tasks):
        parser.error("capacity and interval must be positive; tasks distinct and spacing nonnegative")
    args.state.parent.mkdir(parents=True, exist_ok=True)
    with args.state.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                state = step(args.manifests, args.observe, tasks=args.tasks, capacity=args.capacity,
                             state_path=args.state, spacing=args.spacing)
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                print(json.dumps({"at": now(), "error": f"{type(error).__name__}: {error}"}), flush=True)
            else:
                if state["done"]:
                    print(json.dumps({"at": now(), "counts": state["counts"], "done": True}), flush=True)
                    return 1 if state["counts"].get("blocked") else 0
            time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
