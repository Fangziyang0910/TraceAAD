"""Start a V11.1 batch from an explicit per-run LLM assignment table.

The launcher validates endpoint capacities before creating any tmux sessions.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

from experiments.infra.base import (
    BACKENDS, BACKEND_CAPACITY, TASKS, TASK_SHORT,
)


ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = Path(__file__).resolve().parent / "results"
DEFAULT_ASSIGNMENTS = Path(__file__).with_name("manual_assignments.json")


def build_assignments(batch: str, repeats: int = 5) -> list[dict[str, object]]:
    # Fill a 25-run batch without exceeding any endpoint.  The two server3
    # labels are independent pools and may each receive nine runs.
    remaining = dict(BACKEND_CAPACITY)
    order = tuple(BACKEND_CAPACITY)
    backends: list[str] = []
    rows: list[dict[str, object]] = []
    for _ in range(repeats * len(TASKS)):
        available = [name for name in order if remaining[name] > 0]
        if not available:
            raise ValueError("assignment template exceeds endpoint capacity")
        backend = max(
            available,
            key=lambda name: (remaining[name] / BACKEND_CAPACITY[name], -order.index(name)),
        )
        backends.append(backend)
        remaining[backend] -= 1
    index = 0
    for repeat in range(1, repeats + 1):
        for task in TASKS:
            backend = backends[index % len(backends)]
            rows.append({
                "task": task,
                "repeat": repeat,
                "seed": repeat - 1,
                "backend": backend,
                "run_name": f"{batch}_{TASK_SHORT[task]}_v111_rep{repeat}",
            })
            index += 1
    return rows


def _session(prefix: str, row: dict[str, object]) -> str:
    return f"{prefix}_{TASK_SHORT[str(row['task'])]}_r{int(row['repeat'])}"


def _validate(rows: list[dict[str, object]], batch: str) -> None:
    expected = {(task, repeat) for repeat in range(1, 6) for task in TASKS}
    got = {(str(row.get("task")), int(row.get("repeat", 0))) for row in rows}
    if got != expected or len(rows) != 25:
        raise ValueError("assignments must contain exactly one row for each of 5 tasks x 5 repeats")
    for row in rows:
        if row.get("backend") not in BACKENDS:
            raise ValueError(f"invalid backend in assignment: {row.get('backend')}")
        task = str(row["task"])
        repeat = int(row["repeat"])
        row["seed"] = int(row.get("seed", repeat - 1))
        row["run_name"] = str(row.get("run_name") or f"{batch}_{TASK_SHORT[task]}_v111_rep{repeat}")
    counts = {name: 0 for name in BACKEND_CAPACITY}
    for row in rows:
        counts[str(row["backend"])] += 1
    over = {
        name: (count, BACKEND_CAPACITY[name])
        for name, count in counts.items()
        if count > BACKEND_CAPACITY[name]
    }
    if over:
        raise ValueError(f"assignment exceeds endpoint capacity: {over}")


def launch(batch: str, assignments_path: Path, prefix: str, delay: float) -> dict[str, object]:
    rows = json.loads(assignments_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("assignment file must contain a JSON list")
    rows = [dict(row) for row in rows]
    _validate(rows, batch)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = RESULTS_ROOT / f"batch_{batch}.json"
    if manifest_path.exists():
        raise FileExistsError(f"batch manifest already exists: {manifest_path}")

    sessions = [_session(prefix, row) for row in rows]
    for row, session in zip(rows, sessions):
        run_dir = RESULTS_ROOT / str(row["task"]) / str(row["run_name"])
        if run_dir.exists():
            raise FileExistsError(f"run directory already exists: {run_dir}")
        probe = subprocess.run(
            ["tmux", "has-session", "-t", f"={session}"],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if probe.returncode == 0:
            raise RuntimeError(f"tmux session already exists: {session}")

    plan = [
        {
            **row,
            "session": session,
            "status": "queued",
            "started_at": None,
        }
        for row, session in zip(rows, sessions)
    ]
    manifest = {
        "method": "v111_manual",
        "batch": batch,
        "session_prefix": prefix,
        "created_at": datetime.now().astimezone().isoformat(),
        "repeats": 5,
        "backends": list(BACKENDS),
        "assignment_file": str(assignments_path.resolve()),
        "plan": plan,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for item in plan:
        command = [
            sys.executable, "-m", "experiments.traceaad_v11_1.run",
            "--task", str(item["task"]), "--backend", str(item["backend"]),
            "--repeat", str(item["repeat"]), "--seed", str(item["seed"]),
            "--run-name", str(item["run_name"]),
        ]
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", str(item["session"]), *command],
            cwd=ROOT, check=True,
        )
        item["status"] = "running"
        item["started_at"] = datetime.now().astimezone().isoformat()
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if delay:
            time.sleep(delay)
    return {"batch": batch, "manifest": str(manifest_path), "runs": len(plan), "prefix": prefix}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--session-prefix", default="v111_manual")
    parser.add_argument("--delay", type=float, default=0.0, help="seconds between tmux launches")
    parser.add_argument("--write-template", action="store_true")
    args = parser.parse_args()
    if args.write_template:
        args.assignments.write_text(
            json.dumps(build_assignments(args.batch), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(args.assignments)
        return
    if args.delay < 0:
        parser.error("--delay must be nonnegative")
    print(json.dumps(launch(args.batch, args.assignments, args.session_prefix, args.delay), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
