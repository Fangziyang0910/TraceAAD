"""Start a V11.1 batch from an explicit per-run LLM assignment table.

Edit the JSON assignment file before launching.  This script deliberately has
no queue, capacity accounting, CPU policy, retries, or backend health checks.
It only records the chosen backend and starts one tmux session per run.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

from experiments.infra.base import BACKENDS, TASKS, TASK_SHORT


ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = Path(__file__).resolve().parent / "results"
DEFAULT_ASSIGNMENTS = Path(__file__).with_name("manual_assignments.json")


def build_assignments(batch: str, repeats: int = 5) -> list[dict[str, object]]:
    backends = ("server1", "server3", "server3b", "local")
    rows: list[dict[str, object]] = []
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
