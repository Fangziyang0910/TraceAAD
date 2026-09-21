"""Start a V10.12 batch from an explicit per-run LLM assignment table.

The launcher validates endpoint capacities and task/repeat counts before creating tmux sessions.
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
from experiments.traceaad_v10_12.freeze import freeze


ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = Path(__file__).resolve().parent / "results"
DEFAULT_ASSIGNMENTS = Path(__file__).with_name("manual_assignments.json")


def _session(prefix: str, row: dict[str, object]) -> str:
    return f"{prefix}_{TASK_SHORT[str(row['task'])]}_r{int(row['repeat'])}"


def _validate(rows: list[dict[str, object]], batch: str, expected_repeats: int = 4) -> None:
    expected = {(task, repeat) for repeat in range(1, expected_repeats + 1) for task in TASKS}
    got = {(str(row.get("task")), int(row.get("repeat", 0))) for row in rows}
    if got != expected or len(rows) != len(expected):
        raise ValueError(
            f"assignments must contain exactly one row for each of 5 tasks x {expected_repeats} repeats "
            f"(expected {len(expected)} rows, got {len(rows)})"
        )
    for row in rows:
        if row.get("backend") not in BACKENDS:
            raise ValueError(f"invalid backend in assignment: {row.get('backend')}")
        task = str(row["task"])
        repeat = int(row["repeat"])
        row["seed"] = int(row.get("seed", repeat - 1))
        row["run_name"] = str(row.get("run_name") or f"{batch}_{TASK_SHORT[task]}_v1012_rep{repeat}")
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


def launch(batch: str, assignments_path: Path, prefix: str, delay: float, do_freeze: bool = True) -> dict[str, object]:
    rows = json.loads(assignments_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("assignment file must contain a JSON list")
    rows = [dict(row) for row in rows]
    _validate(rows, batch, expected_repeats=4)
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

    if do_freeze:
        runtime_dir = RESULTS_ROOT / f"runtime_{batch}"
        if not runtime_dir.exists():
            freeze(batch, prefix)

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
        "method": "v1012",
        "batch": batch,
        "session_prefix": prefix,
        "created_at": datetime.now().astimezone().isoformat(),
        "repeats": 4,
        "n_profile_cards": 2,
        "profile_card_tau": 8.0,
        "backends": list(BACKENDS),
        "assignment_file": str(assignments_path.resolve()),
        "plan": plan,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for item in plan:
        command = [
            sys.executable, "-m", "experiments.traceaad_v10_12.run",
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
    parser.add_argument("--session-prefix", default="v1012")
    parser.add_argument("--delay", type=float, default=0.2, help="seconds between tmux launches")
    parser.add_argument("--no-freeze", action="store_true", help="skip freezing runtime")
    args = parser.parse_args()
    if args.delay < 0:
        parser.error("--delay must be nonnegative")
    result = launch(args.batch, args.assignments, args.session_prefix, args.delay, do_freeze=not args.no_freeze)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

