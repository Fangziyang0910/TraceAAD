"""Launch and watch the 20-run V10.13 batch on three endpoint pools.

The batch has four repeats for each of the five tasks. The fixed allocation is
server1=5, server3:8000=8, and server3:8001=7. All 20 runs can start at
once and remain below the endpoint capacities.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import fcntl
import json
from pathlib import Path
import re
import subprocess
import time

from experiments.infra.base import BACKEND_CAPACITY, BACKENDS, TASKS, TASK_SHORT, free_slots
from experiments.infra.launcher import check_backends, get_summary_status

ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = Path(__file__).resolve().parent / "results"
BACKEND_POOL = ("server1", "server3", "server3b")
TARGET_DISTRIBUTION = {"server1": 5, "server3": 8, "server3b": 7}
REPEATS = 4
BACKEND_BY_REPEAT_TASK = {
    1: {"tsp_construct": "server1", "cvrp_aco": "server3",
        "op_aco": "server3b", "online_bin_packing": "server1",
        "vrptw_construct": "server3"},
    2: {"tsp_construct": "server3", "cvrp_aco": "server3b",
        "op_aco": "server1", "online_bin_packing": "server3",
        "vrptw_construct": "server3b"},
    3: {"tsp_construct": "server3b", "cvrp_aco": "server1",
        "op_aco": "server3", "online_bin_packing": "server3b",
        "vrptw_construct": "server3"},
    4: {"tsp_construct": "server1", "cvrp_aco": "server3b",
        "op_aco": "server3", "online_bin_packing": "server3b",
        "vrptw_construct": "server3"},
}


def _timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def session_alive(session: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", f"={session}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    ).returncode == 0


def build_plan(batch: str, session_prefix: str, repeats: int = REPEATS):
    if repeats != REPEATS:
        raise ValueError("V10.13 formal batch requires exactly four repeats")
    rows = []
    for repeat in range(1, repeats + 1):
        for task in TASKS:
            backend = BACKEND_BY_REPEAT_TASK[repeat][task]
            short = TASK_SHORT[task]
            rows.append({
                "task": task,
                "repeat": repeat,
                "seed": repeat - 1,
                "backend": backend,
                "run_name": f"{batch}_{short}_v1013_rep{repeat}",
                "session": f"{session_prefix}_{short}_r{repeat}",
                "attempts": 0,
                "status": "queued",
                "started_at": None,
                "finished_at": None,
                "last_error": None,
            })
    return rows


def validate_plan(plan):
    expected = {(task, repeat) for repeat in range(1, REPEATS + 1) for task in TASKS}
    actual = {(row["task"], int(row["repeat"])) for row in plan}
    if len(plan) != 20 or actual != expected:
        raise ValueError("V10.13 plan must contain exactly 5 tasks x 4 repeats")
    counts = Counter(row["backend"] for row in plan)
    if counts != Counter(TARGET_DISTRIBUTION):
        raise ValueError(f"V10.13 plan distribution mismatch: {counts}")
    for row in plan:
        if row["backend"] not in BACKEND_POOL:
            raise ValueError(f"unsupported V10.13 backend: {row['backend']}")
        if int(row["repeat"]) not in range(1, REPEATS + 1):
            raise ValueError(f"invalid repeat: {row['repeat']}")


def run_dir(row):
    return RESULTS_ROOT / row["task"] / row["run_name"]


def refresh(plan):
    for row in plan:
        status = get_summary_status(run_dir(row))
        if status == "finished":
            row["status"] = "finished"
            row["finished_at"] = row.get("finished_at") or _timestamp()
        elif status in {"error", "interrupted", "aborted", "uncertain_evaluation"}:
            row["status"] = "blocked"
            row["last_error"] = status
        elif session_alive(row["session"]):
            row["status"] = "running"
        elif row["status"] == "running":
            row["status"] = "queued"
            row["last_error"] = "session ended before a finished summary"


def healthy_capacity():
    available = free_slots()
    for backend in BACKEND_POOL:
        if available.get(backend, 0) <= 0:
            continue
        try:
            check_backends([backend])
        except Exception as exc:
            print(f"V10.13: {backend} unavailable: {exc}", flush=True)
            available[backend] = 0
    return {backend: min(available.get(backend, 0), BACKEND_CAPACITY[backend])
            for backend in BACKEND_POOL}


def launch_row(row):
    if run_dir(row).exists():
        raise FileExistsError(f"run directory already exists: {run_dir(row)}")
    if session_alive(row["session"]):
        raise RuntimeError(f"tmux session already exists: {row['session']}")
    command = [
        "uv", "run", "python", "-m", "experiments.traceaad_v10_13.run",
        "--task", row["task"], "--backend", row["backend"],
        "--repeat", str(row["repeat"]), "--seed", str(row["seed"]),
        "--run-name", row["run_name"],
    ]
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", row["session"],
         "-c", str(ROOT), *command],
        cwd=ROOT, check=True,
    )


def write_manifest(path, payload):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def start_available(plan, manifest, max_attempts):
    capacity = healthy_capacity()
    for row in plan:
        if row["status"] != "queued" or capacity.get(row["backend"], 0) <= 0:
            continue
        if row["attempts"] >= max_attempts:
            row["status"] = "blocked"
            row["last_error"] = "maximum launch attempts exceeded"
            continue
        row["attempts"] += 1
        row["status"] = "launching"
        row["started_at"] = row["started_at"] or _timestamp()
        write_manifest(manifest, payload={**_manifest_payload(manifest), "plan": plan})
        try:
            launch_row(row)
        except Exception as exc:
            row["status"] = "queued"
            row["last_error"] = str(exc)
            print(f"V10.13: launch failed for {row['run_name']}: {exc}", flush=True)
        else:
            row["status"] = "running"
            capacity[row["backend"]] -= 1
            print(f"V10.13: started {row['run_name']} on {row['backend']}", flush=True)
        write_manifest(manifest, payload={**_manifest_payload(manifest), "plan": plan})
    return plan


def _manifest_payload(manifest):
    return json.loads(manifest.read_text()) if manifest.exists() else {}


def run_batch(args):
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = RESULTS_ROOT / f"batch_{args.batch}.json"
    lock_path = manifest.with_suffix(".lock")
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if manifest.exists():
            payload = json.loads(manifest.read_text())
            if payload.get("method") != "v1013_three_pool":
                raise ValueError("existing V10.13 batch has a different method")
            plan = payload["plan"]
        else:
            plan = build_plan(args.batch, args.session_prefix)
            validate_plan(plan)
            for row in plan:
                if run_dir(row).exists() or session_alive(row["session"]):
                    raise FileExistsError(f"existing V10.13 artifact: {row['run_name']}")
            payload = {
                "method": "v1013_three_pool",
                "batch": args.batch,
                "session_prefix": args.session_prefix,
                "created_at": _timestamp(),
                "updated_at": _timestamp(),
                "repeats": REPEATS,
                "backends": list(BACKEND_POOL),
                "target_distribution": TARGET_DISTRIBUTION,
                "plan": plan,
            }
            write_manifest(manifest, payload)

        validate_plan(plan)
        while True:
            refresh(plan)
            start_available(plan, manifest, args.max_attempts)
            refresh(plan)
            payload = _manifest_payload(manifest)
            payload.update({"updated_at": _timestamp(), "plan": plan})
            write_manifest(manifest, payload)
            counts = Counter(row["status"] for row in plan)
            print(f"V10.13 {payload['updated_at']} {dict(counts)}", flush=True)
            if not args.watch or all(row["status"] in {"finished", "blocked"} for row in plan):
                return payload
            time.sleep(args.interval)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--session-prefix", default="v1013")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.interval < 1 or args.max_attempts < 1:
        parser.error("interval and max-attempts must be positive")
    if not all(re.fullmatch(r"[A-Za-z0-9_-]+", value)
               for value in (args.batch, args.session_prefix)):
        parser.error("batch and session-prefix must contain only letters, numbers, _ or -")
    if args.dry_run:
        plan = build_plan(args.batch, args.session_prefix)
        validate_plan(plan)
        print(json.dumps({"plan": plan, "distribution": TARGET_DISTRIBUTION}, indent=2))
        return
    payload = run_batch(args)
    print(json.dumps({"batch": payload["batch"], "manifest": str(RESULTS_ROOT / f"batch_{args.batch}.json"),
                      "status_counts": dict(Counter(row["status"] for row in payload["plan"]))}, indent=2))


if __name__ == "__main__":
    main()
