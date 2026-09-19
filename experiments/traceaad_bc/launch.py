"""Queue one B or C arm across the five frozen tasks."""

import argparse
from collections import Counter
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import re
import time

from experiments.infra.base import (
    BACKEND_DISPLAY_NAMES,
    BACKEND_CAPACITY,
    BACKENDS,
    PRIMARY_BACKENDS,
    LaunchItem,
    TASKS,
    TASK_SHORT,
    free_slots,
    item_is_running,
    launch_items,
)
from experiments.infra.equivalent_backends import prepare_resume
from experiments.infra.launcher import check_backends, get_summary_status
from llm4ad.method.traceaad_v11_0.core import atomic_json

RESULTS_ROOT = Path(__file__).resolve().parent / "results"
BACKEND_NAMES = tuple(PRIMARY_BACKENDS)
ARMS = ("B", "C")


def allocate(plan, available, backend_pool, *, defer_cvrp=False):
    remaining, assignments = dict(available), []
    used_by_task = {
        task: {
            row["backend"] for row in plan
            if row["task"] == task and row["backend"]
        }
        for task in TASKS
    }
    for row in plan:
        if row["status"] != "queued" or (defer_cvrp and row["task"] == "cvrp_aco"):
            continue
        candidates = [backend for backend in backend_pool if remaining.get(backend, 0) > 0]
        if not candidates:
            continue
        used = used_by_task[row["task"]]
        candidates.sort(key=lambda backend: (
            backend in used,
            (BACKEND_CAPACITY[backend] - remaining[backend] + 1) / BACKEND_CAPACITY[backend],
            BACKEND_NAMES.index(backend),
        ))
        backend = candidates[0]
        remaining[backend] -= 1
        used.add(backend)
        assignments.append((row, backend))
    return assignments


def healthy_slots(available, backend_pool):
    available = dict(available)
    for backend in backend_pool:
        if available.get(backend, 0) > 0:
            try:
                check_backends([backend])
            except Exception:
                available[backend] = 0
    return available


def build_plan(batch, prefix, arm, *, repeats=3, cvrp_last=False, n_references=8):
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    order = [(repeat, task) for repeat in range(1, repeats + 1) for task in TASKS]
    if cvrp_last:
        order.sort(key=lambda item: item[1] == "cvrp_aco")
    return [
        dict(
            task=task,
            repeat=repeat,
            seed=repeat - 1,
            backend=None,
            backend_label=None,
            arm=arm,
            n_references=n_references,
            run_name=f"{batch}_{TASK_SHORT[task]}_bc{arm.lower()}_rep{repeat}",
            session=f"{prefix}_{TASK_SHORT[task]}_r{repeat}",
            attempts=0,
            status="queued",
        )
        for repeat, task in order
    ]


def cvrp_group_finished(batch):
    manifest = RESULTS_ROOT / f"batch_{batch}.json"
    if not manifest.exists():
        return False
    rows = [
        row for row in json.loads(manifest.read_text())["plan"]
        if row["task"] == "cvrp_aco"
    ]
    return bool(rows) and all(
        get_summary_status(launch_item(row).run_dir) == "finished" for row in rows
    )


def launch_item(row):
    return LaunchItem(
        task=row["task"],
        repeat=row["repeat"],
        seed=row["seed"],
        backend=row["backend"],
        session=row["session"],
        run_name=row["run_name"],
        run_dir=RESULTS_ROOT / row["task"] / row["run_name"],
        module="experiments.traceaad_bc.run",
        extra_args=(
            "--arm", row["arm"],
            "--n-references", str(row.get("n_references", 8)),
        ),
    )


def refresh(plan, max_attempts):
    for row in plan:
        item = launch_item(row)
        status = get_summary_status(item.run_dir)
        if item_is_running(item):
            row["status"] = "running"
        elif status in ("finished", "blocked"):
            row["status"] = status
        elif row["attempts"] >= max_attempts:
            row["status"] = "stopped"
        else:
            row["status"] = "queued"


def verify_runtime():
    root = Path(__file__).resolve().parents[2]
    manifest = root / "runtime_manifest.json"
    if not manifest.exists():
        raise ValueError("freeze the reviewed source first using experiments.traceaad_bc.freeze")
    payload = json.loads(manifest.read_text())
    for relative, expected in payload["files"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"frozen source changed: {relative}")
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--session-prefix", default="bc")
    parser.add_argument("--repeats", type=int, choices=(1, 2, 3, 4, 5), default=3)
    parser.add_argument("--n-references", type=int, default=8)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backends", default=",".join(BACKEND_NAMES))
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--cvrp-last", action="store_true")
    parser.add_argument("--cvrp-barrier-batches", default="")
    args = parser.parse_args(argv)
    if args.interval < 1 or args.max_attempts < 1:
        parser.error("interval and max-attempts must be positive")
    if args.n_references < 1:
        parser.error("n-references must be positive")
    if not all(re.fullmatch(r"[A-Za-z0-9_-]+", value)
               for value in (args.batch, args.session_prefix)):
        parser.error("batch and session-prefix must contain only letters, numbers, underscore or hyphen")
    cvrp_barriers = tuple(filter(None, args.cvrp_barrier_batches.split(",")))
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", batch) for batch in cvrp_barriers):
        parser.error("CVRP barrier batch names must contain only letters, numbers, underscore or hyphen")
    backend_pool = tuple(dict.fromkeys(args.backends.split(",")))
    if not backend_pool or any(backend not in BACKENDS for backend in backend_pool):
        parser.error(f"backends must be drawn from {', '.join(BACKEND_NAMES)}")

    identity = None if args.dry_run else verify_runtime()
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = RESULTS_ROOT / f"batch_{args.batch}.json"
    with manifest.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if manifest.exists():
            payload = json.loads(manifest.read_text())
            expected = (f"bc_{args.arm.lower()}", args.session_prefix, args.arm,
                        args.repeats, args.n_references, backend_pool,
                        args.cvrp_last, cvrp_barriers)
            actual = (payload.get("method"), payload.get("session_prefix"),
                      payload.get("arm"), payload.get("repeats", 3),
                      payload.get("n_references", 8), tuple(payload.get("backends", ())),
                      bool(payload.get("cvrp_last")), tuple(payload.get("cvrp_barriers", ())))
            if actual != expected:
                raise ValueError("batch identity or scheduling configuration mismatch")
            if not args.dry_run and payload["source_identity"] != identity:
                raise ValueError("batch frozen source mismatch")
        else:
            payload = dict(
                method=f"bc_{args.arm.lower()}",
                batch=args.batch,
                arm=args.arm,
                session_prefix=args.session_prefix,
                source_identity=identity,
                created_at=datetime.now().astimezone().isoformat(),
                repeats=args.repeats,
                n_references=args.n_references,
                cvrp_last=args.cvrp_last,
                cvrp_barriers=cvrp_barriers,
                backends=backend_pool,
                direct=args.direct,
                plan=build_plan(
                    args.batch,
                    args.session_prefix,
                    args.arm,
                    repeats=args.repeats,
                    cvrp_last=args.cvrp_last,
                    n_references=args.n_references,
                ),
            )
            if any(item_is_running(launch_item(row)) or launch_item(row).run_dir.exists()
                   for row in payload["plan"]):
                raise ValueError("existing session or run directory without matching batch manifest")

        last = None
        while True:
            plan = payload["plan"]
            refresh(plan, args.max_attempts)
            available = free_slots()
            if not args.dry_run:
                available = healthy_slots(available, backend_pool)
            defer_cvrp = bool(cvrp_barriers) and not any(
                cvrp_group_finished(batch) for batch in cvrp_barriers
            )
            assignments = allocate(plan, available, backend_pool, defer_cvrp=defer_cvrp)
            if not args.direct:
                assignments = assignments[:1]
            if args.dry_run:
                print(json.dumps({"plan": plan, "free": available,
                                  "next": [row for row, _ in assignments]}, indent=2))
                return
            for row, backend in assignments:
                if free_slots().get(backend, 0) <= 0:
                    continue
                row.update(
                    backend=backend,
                    backend_label=BACKEND_DISPLAY_NAMES[backend],
                    attempts=row["attempts"] + 1,
                    status="launching",
                )
                atomic_json(manifest, payload)
                try:
                    prepare_resume(launch_item(row), BACKENDS)
                    launch_items([launch_item(row)], dry_run=False)
                except Exception as exc:
                    row.update(status="queued", last_launch_error=str(exc))
                else:
                    row["status"] = "running"
            payload["updated_at"] = datetime.now().astimezone().isoformat()
            atomic_json(manifest, payload)
            counts = dict(Counter(row["status"] for row in plan))
            if counts != last:
                print(f'{payload["updated_at"]} {counts} manifest={manifest}', flush=True)
                last = counts
            if not args.watch or all(row["status"] in ("finished", "blocked", "stopped")
                                     for row in plan):
                return
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
