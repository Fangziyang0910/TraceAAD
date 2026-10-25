"""Queue a baseline batch on this host: at most N runs at once, the next when one ends.

Each run holds one slot of the model backend (N = the local server's slots) and
evaluates one instance at a time on a core granted by the host CPU scheduler,
the same pool the TraceAAD runs use.

    uv run python -m experiments.infra.local_queue --method funsearch --suite co6 --batch <批次> --slots 3
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime

from benchmarks.tasks import ALL_TASKS, SUITES, TASK_SHORT
from experiments.infra.base import (
    BACKENDS,
    DEFAULT_BACKEND,
    RESULTS_ROOT,
)
from experiments.infra.launcher import LaunchItem, pending_items, validate_new_items, is_session_alive, launch_command

METHODS = ("eoh", "funsearch", "reevo", "mcts_ahd", "pathwise", "calm", "shinka_evo", "deltaevolve",
           "openevolve", "hsevo", "refineevo", "clade_ahd")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--suite", choices=tuple(SUITES), default="co6")
    parser.add_argument("--tasks", nargs="+", choices=ALL_TASKS)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--slots", type=int, default=3)
    parser.add_argument("--scheduler-socket", default="/tmp/traceaad-1000/scheduler.sock")
    parser.add_argument("--backend", choices=tuple(BACKENDS), default=DEFAULT_BACKEND)
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_plan(args: argparse.Namespace) -> list[LaunchItem]:
    if args.repeats < 1 or len(set(args.tasks or SUITES[args.suite])) != len(args.tasks or SUITES[args.suite]):
        raise ValueError("repeats must be positive and tasks must be distinct")
    plan = []
    for repeat in range(1, args.repeats + 1):
        for task in args.tasks or SUITES[args.suite]:
            short = TASK_SHORT[task]
            name = f"{args.batch}_{short}_{args.method}_rep{repeat}"
            plan.append(
                LaunchItem(
                    task=task,
                    repeat=repeat,
                    backend=args.backend,
                    session=f"{args.batch}_{args.method}_{short}_r{repeat}",
                    run_name=name,
                    run_dir=RESULTS_ROOT / args.method / task / name,
                    seed=repeat - 1,
                    module=f"experiments.{args.method}.run",
                )
            )
    return plan


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.slots < 1 or args.interval < 1:
        raise ValueError("slots and interval must be positive")
    slots = range(args.slots)
    plan = build_plan(args)
    if args.dry_run:
        for item in plan:
            print(f"{item.session} {' '.join(item.command())}")
        return
    # Queue state is in memory; existing runs need explicit recovery.
    running = {}
    started = set()
    while True:
        for slot, item in list(running.items()):
            if not is_session_alive(item.session):
                print(f"[{datetime.now():%m-%d %H:%M:%S}] ended {item.run_name}", flush=True)
                del running[slot]
        pending = [item for item in pending_items(plan) if item.session not in started]
        for slot in (s for s in slots if s not in running):
            if not pending:
                break
            item = pending.pop(0)
            validate_new_items([item])
            launch_command(item.session, ("env", f"TRACEAAD_SCHEDULER_SOCKET={args.scheduler_socket}", *item.command()))
            running[slot] = item
            started.add(item.session)
            print(f"[{datetime.now():%m-%d %H:%M:%S}] started {item.run_name}", flush=True)
        if not running:
            print(f"[{datetime.now():%m-%d %H:%M:%S}] queue empty", flush=True)
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
