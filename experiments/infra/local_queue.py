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

from benchmarks.tasks import SUITES
from experiments.infra.base import ALL_TASKS, BACKENDS, pending_items, validate_new_items
from experiments.infra.launcher import is_session_alive, launch_command
from experiments.launch import METHODS, build_plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--suite", choices=tuple(SUITES), default="co6")
    parser.add_argument("--tasks", nargs="+", choices=ALL_TASKS)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--slots", type=int, default=3)
    parser.add_argument("--scheduler-socket", default="/tmp/traceaad-1000/scheduler.sock")
    parser.add_argument("--backend", choices=tuple(BACKENDS), default="local")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.slots < 1:
        raise ValueError("--slots must be positive")
    slots = range(args.slots)
    args.session_prefix, args.run_arg = f"{args.batch}_{args.method}", []
    plan = build_plan(args)
    if args.dry_run:
        for item in plan:
            print(f"{item.session} {' '.join(item.with_backend(args.backend).command())}")
        return
    # ponytail: queue state stays in memory; restart only after its running jobs end.
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
            item = pending.pop(0).with_backend(args.backend)
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
