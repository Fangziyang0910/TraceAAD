"""Launch comparable baseline runs with the shared backend scheduler."""

from __future__ import annotations

import argparse
from dataclasses import replace

from experiments.infra.base import (
    TASKS,
    add_launch_parser_args,
    build_launch_plan,
    fill_once,
    free_slots,
    watch_and_fill,
)

METHODS = ("eoh", "reevo", "mcts_ahd", "pathwise", "calm", "shinka_evo")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS)
    parser.add_argument(
        "--run-arg", action="append", default=[],
        help="pass one additional argument to every method runner",
    )
    add_launch_parser_args(parser)
    return parser


def build_plan(args: argparse.Namespace):
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    if args.session_prefix == "batch":
        args.session_prefix = args.method
    plan = build_launch_plan(
        args, module=f"experiments.{args.method}.run", method=args.method
    )
    if args.tasks:
        plan = [item for item in plan if item.task in args.tasks]
    if args.run_arg:
        plan = [replace(item, extra_args=tuple(args.run_arg)) for item in plan]
    return plan


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    plan = build_plan(args)
    print(f"batch={args.batch} total={len(plan)} free={free_slots()}", flush=True)
    if args.watch:
        watch_and_fill(plan, interval_sec=args.watch_interval, dry_run=args.dry_run,
                       method_label=args.method)
    else:
        fill_once(plan, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
