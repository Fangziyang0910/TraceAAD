"""Preview or launch a TraceAAD batch on this host."""

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import subprocess

from benchmarks.tasks import (
    ALL_TASKS,
    SUITES,
    TASK_SHORT,
    FUNCTION_SECONDS,
    FUNCTION_LIMITS,
    INSTANCE_SECONDS,
    INSTANCE_LIMITS,
)
from core.scheduling import scheduler_status
from traceaad import Config
from traceaad.config import EXPERIMENT
from .base import BACKENDS, DEFAULT_BACKEND, HOSTED_BACKENDS, REPO_ROOT, RESULTS_ROOT, llm_payload
from .launcher import LaunchItem, launch_plan


def plan_for(args, tasks):
    repeats = args.repeat_ids or range(1, args.repeats + 1)
    if (
        args.repeats < 1
        or args.budget < 1
        or not tasks
        or len(set(tasks)) != len(tasks)
        or len(set(repeats)) != len(repeats)
        or any(repeat < 1 or repeat > args.repeats for repeat in repeats)
    ):
        raise ValueError("require positive budget/repeats, distinct tasks and repeat IDs within range")
    plan = []
    for index, task in enumerate(tasks):
        for repeat in repeats:
            backend = args.backends[(index + repeat + 1) % len(args.backends)]
            name = f"{args.batch}_{TASK_SHORT[task]}_{args.experiment}_rep{repeat}"
            session = f"{args.batch}_{TASK_SHORT[task]}_{args.experiment.removeprefix('traceaad_')}_r{repeat}"
            extra = [f"--budget={args.budget}", f"--eval-workers={args.eval_workers}", "--experiment", args.experiment]
            if args.eval_timeout_seconds is not None:
                extra.append(f"--eval-timeout-seconds={args.eval_timeout_seconds:g}")
            if args.scheduler_socket:
                extra.extend(["--scheduler-socket", args.scheduler_socket])
            if getattr(args, 'version', None) in ('v13', 'v13_1'):
                if args.version == 'v13_1':
                    from experiments.traceaad_v13_1.options import command_arguments
                else:
                    from experiments.traceaad_v13.options import command_arguments
                extra.extend(command_arguments(args))
            item = LaunchItem(
                task,
                repeat,
                backend,
                session,
                name,
                RESULTS_ROOT / args.experiment / task / name,
                repeat - 1,
                f"experiments.traceaad_{args.version}.run" if getattr(args, "version", None) in ("v13", "v13_1") else "experiments.traceaad.run",
                tuple(extra),
            )
            plan.append(
                {
                    **{k: v for k, v in asdict(item).items() if k not in {"module", "extra_args"}},
                    "run_dir": str(item.run_dir),
                    "command": item.command(),
                    "backend": "scheduled" if args.scheduler_socket and backend not in HOSTED_BACKENDS else backend,
                    "startup_log": str(RESULTS_ROOT / args.experiment / "launch_logs" / f"{session}.log"),
                    "status": "planned",
                }
            )
    return plan


def git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv=None, *, version=None):
    config_type = Config
    if version == 'v13':
        from traceaad.v13 import Config as config_type
        from experiments.traceaad_v13 import options
    elif version == 'v13_1':
        from traceaad.v13_1 import Config as config_type
        from experiments.traceaad_v13_1 import options
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--experiment", default=f"traceaad_{version}" if version in ("v13", "v13_1") else EXPERIMENT)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--suite", choices=tuple(SUITES), default="co6")
    parser.add_argument("--tasks", choices=ALL_TASKS, nargs="+")
    parser.add_argument("--backends", choices=tuple(BACKENDS), nargs="+", default=[DEFAULT_BACKEND])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--repeat-ids", type=int, nargs="+")
    parser.add_argument("--budget", type=int, default=Config.budget)
    parser.add_argument("--eval-workers", type=int)
    parser.add_argument("--eval-timeout-seconds", type=float)
    parser.add_argument("--scheduler-socket")
    if version in ('v13', 'v13_1'):
        options.add_arguments(parser)
        parser.add_argument('--output-tokens', type=int, default=config_type.output_tokens)
    args = parser.parse_args(argv)
    args.version = version
    if not args.experiment.replace("_", "").isalnum() or not args.batch.replace("_", "").replace("-", "").isalnum():
        raise ValueError("experiment and batch must be safe directory names")
    tasks = tuple(args.tasks or SUITES[args.suite])
    scheduler = None
    if args.scheduler_socket:
        from pathlib import Path

        args.scheduler_socket = str(Path(args.scheduler_socket).expanduser().resolve())
        scheduler = scheduler_status(args.scheduler_socket)
    if args.eval_workers is None:
        args.eval_workers = scheduler["cpu"]["capacity"] if scheduler else 1
    policy = config_type(
        budget=args.budget,
        eval_workers=args.eval_workers,
        scheduler_socket=args.scheduler_socket,
        eval_timeout_seconds=args.eval_timeout_seconds
        if args.eval_timeout_seconds is not None
        else config_type.eval_timeout_seconds,
        **({**options.values(args), 'output_tokens': args.output_tokens} if version in ('v13', 'v13_1') else {}),
    )
    limits = {
        task: {
            "timeout_scope": "instance",
            "function_budget_scope": "mean_instance",
            "timeout_seconds": args.eval_timeout_seconds
            if args.eval_timeout_seconds is not None
            else INSTANCE_LIMITS.get(task, INSTANCE_SECONDS),
            "function_seconds": FUNCTION_LIMITS.get(task, FUNCTION_SECONDS),
        }
        for task in tasks
    }
    plan = plan_for(args, tasks)
    path = RESULTS_ROOT / args.experiment / f"batch_{args.batch}.json"
    used_backends = {item["backend"] for item in plan}
    profiles = {name: BACKENDS[name] for name in used_backends if name != "scheduled"}
    if "scheduled" in used_backends:
        from .base import BackendProfile

        profiles.update(
            {
                f"scheduled-{i}": BackendProfile(e["base_url"], e["model"], e.get("no_proxy", ""))
                for i, e in enumerate(scheduler["endpoints"])
            }
        )
    manifest = {
        "batch": args.batch,
        "method": f"traceaad_{version}" if version in ("v13", "v13_1") else "traceaad",
        "experiment": args.experiment,
        "created_at": datetime.now().astimezone().isoformat(),
        "suite": args.suite,
        "repeats": args.repeats,
        "repeat_ids": list(args.repeat_ids) if args.repeat_ids else None,
        "eval_workers": policy.eval_workers,
        "final_selection": "training",
        "budget_per_run": args.budget,
        "search_policy": asdict(policy),
        "evaluation_limits": limits,
        "llm": {
            name: llm_payload(base_url=p.base_url, model=p.model, no_proxy=p.no_proxy, max_tokens=policy.output_tokens)
            for name, p in profiles.items()
        },
        "git_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--porcelain"),
        "plan": plan,
    }
    if scheduler:
        # launch_plan checks this snapshot against the live scheduler before starting runs.
        manifest["resource_scheduler"] = scheduler
    if args.launch:
        launch_plan(path, manifest, min_context=32768)
    else:
        print(json.dumps({"manifest": str(path), **manifest}, indent=2))
