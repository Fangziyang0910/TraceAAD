"""Plan or launch explicit legacy and six-task AHD suites on the current host."""

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import subprocess

from experiments.infra.base import SAMPLING_PROFILES, TASKS
from experiments.infra.launcher import launch_plan
from experiments.infra.search_launch import build_plan, implementation_files
from benchmarks.tasks import ALL_TASKS, FUNCTION_SECONDS, SUITES, evaluation_limits
from traceaad.common.config import REVISION

ROOT = Path(__file__).resolve().parents[2]


def plan_for(batch, experiment, module, backends=("server3", "server3b"), repeats=3, tasks=TASKS, budget=1000,
             *, eval_workers=4, eval_timeout_seconds=None, scheduler_socket=None, repeat_ids=None):
    if repeats < 1 or budget < 1 or not tasks or len(set(tasks)) != len(tasks) or any(task not in ALL_TASKS for task in tasks):
        raise ValueError("require positive repeats/budget and distinct registered tasks")
    source = {"plan": [{"task": task, "repeat": repeat, "seed": repeat - 1,
                        "backend": backends[0] if (i + repeat) % 2 else backends[1]}
                       for i, task in enumerate(tasks) for repeat in range(1, repeats + 1)]}
    if repeat_ids is not None:
        if not repeat_ids or len(set(repeat_ids)) != len(repeat_ids) or not set(repeat_ids) <= set(range(1, repeats + 1)):
            raise ValueError('repeat IDs must be distinct and within 1..repeats')
        source['plan'] = [item for item in source['plan'] if item['repeat'] in repeat_ids]
    plan = build_plan(source, batch, experiment, module, budget,
                      eval_workers=eval_workers, eval_timeout_seconds=eval_timeout_seconds,
                      scheduler_socket=scheduler_socket)
    for item in plan:
        item["command"][:3] = [str(ROOT / ".venv/bin/python")]
        item["startup_log"] = str(ROOT / "experiments_result" / experiment / "launch_logs" / (item["session"] + ".log"))
    return plan


def git(*args):
    """Git facts when this tree is a checkout; server3 holds a plain copy, identified by the file hashes."""
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(config_class, experiment, module, method, host="local", backends=("server3", "server3b"), argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--experiment", default=experiment, help="separate result series for the task protocol")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--suite", choices=tuple(SUITES), default="legacy")
    parser.add_argument("--tasks", choices=ALL_TASKS, nargs="+", help="override suite with explicit tasks")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--budget", type=int, default=1000)
    instance_execution = hasattr(config_class, "eval_timeout_seconds")
    if instance_execution:
        parser.set_defaults(suite="co6")
        parser.add_argument("--eval-workers", type=int, help='per-evaluation ceiling; scheduled default: host CPU capacity')
        from benchmarks.tasks import INSTANCE_SECONDS
        parser.add_argument("--eval-timeout-seconds", type=float, default=float(INSTANCE_SECONDS))
        parser.add_argument('--scheduler-socket', help='shared CPU/GPU scheduler on this host')
        parser.add_argument('--repeat-ids', type=int, nargs='+', help='run only these repeats on this host, preserving seeds')
    args = parser.parse_args(argv)
    if not args.experiment.replace('_', '').isalnum() or not args.batch.replace('_', '').replace('-', '').isalnum():
        raise ValueError('experiment and batch must be safe directory names')
    experiment = args.experiment
    tasks = tuple(args.tasks or SUITES[args.suite])
    scheduler = None
    if getattr(args, 'scheduler_socket', None):
        from core.scheduling import scheduler_status
        args.scheduler_socket = str(Path(args.scheduler_socket).expanduser().resolve())
        scheduler = scheduler_status(args.scheduler_socket)
    if instance_execution and args.eval_workers is None:
        args.eval_workers = scheduler['cpu']['capacity'] if scheduler else 1
    execution = ({"eval_workers": args.eval_workers, "eval_timeout_seconds": args.eval_timeout_seconds}
                 if instance_execution else {})
    if scheduler:
        execution['scheduler_socket'] = args.scheduler_socket
    policy = config_class(budget=args.budget, **execution)
    plan = plan_for(args.batch, experiment, module, backends, args.repeats, tasks, args.budget,
                     repeat_ids=getattr(args, 'repeat_ids', None), **execution)
    limits = evaluation_limits(tasks)
    metadata = ({"evaluation_execution": {"timeout_scope": "instance",
                 "timeout_seconds": policy.eval_timeout_seconds, "function_seconds": FUNCTION_SECONDS,
                 "n_workers": policy.eval_workers,
                 "protocol": "isolated-instances-v1"}} if instance_execution else {})
    if instance_execution:
        limits = {task: {"timeout_scope": "instance", "timeout_seconds": policy.eval_timeout_seconds,
                         "function_seconds": FUNCTION_SECONDS}
                  for task in tasks}
    if scheduler:
        metadata['resource_scheduler'] = scheduler
    manifest_path = ROOT / "experiments_result" / experiment / f"batch_{args.batch}.json"
    if not args.launch:
        print(json.dumps({"manifest": str(manifest_path), "plan": plan,
                          "suite": args.suite, "final_selection": "training",
                          "evaluation_limits": limits, **metadata}, indent=2))
        return
    hashes = implementation_files(module)
    manifest = {"batch": args.batch, "method": method, "revision": REVISION,
                "experiment": experiment, "created_at": datetime.now().astimezone().isoformat(),
                "execution_host": platform.node() if scheduler else host,
                "host": {"node": platform.node(), "cpus": os.cpu_count()},
                "status": "launching", "suite": args.suite, "repeats": args.repeats,
                "eval_workers": policy.eval_workers if instance_execution else 4, **metadata,
                "repeat_ids": getattr(args, 'repeat_ids', None),
                "final_selection": "training",
                "budget_per_run": args.budget, "total_budget": len(plan) * args.budget,
                "search_policy": asdict(policy),
                "sampling": {**SAMPLING_PROFILES[False], "enable_thinking": False},
                "evaluation_limits": limits,
                "git_commit": git("rev-parse", "HEAD"),
                "git_status": git("status", "--porcelain"),
                "implementation_files": hashes,
                "load_average_at_launch": list(os.getloadavg()),
                "heldout": "separate evaluation of the frozen training-best program", "plan": plan}
    launch_plan(manifest_path, manifest, min_context=32768)
