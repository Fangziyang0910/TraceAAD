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
from benchmarks.tasks import ALL_TASKS, SUITES, evaluation_limits
from traceaad.common.config import REVISION

ROOT = Path(__file__).resolve().parents[2]


def plan_for(batch, experiment, module, backends=("server3", "server3b"), repeats=3, tasks=TASKS, budget=1000):
    if repeats < 1 or budget < 1 or not tasks or len(set(tasks)) != len(tasks) or any(task not in ALL_TASKS for task in tasks):
        raise ValueError("require positive repeats/budget and distinct registered tasks")
    source = {"plan": [{"task": task, "repeat": repeat, "seed": repeat - 1,
                        "backend": backends[0] if (i + repeat) % 2 else backends[1]}
                       for i, task in enumerate(tasks) for repeat in range(1, repeats + 1)]}
    plan = build_plan(source, batch, experiment, module, budget)
    for item in plan:
        item["command"][:3] = [str(ROOT / ".venv/bin/python")]
        item["startup_log"] = str(ROOT / "experiments_result" / experiment / "launch_logs" / (item["session"] + ".log"))
    return plan


def main(config_class, experiment, module, method, host="local", backends=("server3", "server3b"), argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--experiment", default=experiment, help="separate result series for the task protocol")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--suite", choices=tuple(SUITES), default="legacy")
    parser.add_argument("--tasks", choices=ALL_TASKS, nargs="+", help="override suite with explicit tasks")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--budget", type=int, default=1000)
    args = parser.parse_args(argv)
    if not args.experiment.replace('_', '').isalnum() or not args.batch.replace('_', '').replace('-', '').isalnum():
        raise ValueError('experiment and batch must be safe directory names')
    experiment = args.experiment
    tasks = tuple(args.tasks or SUITES[args.suite])
    plan = plan_for(args.batch, experiment, module, backends, args.repeats, tasks, args.budget)
    manifest_path = ROOT / "experiments_result" / experiment / f"batch_{args.batch}.json"
    if not args.launch:
        print(json.dumps({"manifest": str(manifest_path), "plan": plan,
                          "suite": args.suite, "final_selection": "training",
                          "evaluation_limits": evaluation_limits(tasks)}, indent=2))
        return
    hashes = implementation_files(module)
    manifest = {"batch": args.batch, "method": method, "revision": REVISION,
                "experiment": experiment, "created_at": datetime.now().astimezone().isoformat(),
                "execution_host": host, "host": {"node": platform.node(), "cpus": os.cpu_count()},
                "status": "launching", "suite": args.suite, "repeats": args.repeats, "eval_workers": 4,
                "final_selection": "training",
                "budget_per_run": args.budget, "total_budget": len(plan) * args.budget,
                "search_policy": asdict(config_class(budget=args.budget)),
                "sampling": {**SAMPLING_PROFILES[False], "enable_thinking": False},
                "evaluation_limits": evaluation_limits(tasks),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
                "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).decode(),
                "implementation_files": hashes,
                "load_average_at_launch": list(os.getloadavg()),
                "heldout": "separate evaluation of the frozen training-best program", "plan": plan}
    launch_plan(manifest_path, manifest, min_context=32768)
