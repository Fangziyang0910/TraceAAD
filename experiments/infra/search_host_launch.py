"""Launch a five-task, three-repeat batch on the current host."""

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
from benchmarks.tasks import evaluation_limits
from traceaad.common.config import REVISION

ROOT = Path(__file__).resolve().parents[2]


def plan_for(batch, experiment, module, backends=("server3", "server3b"), repeats=3):
    source = {"plan": [{"task": task, "repeat": repeat, "seed": repeat - 1,
                        "backend": backends[0] if (i + repeat) % 2 else backends[1]}
                       for i, task in enumerate(TASKS) for repeat in range(1, repeats + 1)]}
    plan = build_plan(source, batch, experiment, module)
    for item in plan:
        item["command"][:3] = [str(ROOT / ".venv/bin/python")]
        item["startup_log"] = str(ROOT / "experiments_result" / experiment / "launch_logs" / (item["session"] + ".log"))
    return plan


def main(config_class, experiment, module, method, host="local", backends=("server3", "server3b"), argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args(argv)
    plan = plan_for(args.batch, experiment, module, backends)
    manifest_path = ROOT / "experiments_result" / experiment / f"batch_{args.batch}.json"
    if not args.launch:
        print(json.dumps({"manifest": str(manifest_path), "plan": plan,
                          "evaluation_limits": evaluation_limits()}, indent=2))
        return
    hashes = implementation_files(module)
    manifest = {"batch": args.batch, "method": method, "revision": REVISION,
                "experiment": experiment, "created_at": datetime.now().astimezone().isoformat(),
                "execution_host": host, "host": {"node": platform.node(), "cpus": os.cpu_count()},
                "status": "launching", "repeats": 3, "eval_workers": 4,
                "budget_per_run": 1000, "total_budget": 15000, "search_policy": asdict(config_class()),
                "sampling": {**SAMPLING_PROFILES[False], "enable_thinking": False},
                "evaluation_limits": evaluation_limits(),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
                "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).decode(),
                "implementation_files": hashes,
                "load_average_at_launch": list(os.getloadavg()),
                "heldout": "separate after independent selection", "plan": plan}
    launch_plan(manifest_path, manifest, min_context=32768)
