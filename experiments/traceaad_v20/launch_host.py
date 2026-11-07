"""Preview or start independently recorded Base, V20, V21 or V22 searches on the current host."""

import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
import time

from benchmarks.tasks import CO_TASKS, TASK_SHORT
from traceaad_base.storage import write_json
from .run import request_options
from .scheduling import scheduler_status


ROOT = Path(__file__).resolve().parents[2]


def build_plan(args):
    plan = []
    for task in CO_TASKS:
        for seed in range(args.repeats):
            name = f"{args.batch}_{TASK_SHORT[task]}_traceaad_{args.method}_rep{seed + 1}"
            session = f"{args.batch}_{TASK_SHORT[task]}_{args.method}_r{seed + 1}"
            directory = ROOT / "experiments_result" / args.experiment / task / name
            command = [sys.executable, "-u", "-m", "experiments.traceaad_v20.run",
                       "--method", args.method, "--backend", args.backend, "--task", task, "--run-name", name, "--batch", args.batch,
                       "--experiment", args.experiment, "--seed", str(seed),
                       "--budget", str(args.budget),
                       "--initial-size", str(args.initial_size), "--max-tokens", str(args.max_tokens),
                       "--model-timeout", str(args.model_timeout)]
            if args.scheduler_socket:
                command += ["--scheduler-socket", args.scheduler_socket]
            command += ["--seconds-per-instance", str(args.seconds_per_instance)]
            for feature in ("trajectory-scoring","trajectory-context","reference-selection","experience-learning"):
                if getattr(args,"no_"+feature.replace("-","_"),False):command += ["--no-"+feature]
            if args.env_file:
                command += ["--env-file", str(args.env_file)]
            if args.skip_heldout:
                command += ["--skip-heldout"]
            plan.append({"task": task, "seed": seed, "repeat": seed + 1, "backend": args.backend,
                         "run_name": name, "run_dir": str(directory), "session": session,
                         "command": command, "status": "planned",
                         "startup_log": str(ROOT / "experiments_result" / args.experiment /
                                            "launch_logs" / f"{session}.log")})
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("v20", "v21", "v22", "base"), default="v20")
    parser.add_argument("--backend", choices=("scheduled",), default="scheduled")
    for feature in ("trajectory-scoring","trajectory-context","reference-selection","experience-learning"):
        parser.add_argument("--no-"+feature,action="store_true")
    parser.add_argument("--batch", required=True)
    parser.add_argument("--experiment", default="traceaad_v20")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--initial-size", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--model-timeout", type=float, default=600)
    parser.add_argument("--seconds-per-instance", type=float, default=10.,
                        help="Time allowance per instance; their sum is the whole-task deadline")
    parser.add_argument("--scheduler-socket")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--spacing", type=float, default=3)
    parser.add_argument("--skip-heldout", action="store_true")
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.spacing < 0:
        parser.error("repeats must be positive and spacing must be nonnegative")
    if not math.isfinite(args.seconds_per_instance) or args.seconds_per_instance <= 0:
        parser.error("seconds-per-instance must be finite and positive")
    for name in (args.batch, args.experiment):
        if not name.replace("_", "").replace("-", "").isalnum():
            parser.error("Use alphanumeric names with underscores or hyphens")
    plan = build_plan(args)
    scheduler = scheduler_status(args.scheduler_socket) if args.scheduler_socket else None
    if args.backend == "scheduled" and scheduler is None:
        parser.error("scheduled backend requires --scheduler-socket")
    path = ROOT / "experiments_result" / args.experiment / f"batch_{args.batch}.json"
    manifest = {
        "batch": args.batch, "method": f"traceaad_{args.method}", "experiment": args.experiment,
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "suite": "co6", "repeats": args.repeats,
        "budget_per_run": args.budget, "budget_axis": "评价次数", "budget_basis": "evaluate_calls",
        "output_tokens": args.max_tokens, "temperature": 1., "initial_size": args.initial_size,
        "sampling": request_options(args.method), "final_selection": "training", "eval_workers": 1,
        "evaluation_scope": "whole_task", "resource_scheduler": scheduler,
        "seconds_per_instance": args.seconds_per_instance,
        "source_snapshot": str(ROOT),
        "pipeline": "search -> freeze training best -> independent held-out evaluation",
        "status": "planned", "plan": plan,
    }
    if not args.launch:
        print(json.dumps({"manifest": str(path), **manifest}, ensure_ascii=False, indent=2))
        return
    if path.exists():
        raise FileExistsError(f"Batch manifest already exists: {path}")
    for row in plan:
        if Path(row["run_dir"]).exists() or Path(row["startup_log"]).exists():
            raise FileExistsError(f"Run already exists: {row['run_name']}")
        if subprocess.run(["tmux", "has-session", "-t", row["session"]],
                          capture_output=True).returncode == 0:
            raise FileExistsError(f"Session already exists: {row['session']}")
    write_json(path, manifest)
    for index, row in enumerate(plan):
        if index:
            time.sleep(args.spacing)
        log = Path(row["startup_log"])
        log.parent.mkdir(parents=True, exist_ok=True)
        environment = ["env", "HF_HUB_OFFLINE=1", "PYTHONDONTWRITEBYTECODE=1",
                       "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1",
                       "NUMEXPR_NUM_THREADS=1", "VECLIB_MAXIMUM_THREADS=1", f"PYTHONPATH={ROOT}"]
        shell = "exec " + shlex.join(environment + row["command"]) + " > " + shlex.quote(str(log)) + " 2>&1"
        subprocess.run(["tmux", "new-session", "-d", "-s", row["session"], "-c", str(ROOT), shell], check=True)
        row.update(status="launched", started_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat())
        write_json(path, manifest)
        print(f"started {index + 1}/{len(plan)} {row['session']}", flush=True)
    manifest["status"] = "launched"
    write_json(path, manifest)
    print(json.dumps({"manifest": str(path), "runs": len(plan), "status": "launched"}), flush=True)


if __name__ == "__main__":
    main()
