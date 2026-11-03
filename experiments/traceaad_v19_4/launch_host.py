"""Preview or explicitly launch V19.4 searches with separate run directories."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from benchmarks.tasks import CO_TASKS, TASK_SHORT
from traceaad_v19_4 import Config
from traceaad_v19_4.cli import add_config_arguments
from traceaad_v19_4.search import METHOD
from traceaad_v19_4.storage import write_json


ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--experiment", default=METHOD)
    parser.add_argument("--tasks", nargs="+", choices=CO_TASKS, default=list(CO_TASKS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--base-url", default=os.getenv("TRACEAAD_BASE_URL"))
    parser.add_argument("--model", default=os.getenv("TRACEAAD_MODEL"))
    parser.add_argument("--scheduler-socket", default=os.getenv("TRACEAAD_SCHEDULER_SOCKET"))
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--seconds-per-instance", type=float, default=10.)
    parser.add_argument("--skip-heldout", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--spacing", type=float, default=3.)
    add_config_arguments(parser)
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.spacing < 0:
        parser.error("repeats must be positive and spacing nonnegative")
    for name in (args.batch, args.experiment):
        if not name.replace("_", "").replace("-", "").isalnum():
            parser.error("Use alphanumeric names with underscores or hyphens")
    settings = {k: getattr(args, k) for k in asdict(Config()) if getattr(args, k, None) is not None}
    config = Config(**settings)
    plan = []
    for task in args.tasks:
        for seed in range(args.repeats):
            name = f"{args.batch}_{TASK_SHORT[task]}_traceaad_v19_4_rep{seed + 1}"
            directory = ROOT / "experiments_result" / args.experiment / task / name
            command = [sys.executable, "-u", "-m", "experiments.traceaad_v19_4.run",
                       "--task", task, "--run-name", name, "--batch", args.batch,
                       "--experiment", args.experiment, "--seconds-per-instance", str(args.seconds_per_instance),
                       "--api-key-env", args.api_key_env]
            for key, value in asdict(config).items():
                flag = "--budget" if key == "evaluation_budget" else "--" + key.replace("_", "-")
                if isinstance(value, bool):
                    command.append(flag if value else "--no-" + key.replace("_", "-"))
                else:
                    command.extend([flag, str(seed if key == "seed" else value)])
            for key in ("base_url", "model", "scheduler_socket", "env_file"):
                if getattr(args, key):
                    command.extend(["--" + key.replace("_", "-"), str(getattr(args, key))])
            if args.skip_heldout:
                command.append("--skip-heldout")
            plan.append({"run_name": name, "task": task, "seed": seed, "run_dir": str(directory),
                         "session": name, "command": command})
    manifest = {"method": METHOD, "batch": args.batch,
                "budget_per_run": config.evaluation_budget,
                "budget_basis": "evaluate_calls", "method_params": asdict(config), "plan": plan,
                "pipeline": "search -> freeze training best -> held-out evaluation", "status": "planned"}
    path = ROOT / "experiments_result" / args.experiment / f"batch_{args.batch}.json"
    if not args.launch:
        print(json.dumps({"manifest": str(path), **manifest}, ensure_ascii=False, indent=2))
        return manifest
    if not args.base_url and not args.scheduler_socket:
        parser.error("Launch requires a model endpoint or scheduler")
    if path.exists() or any(Path(row["run_dir"]).exists() for row in plan):
        raise FileExistsError("Batch or run directory already exists")
    for row in plan:
        if subprocess.run(["tmux", "has-session", "-t", row["session"]], capture_output=True).returncode == 0:
            raise FileExistsError("tmux session already exists: " + row["session"])
    write_json(path, manifest)
    for index, row in enumerate(plan):
        if index:
            time.sleep(args.spacing)
        log = ROOT / "experiments_result" / args.experiment / "launch_logs" / (row["session"] + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        command = ["env", "HF_HUB_OFFLINE=1", "TOKENIZERS_PARALLELISM=false", "OMP_NUM_THREADS=1",
                   "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1", f"PYTHONPATH={ROOT}", *row["command"]]
        shell = "exec " + shlex.join(command) + " > " + shlex.quote(str(log)) + " 2>&1"
        subprocess.run(["tmux", "new-session", "-d", "-s", row["session"], "-c", str(ROOT), shell], check=True)
        row["status"] = "launched"
        write_json(path, manifest)
    manifest["status"] = "launched"
    write_json(path, manifest)
    print(json.dumps({"manifest": str(path), "runs": len(plan)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
