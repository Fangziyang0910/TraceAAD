"""Preview or start independently recorded versioned TraceAAD searches on the current host."""

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
from experiments.infra.launcher import is_session_alive
from traceaad_base.storage import write_json
from traceaad_base.selection import SELECTION_RULE
from .run import METHODS, reference_embedding_enabled, request_options, selection_rule_for
from .scheduling import scheduler_status


ROOT = Path(__file__).resolve().parents[2]


def _slug(method):
    return method.replace(".", "_")


def build_plan(args):
    plan = []
    slug = _slug(args.method)
    embedding = reference_embedding_enabled(args.method, args.trajectory_reference_embedding)
    tasks = args.tasks if args.tasks is not None else CO_TASKS
    if not tasks or len(set(tasks)) != len(tasks) or any(task not in CO_TASKS for task in tasks):
        raise ValueError("tasks must be distinct members of co6")
    for task in tasks:
        method_id = METHODS[args.method][0]
        for seed in range(args.repeats):
            name = f"{args.batch}_{TASK_SHORT[task]}_{method_id}_rep{seed + 1}"
            session = f"{args.batch}_{TASK_SHORT[task]}_{slug}_r{seed + 1}"
            directory = ROOT / "experiments_result" / args.experiment / task / name
            command = [sys.executable, "-u", "-m", "experiments.traceaad.run",
                       "--method", args.method, "--backend", args.backend, "--task", task, "--run-name", name, "--batch", args.batch,
                       "--experiment", args.experiment, "--seed", str(seed),
                       "--budget", str(args.budget),
                       "--initial-size", str(args.initial_size), "--max-tokens", str(args.max_tokens),
                       "--model-timeout", str(args.model_timeout)]
            if method_id not in {"evotrace_v1", "evotrace_v2", "evotrace_v3"}:
                command += ["--selection-temperature", str(args.selection_temperature)]
            command += ["--trajectory-reference-embedding" if embedding else "--no-trajectory-reference-embedding"]
            if args.scheduler_socket:
                command += ["--scheduler-socket", args.scheduler_socket]
            command += ["--seconds-per-instance", str(args.seconds_per_instance)]
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


def validate_run(row):
    if Path(row["run_dir"]).exists() or Path(row["startup_log"]).exists():
        raise FileExistsError(f"Run already exists: {row['run_name']}")
    if is_session_alive(row["session"]):
        raise FileExistsError(f"Session already exists: {row['session']}")


def launch_run(row):
    """Start exactly one immutable plan row; its process includes the held-out phase."""
    validate_run(row)
    log = Path(row["startup_log"])
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = ["env", "HF_HUB_OFFLINE=1", "PYTHONDONTWRITEBYTECODE=1",
                   "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1",
                   "NUMEXPR_NUM_THREADS=1", "VECLIB_MAXIMUM_THREADS=1", f"PYTHONPATH={ROOT}"]
    shell = "exec " + shlex.join(environment + row["command"]) + " > " + shlex.quote(str(log)) + " 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", row["session"], "-c", str(ROOT), shell], check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=tuple(METHODS), default="base")
    parser.add_argument("--backend", choices=("scheduled",), default="scheduled")
    parser.add_argument("--batch", required=True)
    parser.add_argument("--experiment", help="Result directory name; defaults to the selected method")
    parser.add_argument("--tasks", nargs="+", choices=CO_TASKS,
                        help="Explicit task subset; default: all six tasks")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--initial-size", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--model-timeout", type=float, default=600)
    parser.add_argument("--selection-temperature", type=float, default=1.0,
                        help="Boltzmann parent temperature in fitness units")
    parser.add_argument("--trajectory-reference-embedding", action=argparse.BooleanOptionalAction,
                        default=None, help="Mechanism 1 defaults to trajectory references; --no-... ablates them")
    parser.add_argument("--seconds-per-instance", type=float, default=10.,
                        help="Time allowance per instance; their sum is the whole-task deadline")
    parser.add_argument("--scheduler-socket")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--spacing", type=float, default=3)
    parser.add_argument("--skip-heldout", action="store_true")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--launch", action="store_true")
    action.add_argument("--enqueue", action="store_true", help="Save a queued manifest without starting workers")
    args = parser.parse_args(argv)
    args.experiment = args.experiment or METHODS[args.method][0]
    try:
        args.trajectory_reference_embedding = reference_embedding_enabled(args.method, args.trajectory_reference_embedding)
    except ValueError as error:
        parser.error(str(error))
    if args.repeats < 1 or args.spacing < 0:
        parser.error("repeats must be positive and spacing must be nonnegative")
    if not math.isfinite(args.seconds_per_instance) or args.seconds_per_instance <= 0:
        parser.error("seconds-per-instance must be finite and positive")
    if not math.isfinite(args.selection_temperature) or args.selection_temperature <= 0:
        parser.error("selection-temperature must be finite and positive")
    for name in (args.batch, args.experiment):
        if not name.replace("_", "").replace("-", "").isalnum():
            parser.error("Use alphanumeric names with underscores or hyphens")
    plan = build_plan(args)
    scheduler = scheduler_status(args.scheduler_socket) if args.scheduler_socket else None
    if args.backend == "scheduled" and scheduler is None:
        parser.error("scheduled backend requires --scheduler-socket")
    path = ROOT / "experiments_result" / args.experiment / f"batch_{args.batch}.json"
    manifest = {
        "batch": args.batch, "method": METHODS[args.method][0], "experiment": args.experiment,
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "suite": "co6", "repeats": args.repeats,
        "tasks": list(args.tasks if args.tasks is not None else CO_TASKS),
        "budget_per_run": args.budget, "budget_axis": "评价次数", "budget_basis": "evaluate_calls",
        "output_tokens": args.max_tokens, "temperature": 1., "initial_size": args.initial_size,
        "selection_rule": selection_rule_for(args.method), "selection_temperature": args.selection_temperature,
        "trajectory_reference_embedding": args.trajectory_reference_embedding,
        "mechanisms": list(METHODS[args.method][1].mechanisms),
        "generation_protocol": METHODS[args.method][1].generation_protocol,
        "sampling": request_options(args.method), "final_selection": "training", "eval_workers": 1,
        "evaluation_scope": "whole_task", "resource_scheduler": scheduler,
        "seconds_per_instance": args.seconds_per_instance,
        "source_snapshot": str(ROOT),
        "pipeline": "search -> freeze training best -> independent held-out evaluation",
        "status": "planned", "plan": plan,
    }
    if METHODS[args.method][0] in {"evotrace_v1", "evotrace_v2", "evotrace_v3"}:
        manifest.pop("selection_temperature")
    if not args.launch and not args.enqueue:
        print(json.dumps({"manifest": str(path), **manifest}, ensure_ascii=False, indent=2))
        return
    if path.exists():
        raise FileExistsError(f"Batch manifest already exists: {path}")
    for row in plan:
        validate_run(row)
    if args.enqueue:
        manifest["status"] = "queued"
        for row in plan:
            row["status"] = "queued"
        write_json(path, manifest)
        print(json.dumps({"manifest": str(path), "runs": len(plan), "status": "queued"}), flush=True)
        return
    write_json(path, manifest)
    for index, row in enumerate(plan):
        if index:
            time.sleep(args.spacing)
        launch_run(row)
        row.update(status="launched", started_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat())
        write_json(path, manifest)
        print(f"started {index + 1}/{len(plan)} {row['session']}", flush=True)
    manifest["status"] = "launched"
    write_json(path, manifest)
    print(json.dumps({"manifest": str(path), "runs": len(plan), "status": "launched"}), flush=True)


if __name__ == "__main__":
    main()
