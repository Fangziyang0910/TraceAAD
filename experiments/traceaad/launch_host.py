"""Preview or start independently recorded versioned TraceAAD searches on the current host."""

import argparse
from dataclasses import asdict
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
from .run import (
    ALLOCATION_FIELDS,
    BACKENDS,
    ECO_BACKEND,
    EVOTRACE_METHODS,
    METHODS,
    TOPOLOGY_FIELDS,
    TREE_PARAM_METHODS,
    V3AllocationConfig,
    V3TopologyConfig,
    V4AllocationConfig,
    V4TopologyConfig,
    eco_routing,
    request_options,
    selection_rule_for,
)
from .scheduling import EVALUATION_WORKERS, scheduler_status


ROOT = Path(__file__).resolve().parents[2]


def _slug(method):
    return method.replace(".", "_")


def build_plan(args):
    plan = []
    method_id = METHODS[args.method][0]
    allocation = {key: getattr(args, key) for key in ALLOCATION_FIELDS
                  if getattr(args, key, None) is not None}
    topology = {key: getattr(args, key) for key in TOPOLOGY_FIELDS
                if getattr(args, key, None) is not None}
    if topology:
        if method_id not in TREE_PARAM_METHODS:
            raise ValueError("Topology triggers apply to EvoTrace V3/V4 only")
        (V4TopologyConfig if method_id == "evotrace_v4" else V3TopologyConfig)(**topology)
    if allocation:
        if method_id not in TREE_PARAM_METHODS:
            raise ValueError("Allocation triggers apply to EvoTrace V3/V4 only")
        (V4AllocationConfig if method_id == "evotrace_v4" else V3AllocationConfig)(**allocation)
    reflection = getattr(args, "reflection", None)
    if reflection is not None and method_id != "evotrace_v4":
        raise ValueError("Reflection toggle applies to EvoTrace V4 only")
    slug = _slug(args.method)
    tasks = args.tasks if args.tasks is not None else CO_TASKS
    if not tasks or len(set(tasks)) != len(tasks) or any(task not in CO_TASKS for task in tasks):
        raise ValueError("tasks must be distinct members of co6")
    for task in tasks:
        for seed in range(args.repeats):
            name = f"{args.batch}_{TASK_SHORT[task]}_{method_id}_rep{seed + 1}"
            session = f"{args.batch}_{TASK_SHORT[task]}_{slug}_r{seed + 1}"
            directory = ROOT / "experiments_result" / args.experiment / task / name
            command = [sys.executable, "-u", "-m", "experiments.traceaad.run",
                       "--method", args.method, "--backend", args.backend, "--task", task,
                       "--run-name", name, "--batch", args.batch,
                       "--experiment", args.experiment, "--seed", str(seed),
                       "--budget", str(args.budget),
                       "--initial-size", str(args.initial_size), "--max-tokens", str(args.max_tokens),
                       "--model-timeout", str(args.model_timeout)]
            if method_id not in EVOTRACE_METHODS:
                command += ["--selection-temperature", str(args.selection_temperature)]
            for key, value in allocation.items():
                command += ["--" + key.replace("_", "-"), str(value)]
            for key, value in topology.items():
                command += ["--" + key.replace("_", "-"), str(value)]
            if reflection is False:
                command += ["--no-reflection"]
            elif reflection is True:
                command += ["--reflection"]
            if args.scheduler_socket:
                command += ["--scheduler-socket", args.scheduler_socket]
            command += ["--seconds-per-instance", str(args.seconds_per_instance)]
            if args.env_file:
                command += ["--env-file", str(args.env_file.resolve())]
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
    parser.add_argument("--backend", choices=BACKENDS, default="scheduled")
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
    parser.add_argument("--rank-half-life", type=float, help="V3/V4 quality weight half-life in ranks")
    parser.add_argument("--explore-references", type=int, help="V3/V4 maximum diverse Explore overview cards")
    parser.add_argument("--max-representatives", type=int, help="V3/V4 maximum mutually uncovered core representatives")
    parser.add_argument("--cover-distance", type=float, help="V3/V4 AST shingle distance covered by a better representative")
    parser.add_argument("--region-size", type=int, help="V3/V4 minimum non-skeleton nodes per simplified region")
    parser.add_argument("--reflection", action=argparse.BooleanOptionalAction, default=None,
                        help="V4 only: enable or disable reflection during tree simplification (default: enabled)")
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
    if scheduler is None:
        parser.error("Evaluation requires --scheduler-socket")
    path = ROOT / "experiments_result" / args.experiment / f"batch_{args.batch}.json"
    manifest = {
        "batch": args.batch, "method": METHODS[args.method][0], "experiment": args.experiment,
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "suite": "co6", "repeats": args.repeats,
        "tasks": list(args.tasks if args.tasks is not None else CO_TASKS),
        "budget_per_run": args.budget, "budget_axis": "评价次数", "budget_basis": "evaluate_calls",
        "output_tokens": args.max_tokens, "temperature": 1., "initial_size": args.initial_size,
        "selection_rule": selection_rule_for(args.method), "selection_temperature": args.selection_temperature,
        "trajectory_reference_embedding": False,
        "mechanisms": list(METHODS[args.method][1].mechanisms),
        "generation_protocol": METHODS[args.method][1].generation_protocol,
        "backend": args.backend, "sampling": request_options(args.method, args.backend),
        "final_selection": "training", "eval_workers": EVALUATION_WORKERS,
        "evaluation_scope": "whole_task", "resource_scheduler": scheduler,
        "seconds_per_instance": args.seconds_per_instance,
        "source_snapshot": str(ROOT),
        "pipeline": "search -> freeze training best -> independent held-out evaluation",
        "status": "planned", "plan": plan,
    }
    if args.backend == ECO_BACKEND:
        manifest["model_routing"] = eco_routing()
    method_id = METHODS[args.method][0]
    if method_id in EVOTRACE_METHODS:
        manifest.pop("selection_temperature")
    if method_id in TREE_PARAM_METHODS:
        alloc_cls = V4AllocationConfig if method_id == "evotrace_v4" else V3AllocationConfig
        topo_cls = V4TopologyConfig if method_id == "evotrace_v4" else V3TopologyConfig
        manifest["allocation"] = asdict(alloc_cls(**{
            key: getattr(args, key) for key in ALLOCATION_FIELDS if getattr(args, key) is not None}))
        manifest["topology"] = asdict(topo_cls(**{
            key: getattr(args, key) for key in TOPOLOGY_FIELDS if getattr(args, key) is not None}))
    if method_id == "evotrace_v4":
        reflection = True if args.reflection is None else args.reflection
        manifest["reflection"] = reflection
        manifest["include_experiences"] = reflection
        manifest["mechanisms"] = list(METHODS[args.method][1].mechanisms_for(reflection))
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
