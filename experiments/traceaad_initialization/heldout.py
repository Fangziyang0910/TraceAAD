"""Evaluate each finished run's training-best program on unseen instances."""

from __future__ import annotations

import argparse
import hashlib
import json
import math

from benchmarks.cvrp_aco import CVRPACOEvaluation
from benchmarks.generated_data_config import get_generated_task_kwargs
from benchmarks.online_bin_packing import OBPEvaluation
from benchmarks.op_aco import OPACOEvaluation
from benchmarks.tsp_construct import TSPEvaluation
from benchmarks.vrptw_construct import VRPTWEvaluation
from core import SecureEvaluator
from traceaad.v10_13.storage import write_json

from .launch import SCHEDULE, run_dir


def make_evaluator(task):
    if task == "cvrp_aco":
        return CVRPACOEvaluation(split="test_50", timeout_seconds=900,
                                 n_ants=30, n_iterations=100,
                                 aco_seed=1234, n_workers=2), "test_50"
    if task == "op_aco":
        return OPACOEvaluation(split="test_50", timeout_seconds=900,
                               n_ants=20, n_iterations=50,
                               aco_seed=1234, n_workers=2), "test_50"
    kwargs = get_generated_task_kwargs(task, "eval")
    cls = {"tsp_construct": TSPEvaluation,
           "online_bin_packing": OBPEvaluation,
           "vrptw_construct": VRPTWEvaluation}[task]
    return cls(**kwargs), "eval"


def evaluate_one(job, evaluators):
    directory = run_dir(job)
    summary_path = directory / "logs" / "run_summary.json"
    if not summary_path.exists():
        return "waiting"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "finished":
        return "waiting"
    best = summary.get("best")
    if not best or not best.get("code"):
        raise ValueError(f"finished run has no best program: {directory}")
    code = best["code"]
    code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
    output = directory / "heldout.json"
    if output.exists():
        previous = json.loads(output.read_text(encoding="utf-8"))
        if previous.get("code_sha256") != code_hash or previous.get("node_id") != best["id"]:
            raise ValueError(f"held-out result does not match final training best: {directory}")
        return "existing"
    if job["task"] not in evaluators:
        evaluators[job["task"]] = make_evaluator(job["task"])
    evaluator, split = evaluators[job["task"]]
    outcome = SecureEvaluator(evaluator).evaluate_program_with_details(code)
    try:
        fitness = float(outcome.result) if outcome.result is not None else None
    except (ValueError, TypeError, OverflowError):
        fitness = None
    if fitness is not None and not math.isfinite(fitness):
        fitness = None
    write_json(output, {
        "task": job["task"], "mode": job["mode"], "repeat": job["repeat"],
        "split": split, "node_id": best["id"], "code_sha256": code_hash,
        "train_fitness": best["fitness"], "heldout_fitness": fitness,
        "failure_kind": outcome.failure_kind, "error": outcome.error,
    })
    return "evaluated"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    jobs = json.loads(SCHEDULE.read_text(encoding="utf-8"))
    evaluators = {}
    counts = {"waiting": 0, "existing": 0, "evaluated": 0, "ready": 0}
    for job in jobs:
        if args.dry_run:
            directory = run_dir(job)
            summary_path = directory / "logs" / "run_summary.json"
            ready = summary_path.exists() and json.loads(
                summary_path.read_text(encoding="utf-8")).get("status") == "finished"
            if ready:
                status = "existing" if (directory / "heldout.json").exists() else "ready"
            else:
                status = "waiting"
        else:
            status = evaluate_one(job, evaluators)
        counts[status] += 1
        if status == "evaluated":
            print(f"held-out complete: {job['task']} {job['mode']} r{job['repeat']}", flush=True)
    print(json.dumps(counts, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
