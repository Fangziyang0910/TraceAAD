"""Evaluate the frozen selected program of a completed current search run."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from benchmarks.cvrp_aco import CVRPACOEvaluation
from benchmarks.generated_data_config import get_generated_task_kwargs
from benchmarks.online_bin_packing import OBPEvaluation
from benchmarks.op_aco import OPACOEvaluation
from benchmarks.tsp_construct import TSPEvaluation
from benchmarks.vrptw_construct import VRPTWEvaluation
from traceaad.common.storage import read_json, save_heldout
from .monitor_results import scale_of_split
from traceaad.common.evaluation import ProgramEvaluator
from traceaad.common.config import REVISION


# Held-out runs measure solution quality, not efficiency, so limits only guard
# against hangs. They match the shared baseline evaluator (TSP 3,000 s,
# VRPTW/OBP 1,000 s) so every method faces the same held-out rule. Growing
# limits by (n/50)^2 was too tight: an O(n^3)-per-step TSP heuristic grows
# about 136x from n=50 to n=200.
HELDOUT_TIMEOUT = {"tsp_construct": 3000, "vrptw_construct": 1000, "online_bin_packing": 1000,
                   "cvrp_aco": 3600, "op_aco": 3600}


def heldout_task(task, split, workers, timeout_seconds=None):
    if timeout_seconds is not None and (not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError("timeout_seconds must be finite and positive")
    if workers < 1:
        raise ValueError("workers must be positive")
    if task in {"cvrp_aco", "op_aco"}:
        allowed = ({"test_20", "test_50", "test_100", "test_200"} if task == "cvrp_aco"
                   else {"test_50", "test_100", "test_200"})
        if split not in allowed:
            raise ValueError("unknown ACO held-out split")
        cls = CVRPACOEvaluation if task == "cvrp_aco" else OPACOEvaluation
        return cls(split=split, timeout_seconds=timeout_seconds or HELDOUT_TIMEOUT[task],
                   n_ants=30 if task == "cvrp_aco" else 20,
                   n_iterations=100 if task == "cvrp_aco" else 50, aco_seed=1234,
                   n_workers=workers)
    kwargs = get_generated_task_kwargs(task, "eval")
    if task in {"tsp_construct", "vrptw_construct"}:
        if split not in {"eval", "eval_50", "eval_100", "eval_200"}:
            raise ValueError("constructive tasks use eval_50/eval_100/eval_200")
        size = 50 if split == "eval" else int(split.split("_")[1])
        kwargs["problem_size"] = size
        kwargs["timeout_seconds"] = timeout_seconds or HELDOUT_TIMEOUT[task]
    elif task == "online_bin_packing":
        if split != "eval":
            try:
                _, items, capacity = split.split("_")
                items, capacity = int(items), int(capacity)
            except (ValueError, TypeError) as exc:
                raise ValueError("OBP split must be eval_<items>_<capacity>") from exc
            if items not in {1000, 5000, 10000} or capacity not in {100, 500}:
                raise ValueError("unsupported OBP held-out size/capacity")
            kwargs["dataset_specs"] = [{"n_instances": 5, "n_items": items,
                                        "capacities": [capacity]}]
        kwargs["timeout_seconds"] = timeout_seconds or HELDOUT_TIMEOUT[task]
    else:
        raise ValueError(f"unknown task: {task}")
    cls = {"tsp_construct": TSPEvaluation, "vrptw_construct": VRPTWEvaluation,
           "online_bin_packing": OBPEvaluation}[task]
    return cls(**kwargs)


def evaluate_run(run_dir, *, split=None, workers=2, timeout_seconds=None):
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    if summary["status"] != "finished" or summary["best"]["id"] != selection["selected_node"]:
        raise ValueError("held-out evaluation requires a completed, frozen selection")
    task = config["task"]
    split = split or ("test_50" if task in {"cvrp_aco", "op_aco"} else "eval")
    code = (run_dir / "best_program.py").read_text(encoding="utf-8")
    key = hashlib.sha256(code.encode("utf-8")).hexdigest()
    if key != selection["selected_key"]:
        raise ValueError("selected program changed after selection")
    evaluation = heldout_task(task, split, workers, timeout_seconds)
    seeds = tuple(config["method_params"]["evaluation_seeds"])
    evaluator = ProgramEvaluator(evaluation, seeds, "heldout", measure_calls=config.get("method") != "v1015")
    protocol = evaluator.protocol
    previous = next((r for r in read_json(run_dir / "heldout.json", [])
                     if not r["variant"] and r.get("split") == split), None)
    if previous:
        if (previous["key"] != key or previous.get("node_id") != selection["selected_node"]
                or previous.get("task") != task or previous.get("split") != split
                or previous.get("timeout_seconds") != evaluation.timeout_seconds
                or (previous.get("revision") == REVISION and previous["protocol"] != protocol)):
            raise ValueError("existing held-out result belongs to another program or protocol")
        return previous
    outcome = evaluator.evaluate(code, key)
    records = outcome["evaluations"]
    result = {"task": task, "split": split, "scale": str(scale_of_split(task, split)),
              "verification": "verified", "variant": "", "node_id": selection["selected_node"],
              "key": key, "protocol": protocol, "revision": REVISION, "fitness": outcome["fitness"],
              "scores": [r["score"] for r in records if r["valid"]], "evaluation_seeds": list(seeds),
              "timeout_seconds": evaluation.timeout_seconds,
              "failures": [{"kind": r["failure_kind"], "error": r["error"]} for r in records if not r["valid"]]}
    save_heldout(run_dir, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float)
    args = parser.parse_args(argv)
    print(json.dumps(evaluate_run(args.run_dir, split=args.split, workers=args.workers,
                                  timeout_seconds=args.timeout_seconds), indent=2))


if __name__ == "__main__":
    main()
