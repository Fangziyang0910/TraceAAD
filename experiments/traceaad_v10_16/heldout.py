"""Evaluate a completed V10.16 run's selected program on a held-out split."""

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
from core import SecureEvaluator
from experiments.infra.base import use_cpu_timeout
from traceaad.v10_13.storage import write_json
from traceaad.v10_16.evaluation import SeededEvaluation, protocol_identity


def heldout_task(task, split, workers, timeout_seconds=None):
    return use_cpu_timeout(_heldout_task(task, split, workers, timeout_seconds))


def _heldout_task(task, split, workers, timeout_seconds=None):
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
        return cls(split=split, timeout_seconds=timeout_seconds or 900,
                   n_ants=30 if task == "cvrp_aco" else 20,
                   n_iterations=100 if task == "cvrp_aco" else 50, aco_seed=1234,
                   n_workers=workers)
    kwargs = get_generated_task_kwargs(task, "eval")
    if task in {"tsp_construct", "vrptw_construct"}:
        if split not in {"eval", "eval_50", "eval_100", "eval_200"}:
            raise ValueError("constructive tasks use eval_50/eval_100/eval_200")
        size = 50 if split == "eval" else int(split.split("_")[1])
        kwargs["problem_size"] = size
        kwargs["timeout_seconds"] = timeout_seconds or max(kwargs["timeout_seconds"],
                                                              kwargs["timeout_seconds"] * (size // 50) ** 2)
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
        if timeout_seconds is not None:
            kwargs["timeout_seconds"] = timeout_seconds
    else:
        raise ValueError(f"unknown task: {task}")
    cls = {"tsp_construct": TSPEvaluation, "vrptw_construct": VRPTWEvaluation,
           "online_bin_packing": OBPEvaluation}[task]
    return cls(**kwargs)


def evaluate_run(run_dir, *, split=None, workers=2, timeout_seconds=None):
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "logs" / "run_summary.json").read_text(encoding="utf-8"))
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
    protocol, _ = protocol_identity(evaluation, seeds, "heldout")
    output = run_dir / f"heldout_{split}.json"
    if output.exists():
        previous = json.loads(output.read_text(encoding="utf-8"))
        if previous["key"] != key or previous["protocol"] != protocol:
            raise ValueError("existing held-out result belongs to another program or protocol")
        return previous
    evaluator = SecureEvaluator(SeededEvaluation(evaluation))
    outcomes = [evaluator.evaluate_program_with_details(evaluation.template_program,
                source=code, seed=seed) for seed in seeds]
    scores = [o.result["score"] for o in outcomes if isinstance(o.result, dict)]
    fitness = sum(scores) / len(scores) if len(scores) == len(seeds) else None
    if fitness is not None and not math.isfinite(fitness):
        fitness = None
    result = {"task": task, "split": split, "node_id": selection["selected_node"],
              "key": key, "protocol": protocol, "fitness": fitness, "scores": scores,
              "timeout_seconds": evaluation.timeout_seconds,
              "failures": [{"kind": o.failure_kind, "error": o.error} for o in outcomes if o.result is None]}
    write_json(output, result)
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
