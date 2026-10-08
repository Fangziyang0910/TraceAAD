"""Evaluate the frozen selected program of a completed current search run."""

import argparse
import json
from pathlib import Path

from benchmarks.tasks import heldout_task, scale_of_split
from traceaad.common.storage import save_heldout, selected_program, stored_heldout
from .evaluation_execution import heldout_evaluator
from core.evaluate import EVALUATION_SEED
from traceaad.common.config import REVISION


def evaluate_run(run_dir, *, split=None, workers=None, timeout_seconds=None, scheduler_socket=None):
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    best = selected_program(run_dir)
    if not best:
        raise ValueError("held-out evaluation requires a completed, frozen selection")
    if not best["verified"]:
        raise ValueError("selected program changed after selection")
    task = config["task"]
    split = split or ("test_50" if task in {"cvrp_aco", "op_aco"} else "eval")
    code, key = best["code"], best["key"]
    seeds = tuple(config.get("method_params", {}).get("evaluation_seeds", [EVALUATION_SEED]))
    scale = str(scale_of_split(task, split))
    previous = stored_heldout(run_dir, scale)
    if previous:
        if (previous["key"] != key or previous.get("node_id") != best["id"]
                or previous.get("task") != task):
            raise ValueError("existing held-out result belongs to another program")
        return previous
    evaluator, execution = heldout_evaluator(config, split, seeds, workers=workers,
        timeout_seconds=timeout_seconds, measure_calls=True,
        task_factory=heldout_task, scheduler_socket=scheduler_socket)
    protocol = evaluator.protocol
    outcome = evaluator.evaluate(code, key)
    records = outcome["evaluations"]
    result = {"task": task, "split": split, "scale": scale,
              "verification": "verified", "variant": "", "node_id": best["id"],
              "key": key, "protocol": protocol, "revision": REVISION, "fitness": outcome["fitness"],
              "scores": [r["score"] for r in records if r["valid"]], "evaluation_seeds": list(seeds),
              **execution, "evaluations": records,
              "failures": [{"kind": r["failure_kind"], "error": r["error"]} for r in records if not r["valid"]]}
    save_heldout(run_dir, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split")
    parser.add_argument("--workers", type=int, help="instance workers; defaults to the saved execution settings")
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument('--scheduler-socket', help='override saved CPU scheduler; empty string disables it')
    args = parser.parse_args(argv)
    print(json.dumps(evaluate_run(args.run_dir, split=args.split, workers=args.workers,
                                  timeout_seconds=args.timeout_seconds, scheduler_socket=args.scheduler_socket), indent=2))


if __name__ == "__main__":
    main()
