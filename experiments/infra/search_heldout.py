"""Evaluate the frozen selected program of a completed current search run."""

import argparse
import json
from pathlib import Path

from benchmarks.tasks import heldout_task, scale_of_split
from traceaad.common.storage import read_json, save_heldout, selected_program
from traceaad.common.evaluation import ProgramEvaluator
from traceaad.common.config import REVISION


def evaluate_run(run_dir, *, split=None, workers=2, timeout_seconds=None):
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
    evaluation = heldout_task(task, split, workers, timeout_seconds)
    seeds = tuple(config["method_params"]["evaluation_seeds"])
    evaluator = ProgramEvaluator(evaluation, seeds, "heldout", measure_calls=config.get("method") != "v1015")
    protocol = evaluator.protocol
    previous = next((r for r in read_json(run_dir / "heldout.json", [])
                     if not r["variant"] and r.get("split") == split), None)
    if previous:
        if (previous["key"] != key or previous.get("node_id") != best["id"]
                or previous.get("task") != task or previous.get("split") != split
                or previous.get("timeout_seconds") != evaluation.timeout_seconds
                or (previous.get("revision") == REVISION and previous["protocol"] != protocol)):
            raise ValueError("existing held-out result belongs to another program or protocol")
        return previous
    outcome = evaluator.evaluate(code, key)
    records = outcome["evaluations"]
    result = {"task": task, "split": split, "scale": str(scale_of_split(task, split)),
              "verification": "verified", "variant": "", "node_id": best["id"],
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
