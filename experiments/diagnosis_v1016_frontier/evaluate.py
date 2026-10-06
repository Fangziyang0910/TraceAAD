"""Does training-score progress carry over to independent instances, late in the search too?

usage: PYTHONPATH=. uv run python -m experiments.diagnosis_v1016_frontier.evaluate [WORKERS]

``frontier.json`` lists, for every V10.15-6 and V10.16 run, each program that raised
the run's best training score (extracted from the server3 logs). Each is evaluated on
the run's selection set (the protocol of experiments/traceaad_v10_16/run.py, identical
to V10.15) with the selection time limit. Results are appended to ``selection.jsonl``;
reruns skip finished programs.
"""
import experiments  # noqa: F401
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from core import SecureEvaluator
from experiments.infra.base import build_task
from benchmarks.tasks import selection_task
from traceaad.common.evaluation import SeededEvaluation

OUT = "experiments_result/diagnosis_v1016_frontier"
LOCAL = threading.local()


def evaluator(task):
    cache = LOCAL.__dict__.setdefault("cache", {})
    if task not in cache:
        search, _ = build_task(task, 4, condition="traceaad")
        selection = selection_task(task, search)
        cache[task] = SecureEvaluator(SeededEvaluation(selection))
    return cache[task]


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    programs = json.load(open(f"{OUT}/frontier.json"))
    path = f"{OUT}/selection.jsonl"
    done = set()
    if os.path.exists(path):
        done = {(r["run"], r["id"]) for r in map(json.loads, open(path))}
    jobs = [p for p in programs if (p["run"], p["id"]) not in done]
    print(f"{len(jobs)} programs to evaluate", flush=True)
    lock = threading.Lock()

    def run(p):
        secure = evaluator(p["task"])
        result = secure.evaluate_program_with_details(p['code'], seed=730241)
        value = result.result
        score = value["score"] if isinstance(value, dict) else None
        record = {"run": p["run"], "id": p["id"], "selection_fitness": score,
                  "failure": None if score is not None else (result.failure_kind or "error")}
        with lock:
            with open(path, "a") as f:
                f.write(json.dumps(record) + "\n")
        print(p["run"], p["id"], score, flush=True)

    # ACO tasks already use four evaluation workers each; run them with fewer threads.
    light = [p for p in jobs if p["task"] not in ("cvrp_aco", "op_aco")]
    heavy = [p for p in jobs if p["task"] in ("cvrp_aco", "op_aco")]
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(run, light))
    with ThreadPoolExecutor(max(1, workers // 4)) as pool:
        list(pool.map(run, heavy))


if __name__ == "__main__":
    main()
