"""Re-evaluate ACO programs with several evaluation seeds: how much of the training score is seed noise?

usage: PYTHONPATH=. uv run python -m experiments.diagnosis_aco_noise.audit [--workers 6]

For OP and CVRP runs of V10.17-V10.19, every program that set a new training best from
candidate 50 on (the "frontier") and ten randomly drawn valid programs per run (a control
that was not chosen by its score) are evaluated on the training set with the search
condition at eight evaluation seeds: the search seed 730241 and seeds 1-7. A program's
search score is one draw; the mean over the seven other seeds estimates its expected score.
Results are appended to results.jsonl; reruns skip finished programs.
"""

import argparse
import json
import os
from pathlib import Path
import random
import threading
from concurrent.futures import ThreadPoolExecutor

import experiments  # noqa: F401  (sets the evaluation thread limits)
from benchmarks.tasks import training_task
from traceaad.common.evaluation import ProgramEvaluator
from traceaad.common.state import Facts

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("ACO_NOISE_OUT", ROOT / "experiments_result" / "diagnosis_aco_noise"))
SEEDS = (730241, 1, 2, 3, 4, 5, 6, 7)
VERSIONS = ("traceaad_v10_17", "traceaad_v10_18", "traceaad_v10_19")


def programs_to_audit(task):
    rows = []
    for version in VERSIONS:
        for run in sorted((ROOT / "experiments_result" / version / task).glob("*rep*")):
            summary = run / "summary.json"
            if not summary.exists() or json.loads(summary.read_text()).get("status") != "finished":
                continue  # a running search is not audited
            facts = Facts(run)
            valid = sorted(facts.valid.values(), key=lambda p: p["id"])
            best, frontier = None, []
            for p in valid:
                if best is None or p["fitness"] < best["fitness"]:
                    best = p
                    if p["id"] >= 50:
                        frontier.append(p)
            chosen = {p["id"] for p in frontier}
            control = random.Random(f"{run.name}:control").sample([p for p in valid if p["id"] not in chosen],
                                                                   min(10, len(valid) - len(chosen)))
            for p, kind in [(p, "frontier") for p in frontier] + [(p, "control") for p in control]:
                rows.append({"task": task, "version": version, "run": run.name, "id": p["id"], "kind": kind,
                             "archived": p["fitness"], "code": p["code"]})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--tasks", nargs="+", default=["op_aco", "cvrp_aco"])
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "results.jsonl"
    done = {(r["task"], r["run"], r["id"]) for r in map(json.loads, open(path))} if path.exists() else set()
    jobs = [row for task in args.tasks for row in programs_to_audit(task) if (row["task"], row["run"], row["id"]) not in done]
    print(f"{len(jobs)} programs", flush=True)
    local, lock = threading.local(), threading.Lock()

    def run(row):
        cache = local.__dict__.setdefault("evaluators", {})
        if row["task"] not in cache:
            cache[row["task"]] = ProgramEvaluator(training_task(row["task"], condition="traceaad")[0], SEEDS, "audit")
        outcome = cache[row["task"]].evaluate(row["code"], "audit")
        record = {k: v for k, v in row.items() if k != "code"}
        record["scores"] = {str(e["seed"]): e["score"] for e in outcome["evaluations"]}
        record["failure"] = outcome["failure"]["kind"] if outcome["failure"] else None
        with lock, open(path, "a") as f:
            f.write(json.dumps(record) + "\n")

    with ThreadPoolExecutor(args.workers) as pool:
        for future in [pool.submit(run, row) for row in jobs]:
            future.result()


if __name__ == "__main__":
    main()
