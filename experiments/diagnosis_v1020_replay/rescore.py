"""Re-evaluate the ACO children and parents of the replay at eight evaluation seeds.

usage: PYTHONPATH=. uv run python -m experiments.diagnosis_v1020_replay.rescore [--workers 6]

On OP and CVRP one training evaluation is one ACO run per instance: re-running the same program
with another seed moves its score by about 1% (OP) and 0.6% (CVRP), and the parent, being the best
of its run at the search seed, is a lucky draw. The seed-mean over 730241 and 1-7 compares a child
with its parent without that bias. Writes rescored.jsonl.
"""

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import experiments  # noqa: F401  (sets the evaluation thread limits)
from benchmarks.tasks import training_task
from traceaad.common.evaluation import ProgramEvaluator
from traceaad.common.state import Facts
from experiments.diagnosis_v1020_replay.replay import OUT, run_dir

SEEDS = (730241, 1, 2, 3, 4, 5, 6, 7)
ACO = ("op_aco", "cvrp_aco")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)
    parents = json.load(open(OUT / "parents.json"))
    rows = [json.loads(line) for line in open(OUT / "results.jsonl")]
    path = OUT / "rescored.jsonl"
    done = {(r["task"], r["arm"], r["sample"]) for r in map(json.loads, open(path))} if path.exists() else set()
    jobs = [{"task": t, "arm": "parent", "sample": 0,
             "code": Facts(run_dir(t)).programs[parents[t]["parent_id"]]["code"]} for t in ACO if t in parents]
    jobs += [r for r in rows if r["task"] in ACO and r["arm"] != "parent" and r.get("status") == "valid"]
    jobs = [j for j in jobs if (j["task"], j["arm"], j["sample"]) not in done]
    local, lock = threading.local(), threading.Lock()

    def run(job):
        cache = local.__dict__.setdefault("evaluators", {})
        if job["task"] not in cache:
            cache[job["task"]] = ProgramEvaluator(training_task(job["task"], condition="traceaad")[0], SEEDS, "rescore")
        outcome = cache[job["task"]].evaluate(job["code"], "rescore")
        record = {"task": job["task"], "arm": job["arm"], "sample": job["sample"],
                  "scores": {str(e["seed"]): e["score"] for e in outcome["evaluations"]},
                  "failure": outcome["failure"]["kind"] if outcome["failure"] else None}
        with lock, open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(job["task"], job["arm"], job["sample"], outcome["fitness"], flush=True)

    with ThreadPoolExecutor(args.workers) as pool:
        for future in [pool.submit(run, job) for job in jobs]:
            future.result()


if __name__ == "__main__":
    main()
