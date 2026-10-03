"""Equal development for a new design and for the search best at the end of a V10.16 run.

usage: PYTHONPATH=. uv run python -m experiments.diagnosis_v1016_develop.develop [GENERATIONS] [PARALLEL]

Question: late in a search, a new design meets programs that many steps have already
developed. Is it behind because the design is worse, or because it has not been
developed yet? Selection by training score alone cannot tell, so the logs cannot either
(the new programs that were developed were picked for their score). Here the choice is
fixed in advance. For each completed run in ``source/`` the arms are

* ``best``: the run's training best;
* ``new1``, ``new2``: the two best-scoring programs written by Explore after attempt 500
  whose token similarity to the training best is at most 0.5.

Each arm restores the run in its own copy and spends GENERATIONS model generations
(repairs included) on Refine with V10.16's own prompts, evaluation and repair, always
starting from the best program the arm has reached. The arm's best and its starting
program are then evaluated on the run's selection set. The run's frozen identity check
is skipped because this host's environment differs from server3; budgets are counted by
this study, not by the run's configuration.
"""
import experiments  # noqa: F401
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from experiments.infra.base import BACKENDS, build_llm_client, build_task
from experiments.traceaad_v10_16.run import TRAIN_TIMEOUT, selection_task
from traceaad.v10_16 import Config, TraceAADV1016
from traceaad.v10_16.canonical import similarity
from traceaad.v10_16.prompts import ContextTooLong

OUT = Path("experiments_result/diagnosis_v1016_develop")
LATE = 500
MAX_SIMILARITY = 0.5


class Restored(TraceAADV1016):
    def _restore(self, state):
        super()._restore({**state, "identity": self.identity})


def arms(run_dir):
    nodes = []
    with open(run_dir / "search.jsonl") as f:
        for line in f:
            if line[:30].replace(" ", "").startswith('{"kind":"node"'):
                nodes.append(json.loads(line)["data"])
    best = max(nodes, key=lambda n: (n["fitness"], -n["id"]))
    late = [n for n in nodes if n["action"] == "Explore" and n["attempt_id"] > LATE
            and similarity(n["code"], best["code"]) <= MAX_SIMILARITY]
    late.sort(key=lambda n: (-n["fitness"], n["id"]))
    return [("best", best["id"])] + [(f"new{i + 1}", n["id"]) for i, n in enumerate(late[:2])]


def develop(source, arm, start_id, generations, backend):
    task = source.parent.name
    name = f"{source.name}_{arm}"
    record_path = OUT / "arms" / f"{name}.json"
    if record_path.exists():
        return json.loads(record_path.read_text())
    work = OUT / "work" / name
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    for item in ("search.jsonl", "run_config.json"):
        shutil.copy(source / item, work / item)
    run_config = json.loads((source / "run_config.json").read_text())
    evaluation, _ = build_task(task, 4)
    if task in TRAIN_TIMEOUT:
        evaluation.timeout_seconds = TRAIN_TIMEOUT[task]
    profile = BACKENDS[backend]
    llm = build_llm_client(base_url=profile.base_url, model=profile.model, no_proxy=profile.no_proxy,
                           max_tokens=8192)
    config = replace(Config(seed=run_config["seed"]), budget=10 ** 6)
    method = Restored(evaluation=evaluation, llm=llm, run_dir=work, config=config, task=task,
                      selection_evaluation=selection_task(task, evaluation))
    first = method.attempts
    start = method.archive[start_id]
    current, steps, stop = start, [], None
    while method.attempts - first < generations:
        try:
            request = method.prompts.build("Refine", current)
        except ContextTooLong:
            stop = "context_too_long"
            break
        request.update(sampled_action="Refine", fallbacks=[], parent_id=current["id"], reference_id=None,
                       selection=None, reference_selection=None)
        before = method.attempts
        method._attempt(request, parent=current, action="Refine", selection=None)
        for aid in range(before + 1, method.attempts + 1):
            attempt = method.attempts_table[aid]
            steps.append({"generation": aid - first, "action": attempt["action"], "parent": attempt["parent_id"],
                          "status": attempt["status"], "fitness": attempt["fitness"]})
        reached = [method.archive[s] for s in range(first + 1, method.attempts + 1) if s in method.archive]
        current = max([start] + reached, key=lambda n: (n["fitness"], -n["id"]))
    selection = {}
    for label, node in (("start", start), ("reached", current)):
        fitness, _, failure, _, _ = method._evaluate(node["code"], node["key"], role="selection")
        selection[label] = {"id": node["id"], "training": node["fitness"], "selection": fitness, "failure": failure}
    record = {"run": source.name, "task": task, "arm": arm, "start": start_id,
              "generations": method.attempts - first, "stop": stop, "steps": steps, "selection": selection}
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=1))
    llm.close()
    shutil.rmtree(work)
    print(f"{name}: training {start['fitness']:.6g} -> {current['fitness']:.6g}", flush=True)
    return record


def main():
    generations = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    parallel = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    jobs = []
    for source in sorted((OUT / "source").glob("*/*")):
        for arm, start in arms(source):
            jobs.append((source, arm, start))
    print(f"{len(jobs)} arms", flush=True)
    backends = ("server3", "server3b")
    with ThreadPoolExecutor(parallel) as pool:
        futures = [pool.submit(develop, s, a, n, generations, backends[i % 2]) for i, (s, a, n) in enumerate(jobs)]
        for future in futures:
            try:
                future.result()
            except Exception as exc:  # keep the other arms running; the log names the failure
                print(f"arm failed: {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
