"""Fixed-parent replay before the V10.20 batch: does Deepen keep the rule and add search?

usage: PYTHONPATH=. uv run python -m experiments.diagnosis_v1020_replay.replay [--samples 8] [--workers 6]
       [--tasks op_aco cvrp_aco ...] [--arms R0 R1 D]

For each task one strong program from a finished local run is the parent (local runs, so
the evaluation times in the context match this host). Its context is the run's committed
facts before the last Refine attempt that started from it, so every arm sees the state of
an actual decision. Three arms write the next generation from that state:

- R0: V10.17 Refine (V10.17 text);
- R1: V10.20 Refine (V10.20 facts about computation);
- D:  V10.20 Deepen (same facts, Deepen goal).

Two control arms separate why D adds little computation (run after the first three):

- Dx:   D plus one sentence naming concrete searches for the task family (diagnostic only:
        V10.20's prompts do not list methods). If Dx adds computation and holds the parent's
        quality, the goal wording is the limit; if not, writing such a search is.
- Dmin: D with only the task, the evaluation and the current algorithm, without its formation
        path and attempts. If Dmin adds more computation than D, the history of small edits
        anchors the step.

Each child is evaluated with the search evaluator of the task (training set, training time
limit, evaluation seed 730241) on this host. The parent is evaluated again in the same pool,
before and after the children, so a child is compared with its parent under the same load.
Results are appended to ``results.jsonl``; reruns skip finished samples. ``analyze.py``
summarizes them.
"""

import argparse
import json
import os
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor

import experiments  # noqa: F401  (sets the evaluation thread limits)
from benchmarks.tasks import TASKS, training_task
from core.llm import ModelCallError, generate
from experiments.infra.base import BACKENDS, build_llm_client
from traceaad.common.canonical import canonical, key
from traceaad.common.delivery import DeliveryError, SourceError, parse_response
from traceaad.common.evaluation import ProgramEvaluator
from traceaad.common.state import Facts
from traceaad.v10_17 import Config as V1017Config
from traceaad.v10_17.prompts import PromptBuilder as V1017Prompts
from traceaad.v10_20 import Config
from traceaad.v10_20.prompts import PromptBuilder

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("V1020_REPLAY_OUT", ROOT / "experiments_result" / "diagnosis_v1020_replay"))
PARENTS = {  # task: (experiment, repeat); the parent is that run's best training program
    "tsp_construct": ("traceaad_v10_19", "rep3"),
    "cvrp_aco": ("traceaad_v10_19", "rep2"),
    "op_aco": ("traceaad_v10_19", "rep1"),
    "online_bin_packing": ("traceaad_v10_18", "rep1"),
    "vrptw_construct": ("traceaad_v10_19", "rep3"),
}
ARMS = {"R0": ("v1017", "Refine"), "R1": ("v1020", "Refine"), "D": ("v1020", "Deepen"),
        "Dx": ("v1020", "Deepen"), "Dmin": ("v1020", "Deepen")}
EXAMPLES = {  # Dx only
    "op_aco": "aco", "cvrp_aco": "aco", "tsp_construct": "construct", "vrptw_construct": "construct"}
EXAMPLE_TEXT = {
    "aco": ("For example, the function can construct and improve complete solutions guided by the current matrix, "
            "or run an ant colony with the current matrix for many more iterations than the evaluator does, and "
            "return a matrix that strongly favors the edges of the best solution it found."),
    "construct": ("For example, for each candidate the function can complete the remaining solution with the current "
                  "rule and choose the candidate whose completed solution is best."),
}
SEED = 730241


def run_dir(task):
    experiment, repeat = PARENTS[task]
    found = sorted((ROOT / "experiments_result" / experiment / task).glob(f"*_{repeat}"))
    if len(found) != 1:
        raise FileNotFoundError(f"{experiment}/{task}/{repeat}: {found}")
    return found[0]


def decision_state(task):
    """The parent and the facts committed before the last Refine attempt that started from it."""
    facts = Facts(run_dir(task))
    parent = min(facts.valid.values(), key=lambda p: (p["fitness"], p["id"]))
    refines = [a["id"] for a in facts.attempts.values()
               if a["parent_id"] == parent["id"] and a["action"] == "Refine" and a.get("repair_of") is None]
    cutoff = max(refines) - 1 if refines else max(facts.attempts)
    programs = {i: p for i, p in facts.programs.items() if i <= cutoff}
    attempts = {i: a for i, a in facts.attempts.items() if i <= cutoff}
    return parent, programs, attempts, cutoff


def merge(path, entries):
    data = json.load(open(path)) if path.exists() else {}
    data.update(entries)
    with open(path, "w") as f:
        json.dump(data, f, indent=1)


class Study:
    def __init__(self, tasks):
        self.llms = [build_llm_client(base_url=BACKENDS[b].base_url, model=BACKENDS[b].model,
                                      no_proxy=BACKENDS[b].no_proxy, max_tokens=8192) for b in ("server3", "server3b")]
        self.local = threading.local()
        self.lock = threading.Lock()
        self.states, self.prompts = {}, {}
        for task in tasks:
            parent, programs, attempts, cutoff = decision_state(task)
            evaluation, _ = training_task(task, condition="traceaad")
            builders = {"v1017": V1017Prompts(self.llms[0], task, evaluation, programs, attempts, V1017Config()),
                        "v1020": PromptBuilder(self.llms[0], task, evaluation, programs, attempts, Config())}
            self.states[task] = {"parent": parent, "cutoff": cutoff, "template": str(evaluation.template_program)}
            for arm, (version, action) in ARMS.items():
                builder = builders[version]
                if arm == "Dmin":
                    sections = builder.common + [builder._current(parent), builder.DEEPEN, builder.output_format("Deepen")]
                    self.prompts[task, arm] = builder._result(sections, "Deepen")
                    continue
                request = builder.build(action, parent)
                if arm == "Dx" and task in EXAMPLES:
                    request["prompt"] = request["prompt"].replace(
                        builder.DEEPEN, builder.DEEPEN + " " + EXAMPLE_TEXT[EXAMPLES[task]])
                self.prompts[task, arm] = request
        OUT.mkdir(parents=True, exist_ok=True)
        merge(OUT / "prompts.json", {f"{task}/{arm}": {"prompt": r["prompt"], "input_tokens": r["input_tokens"],
                                                      "trims": r["trims"]} for (task, arm), r in self.prompts.items()})
        merge(OUT / "parents.json", {task: {"run": str(run_dir(task).relative_to(ROOT)), "parent_id": s["parent"]["id"],
                                            "cutoff": s["cutoff"], "fitness": s["parent"]["fitness"],
                                            "eval_seconds": s["parent"]["eval_seconds"]} for task, s in self.states.items()})

    def evaluator(self, task):
        cache = self.local.__dict__.setdefault("evaluators", {})
        if task not in cache:
            cache[task] = ProgramEvaluator(training_task(task, condition="traceaad")[0], (SEED,), "search")
        return cache[task]

    def evaluate(self, task, code):
        outcome = self.evaluator(task).evaluate(code, key(code))
        failure = outcome["failure"]
        return {"status": failure["kind"] if failure else "valid", "fitness": outcome["fitness"],
                "eval_seconds": outcome["seconds"], "calls": outcome["calls"],
                "function_seconds": outcome["function_seconds"],
                "error": (failure["error"] or "")[-300:] if failure else None}

    def run(self, job, index):
        task, arm, sample = job
        record = {"task": task, "arm": arm, "sample": sample, "parent_id": self.states[task]["parent"]["id"]}
        if arm == "parent":
            record.update(self.evaluate(task, self.states[task]["parent"]["code"]))
        else:
            request = self.prompts[task, arm]
            try:
                details = generate(self.llms[index % 2], request["prompt"], max_tokens=8192)
            except ModelCallError as exc:
                record.update(status="model_error", error=str(exc)[:300])
            else:
                record.update(response=details.get("content", ""), finish_reason=details.get("finish_reason"),
                              usage=details.get("usage"), input_tokens=request["input_tokens"])
                try:
                    code, idea, _ = parse_response(record["response"], record["finish_reason"],
                                                   self.states[task]["template"])
                except DeliveryError as exc:
                    record.update(status="delivery_failed", error=str(exc)[:300])
                except SourceError as exc:
                    record.update(status="invalid_source", error=str(exc)[:300], code=exc.code)
                else:
                    try:
                        code = canonical(code)
                    except (SyntaxError, ValueError):
                        pass
                    record.update(idea=idea, code=code, **self.evaluate(task, code))
        with self.lock, open(OUT / "results.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"{task} {arm} {sample}: {record.get('status')} {record.get('fitness')} "
              f"{record.get('eval_seconds') or 0:.1f}s", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    args = parser.parse_args(argv)
    study = Study(args.tasks)
    done = set()
    if (OUT / "results.jsonl").exists():
        done = {(r["task"], r["arm"], r["sample"]) for r in map(json.loads, open(OUT / "results.jsonl"))}
    children = [(task, arm, s) for s in range(args.samples) for task in args.tasks for arm in args.arms]
    parents = [[(task, "parent", s) for task in args.tasks] for s in sorted({0, args.samples - 1})]
    jobs = [job for job in parents[0] + children + parents[-1] if job not in done]
    jobs = list(dict.fromkeys(jobs))
    print(f"{len(jobs)} jobs", flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        for future in [pool.submit(study.run, job, i) for i, job in enumerate(jobs)]:
            future.result()


if __name__ == "__main__":
    main()
