"""Fixed-state replay: do the Explore and Crossover goals change what the model writes?

usage: PYTHONPATH=. uv run python -m experiments.diagnosis_v1021_replay.replay [--samples 8] [--workers 18]

Decision states come from the stopped V10.21 runs (repeats 1 and 2 of each task): the
latest Explore (or Crossover) attempt whose starting program is valid under the current
5 s per-instance limit, with the run's facts committed before that attempt. Every arm
uses V10.21's prompts and the same material; only the goal and its Analysis line differ.

- E0: Explore as in V10.21, "scores better than the best found so far by changing how the
      current algorithm makes its decisions, not by tuning it".
- E1: the same, measured against the current algorithm instead of the search's best.
- C0: Crossover as in V10.21, "bring in what the reference algorithm does well".
- C1: decide which computations of the reference, if any, can improve the current one.

Children are evaluated on the training set under the current execution (5 s per instance,
instances scheduled on this host's CPU pool, seed 730241). Each starting program is
evaluated once in the same way. Results go to ``results.jsonl``; reruns skip finished jobs.
"""

import argparse
import json
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor

import experiments  # noqa: F401  (evaluation thread limits)
from benchmarks.tasks import CO_TASKS, INSTANCE_SECONDS, training_task
from core.llm import ModelCallError, generate
from experiments.infra.base import BACKENDS, build_llm_client
from traceaad.common.canonical import canonical, key
from traceaad.common.delivery import DeliveryError, SourceError, parse_response
from traceaad.common.instance_evaluation import InstanceProgramEvaluator
from traceaad.common.prompts import ANALYSIS, KNOWN
from traceaad.common.state import Facts
from traceaad.v10_21 import Config
from traceaad.v10_21.prompts import PromptBuilder

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments_result" / "diagnosis_v1021_replay"
SOCKET = "/tmp/traceaad-1000/scheduler.sock"
SEED = 730241

E1 = """[Your Task: Explore]
Write an algorithm that scores better than the current algorithm by changing how it makes its decisions, not by tuning it."""
C1 = f"""[Your Task: Crossover]
Decide which computations of the reference algorithm, if any, can improve the current algorithm, bring them in while keeping the current algorithm's core idea, and write a version that scores better than it. {KNOWN}"""
C1_ANALYSIS = ("which computations of the reference algorithm, if any, could improve the current algorithm, "
               "and how to bring them in while keeping the current algorithm's core idea")


class E1Prompts(PromptBuilder):
    EXPLORE = E1


class C1Prompts(PromptBuilder):
    CROSSOVER = C1
    ANALYSIS = {**ANALYSIS, "Crossover": C1_ANALYSIS}


ARMS = {"E0": (PromptBuilder, "Explore"), "E1": (E1Prompts, "Explore"),
        "C0": (PromptBuilder, "Crossover"), "C1": (C1Prompts, "Crossover")}


def run_dir(task, repeat):
    return next((ROOT / "experiments_result" / "traceaad_v10_21" / task).glob(f"*_rep{repeat}"))


class Study:
    def __init__(self, tasks, repeats):
        self.llms = [build_llm_client(base_url=BACKENDS[b].base_url, model=BACKENDS[b].model,
                                      no_proxy=BACKENDS[b].no_proxy, max_tokens=8192) for b in ("server3", "server3b")]
        self.lock, self.local = threading.Lock(), threading.local()
        self.states, self.prompts = {}, {}
        OUT.mkdir(parents=True, exist_ok=True)
        for task in tasks:
            evaluation, _ = training_task(task, condition="traceaad")
            for repeat in repeats:
                facts = Facts(run_dir(task, repeat))
                for action in ("Explore", "Crossover"):
                    state = self.state(task, facts, action)
                    name = f"{task}/rep{repeat}/{action}"
                    self.states[name] = state
                    for arm, (builder_class, arm_action) in ARMS.items():
                        if arm_action != action:
                            continue
                        builder = builder_class(self.llms[0], task, evaluation, state["programs"], state["attempts"], Config())
                        request = builder.build(action, state["parent"], reference=state["reference"])
                        self.prompts[name, arm] = request
        with open(OUT / "states.json", "w") as f:
            json.dump({name: {"parent_id": s["parent"]["id"], "reference_id": s["reference"]["id"] if s["reference"] else None,
                              "attempt": s["attempt"], "parent_fitness": s["parent_fitness"],
                              "search_best": s["search_best"]} for name, s in self.states.items()}, f, indent=1)
        with open(OUT / "prompts.json", "w") as f:
            json.dump({f"{name}/{arm}": r["prompt"] for (name, arm), r in self.prompts.items()}, f, indent=1)

    def evaluator(self, task):
        cache = self.local.__dict__.setdefault("evaluators", {})
        if task not in cache:
            cache[task] = InstanceProgramEvaluator(training_task(task, condition="traceaad")[0], (SEED,), "search",
                                                   timeout_seconds=INSTANCE_SECONDS, n_workers=16, scheduler_socket=SOCKET)
        return cache[task]

    def evaluate(self, task, code):
        outcome = self.evaluator(task).evaluate(code, key(code))
        failure = outcome["failure"]
        return {"status": failure["kind"] if failure else "valid", "fitness": outcome["fitness"],
                "error": (failure["error"] or "")[-300:] if failure else None}

    def state(self, task, facts, action):
        """The latest attempt of this step whose starting program is valid under the current limit."""
        cache = {}
        for attempt in sorted(facts.attempts.values(), key=lambda a: -a["id"]):
            if attempt["action"] != action or attempt.get("repair_of") is not None:
                continue
            parent = facts.programs.get(attempt["parent_id"])
            reference = facts.programs.get(attempt.get("reference_id")) if action == "Crossover" else None
            if parent is None or not parent["valid"] or (action == "Crossover" and reference is None):
                continue
            if parent["id"] not in cache:
                cache[parent["id"]] = self.evaluate(task, parent["code"])
            measured = cache[parent["id"]]
            if measured["status"] != "valid":
                continue
            cutoff = attempt["id"] - 1
            programs = {i: p for i, p in facts.programs.items() if i <= cutoff}
            valid = [p for p in programs.values() if p["valid"]]
            return {"parent": parent, "reference": reference, "attempt": attempt["id"],
                    "programs": programs, "attempts": {i: a for i, a in facts.attempts.items() if i <= cutoff},
                    "parent_fitness": measured["fitness"], "search_best": min(p["fitness"] for p in valid),
                    "template": str(training_task(task, condition="traceaad")[0].template_program)}
        raise ValueError(f"{task}: no {action} state with a valid starting program")

    def run(self, job, index):
        name, arm, sample = job
        task = name.split("/")[0]
        state = self.states[name]
        record = {"state": name, "task": task, "arm": arm, "sample": sample}
        try:
            details = generate(self.llms[index % 2], self.prompts[name, arm]["prompt"], max_tokens=8192)
        except ModelCallError as exc:
            record.update(status="model_error", error=str(exc)[:300])
        else:
            record.update(response=details.get("content", ""), usage=details.get("usage"))
            try:
                code, idea, _ = parse_response(record["response"], details.get("finish_reason"), state["template"])
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
        print(f"{name} {arm} {sample}: {record.get('status')} {record.get('fitness')}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--tasks", nargs="+", default=list(CO_TASKS), choices=list(CO_TASKS))
    parser.add_argument("--repeats", nargs="+", type=int, default=[1, 2])
    args = parser.parse_args(argv)
    study = Study(args.tasks, args.repeats)
    done = set()
    if (OUT / "results.jsonl").exists():
        done = {(r["state"], r["arm"], r["sample"]) for r in map(json.loads, open(OUT / "results.jsonl"))}
    jobs = [(name, arm, s) for s in range(args.samples) for (name, arm) in study.prompts if (name, arm, s) not in done]
    print(f"{len(jobs)} jobs", flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        for future in [pool.submit(study.run, job, i) for i, job in enumerate(jobs)]:
            future.result()


if __name__ == "__main__":
    main()
