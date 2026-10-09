"""Fixed-state replay: does Explore repeat a failed direction because it sees only successes?

usage: PYTHONPATH=. .venv/bin/python -m experiments.diagnosis_explore_context.replay [--samples 5] [--workers 12]

States: in each of the six TSP runs that built a completion simulation (local P-core and
server3 batches of 2026-10-09), the original Explore attempts closest to attempts 400, 600
and 800, with the run's facts committed before each attempt. Arms differ only in the
global section that follows the current algorithm's own attempts:

- A: V10.21 as run, "How the Best Score Improved in This Search" (changes that set a new best).
- B: no list; only the line stating the best score found so far.
- C: the most recent Explore attempts of the whole search, from any algorithm, with their
     outcomes relative to their own starting algorithm, and the best-score line.

Children are evaluated once on the training set under the current protocol (2 s CPU per
instance inside the function, 20 s per instance) on this host's P-core pool. Results go to
``results.jsonl``; reruns skip finished jobs.
"""

import argparse
import gzip
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import experiments  # noqa: F401  (evaluation thread limits)
from benchmarks.tasks import INSTANCE_SECONDS, training_task
from core.llm import ModelCallError, generate
from experiments.infra.base import BACKENDS, build_llm_client
from traceaad.common.canonical import canonical, key
from traceaad.common.delivery import DeliveryError, SourceError, parse_response
from traceaad.common.history import final_attempt, score_text
from traceaad.common.instance_evaluation import InstanceProgramEvaluator
from traceaad.common.state import Facts
from traceaad.v10_21 import Config
from traceaad.v10_21.prompts import PromptBuilder

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments_result" / "diagnosis_explore_context"
RUNS = ROOT / "experiments_result" / "traceaad_v10_21" / "tsp_construct"
BATCHES = ("20261009_cpu_local_p", "20261009_cpu_server3")
TARGETS = (400, 600, 800)
SOCKET = "/tmp/traceaad-1000/scheduler.sock"
SEED = 730241
TASK = "tsp_construct"


def best_line(programs):
    best = min((p for p in programs.values() if p["valid"]), key=lambda p: (p["fitness"], p["id"]))
    return f"Best score found so far in this search: {score_text(best['score'])}."


class NoList(PromptBuilder):
    def _progress_section(self, shown):
        return best_line(self.programs), []


class RecentExplores(PromptBuilder):
    def _progress_section(self, shown):
        explores = [a for a in sorted(self.attempts.values(), key=lambda a: a["id"])
                    if a["action"] == "Explore" and a.get("repair_of") is None
                    and not (a.get("exploration") or {}).get("step")
                    and a["parent_id"] in self.programs][-shown:] if shown else []
        text = "[Recent Explore Attempts in This Search]\n"
        if explores:
            text += (f"The most recent {len(explores)} Explore attempts in this search, from any algorithm, "
                     "oldest first, with their evaluated outcomes relative to the algorithm each started from.\n\n")
            entries = []
            for attempt in explores:
                start = self.programs[attempt["parent_id"]]
                outcome = self._outcome(attempt, start, set())
                repair = final_attempt(attempt, self.attempts)
                if repair["id"] != attempt["id"]:
                    outcome += "; repaired: " + self._outcome(repair, start, set())
                entries.append(f"Attempt {attempt['id']} · Explore from an algorithm scoring {score_text(start['score'])}"
                               f" · Design: {' '.join((attempt.get('idea') or '(none)').split())}\n  → {outcome}")
            text += "\n\n".join(entries) + "\n\n"
        return text + best_line(self.programs), [a["id"] for a in explores]


ARMS = {"A": PromptBuilder, "B": NoList, "C": RecentExplores}


def explore_states(facts):
    original = sorted(a["id"] for a in facts.attempts.values()
                      if a["action"] == "Explore" and a.get("repair_of") is None
                      and not (a.get("exploration") or {}).get("step")
                      and facts.programs.get(a["parent_id"], {}).get("valid"))
    chosen = []
    for target in TARGETS:
        pick = min((i for i in original if i not in chosen), key=lambda i: abs(i - target))
        chosen.append(pick)
    return chosen


class Study:
    def __init__(self):
        self.llms = [build_llm_client(base_url=BACKENDS[b].base_url, model=BACKENDS[b].model,
                                      no_proxy=BACKENDS[b].no_proxy, max_tokens=8192) for b in ("server3", "server3b")]
        self.lock, self.local = threading.Lock(), threading.local()
        self.template = str(training_task(TASK, condition="traceaad")[0].template_program)
        evaluation = training_task(TASK, condition="traceaad")[0]
        self.states, self.prompts = {}, {}
        OUT.mkdir(parents=True, exist_ok=True)
        mismatches = []
        for run in sorted(p for p in RUNS.iterdir() if p.name.startswith(BATCHES)):
            facts = Facts(run)
            logged = self.logged_prompts(run)
            for attempt_id in explore_states(facts):
                cutoff = attempt_id - 1
                programs = {i: p for i, p in facts.programs.items() if i <= cutoff}
                attempts = {i: a for i, a in facts.attempts.items() if i <= cutoff}
                parent = programs[facts.attempts[attempt_id]["parent_id"]]
                name = f"{run.name.split('_tsp_')[0].removeprefix('20261009_cpu_')}_r{run.name[-1]}/{attempt_id}"
                self.states[name] = {"parent_id": parent["id"], "parent_fitness": parent["fitness"],
                                     "search_best": min(p["fitness"] for p in programs.values() if p["valid"]),
                                     "parent_code": parent["code"]}
                for arm, builder_class in ARMS.items():
                    builder = builder_class(self.llms[0], TASK, evaluation, programs, attempts, Config())
                    self.prompts[name, arm] = builder.build("Explore", parent)["prompt"]
                if self.prompts[name, "A"] != logged.get(attempt_id):
                    mismatches.append(name)
        if mismatches:
            raise RuntimeError(f"arm A differs from the logged prompt: {mismatches}")
        with open(OUT / "states.json", "w") as f:
            json.dump({n: {k: v for k, v in s.items() if k != "parent_code"} for n, s in self.states.items()}, f, indent=1)
        with open(OUT / "prompts.json", "w") as f:
            json.dump({f"{name}/{arm}": p for (name, arm), p in self.prompts.items()}, f, indent=1)

    @staticmethod
    def logged_prompts(run):
        path = run / "calls.jsonl.gz"
        opener = gzip.open if path.exists() else open
        path = path if path.exists() else run / "calls.jsonl"
        with opener(path, "rt") as f:
            return {c["request_id"]: c["prompt"] for c in map(json.loads, f)}

    def evaluator(self):
        if not hasattr(self.local, "evaluator"):
            self.local.evaluator = InstanceProgramEvaluator(training_task(TASK, condition="traceaad")[0], (SEED,), "search",
                                                            timeout_seconds=INSTANCE_SECONDS, n_workers=8,
                                                            scheduler_socket=SOCKET)
        return self.local.evaluator

    def evaluate(self, code):
        outcome = self.evaluator().evaluate(code, key(code))
        failure = outcome["failure"]
        return {"status": failure["kind"] if failure else "valid", "fitness": outcome["fitness"],
                "error": (failure["error"] or "")[-300:] if failure else None}

    def run(self, job, index):
        name, arm, sample = job
        record = {"state": name, "arm": arm, "sample": sample}
        try:
            details = generate(self.llms[index % 2], self.prompts[name, arm], max_tokens=8192)
        except ModelCallError as exc:
            record.update(status="model_error", error=str(exc)[:300])
        else:
            record.update(response=details.get("content", ""), usage=details.get("usage"))
            try:
                code, idea, _ = parse_response(record["response"], details.get("finish_reason"), self.template)
            except DeliveryError as exc:
                record.update(status="delivery_failed", error=str(exc)[:300])
            except SourceError as exc:
                record.update(status="invalid_source", error=str(exc)[:300], code=exc.code)
            else:
                try:
                    code = canonical(code)
                except (SyntaxError, ValueError):
                    pass
                record.update(idea=idea, code=code, **self.evaluate(code))
        with self.lock, open(OUT / "results.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"{name} {arm} {sample}: {record.get('status')} {record.get('fitness')}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--prompts-only", action="store_true")
    args = parser.parse_args(argv)
    study = Study()
    if args.prompts_only:
        print(f"{len(study.states)} states, {len(study.prompts)} prompts")
        return
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
