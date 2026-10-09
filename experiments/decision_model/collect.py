"""Matched three-action replays; every chain includes at most four candidates."""

import experiments  # noqa: F401
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import threading

from benchmarks.tasks import training_task
from experiments.infra.scheduled_llm import ScheduledLLM
from traceaad.common.config import REVISION
from traceaad.common.storage import append_jsonl, committed_rows, write_json
from traceaad.v10_22 import TraceAADV1022
from traceaad.v10_23 import TraceAADV1023
from .data import ACTIONS, outcome

METHODS = {"v1022": TraceAADV1022, "v1023": TraceAADV1023}


def prompt_identity(policy):
    # Parent builders also supply task, history and repair material.
    return {cls.__module__: hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest()
            for cls in METHODS[policy].PromptBuilder.__mro__ if cls is not object}


def save_manifest(path, manifest):
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("refusing to resume a different collection design")
    write_json(path, manifest)


def initialize_branch(source, work, cutoff):
    work.mkdir(parents=True)
    os.link(source / "programs.jsonl", work / "programs.jsonl")
    last_progress = {}
    for row in committed_rows(source):
        if row.get("kind") == "candidate" and row["attempt"]["id"] > cutoff:
            break
        append_jsonl(work / "events.jsonl", row)
        last_progress = row.get("progress", last_progress)
    write_json(work / "resume.json", {
        "state": {"phase": "search", "model_calls": last_progress.get("model_calls", 0),
                  "attempts": cutoff, "elapsed": 0,
                  "started_at": datetime.now().astimezone().isoformat()},
        "files": {name: (work / name).stat().st_size for name in ("events.jsonl", "programs.jsonl")},
    })


def develop_block(method, parent_id, reference_id, action, cutoff):
    original = method.programs[parent_id]
    while method.attempts < method.config.budget:
        before = method.attempts
        if method.progress.repair_id is not None:
            method._repair()
        else:
            new = [p for i, p in method.archive.items() if i > cutoff]
            parent = min(new, key=lambda p: (p["fitness"], p["id"])) if new else original
            step = action if before == cutoff else "Refine"
            reference = method.programs[reference_id] if step == "Crossover" else None
            request = method.prompts.build(step, parent, reference=reference)
            if request["action"] != step:
                raise ValueError("requested action cannot fit; do not relabel its fallback")
            request.update(sampled_action=step, fallbacks=[], parent_id=parent["id"],
                           reference_id=reference["id"] if reference else None,
                           selection=None, reference_selection=None)
            method._attempt(request, parent=parent, reference=reference)
        if method.attempts <= before or method.attempts > method.config.budget:
            raise RuntimeError("candidate accounting failed")


class Collector:
    def __init__(self, dataset, output, socket, workers, prompt_policy="v1022"):
        self.dataset, self.output, self.socket, self.workers = dataset, output, socket, workers
        self.metadata = json.loads((dataset / "metadata.json").read_text())
        if self.metadata["revision"] != REVISION:
            raise ValueError("collection evaluator revision differs from frozen states")
        self.method_class = METHODS[prompt_policy]
        self.prompt_policy = prompt_policy
        self.local = threading.local()
        self.lock = threading.Lock()

    def evaluation(self, task):
        evaluations = self.local.__dict__.setdefault("evaluations", {})
        if task not in evaluations:
            evaluations[task] = training_task(task, condition="traceaad")[0]
        return evaluations[task]

    def run(self, job):
        state, action, repeat = job
        if state["search_budget"] - state["cutoff"] < 4:
            raise ValueError("frozen state has fewer than four candidates remaining")
        name = f"{state['run_id']}/{state['state_id']}/{action}/rep{repeat}"
        work = self.output / "chains" / name
        result_path = work / "result.json"
        if result_path.exists():
            return json.loads(result_path.read_text())
        source = self.dataset / state["source"]
        if not work.exists():
            initialize_branch(source, work, state["cutoff"])
        llm = ScheduledLLM(self.socket, max_tokens=8192, label="decision-data:" + name)
        config = self.method_class.Config(budget=state["cutoff"] + 4, seed=repeat,
                                          scheduler_socket=self.socket, eval_workers=4)
        write_json(work / "run_config.json", {"result_format": "traceaad-results-v2", "task": state["task"],
                    "origin": state, "action": action, "repeat": repeat, "candidate_budget": 4,
                    "method_params": asdict(config), "revision": self.metadata["revision"],
                    "method": self.prompt_policy, "prompt_sources_sha256": prompt_identity(self.prompt_policy)})
        method = self.method_class(evaluation=self.evaluation(state["task"]), llm=llm,
                                   run_dir=work, config=config, task=state["task"])
        try:
            if method.attempts < state["cutoff"] or method.attempts > config.budget:
                raise ValueError("branch history does not match frozen cutoff and budget")
            develop_block(method, state["parent_id"], state["reference_id"], action, state["cutoff"])
            first = method.attempts_table[state["cutoff"] + 1]
            node = method.programs.get(first["program_id"]) if first["status"] == "valid" else None
            first_fitnesses = [node["fitness"]] if node and node["id"] > state["cutoff"] else []
            all_fitnesses = [p["fitness"] for i, p in method.archive.items() if i > state["cutoff"]]
            scale = self.metadata["score_scales"][state["task"]]
            gold1, gain1 = outcome(state["parent_fitness"], state["frontier_fitness"], first_fitnesses, scale)
            gold4, gain4 = outcome(state["parent_fitness"], state["frontier_fitness"], all_fitnesses, scale)
            result = {"job": name, "state": state, "action": action, "repeat": repeat,
                      "prompt_policy": self.prompt_policy,
                      "candidates": method.attempts - state["cutoff"], "gold": {"1": gold1, "4": gold4},
                      "normalized_gain": {"1": gain1, "4": gain4},
                      "first_status": first["status"],
                      "steps": [{"attempt": i, "action": a["action"], "status": a["status"],
                                 "parent_id": a["parent_id"], "program_id": a["program_id"]}
                                for i, a in method.attempts_table.items() if i > state["cutoff"]]}
            write_json(result_path, result)
            with self.lock:
                append_jsonl(self.output / "results.jsonl", result)
            # Keep decision-test outcomes out of progress logs inspected during development.
            print(f"{name}: split={state['split']}, candidates={result['candidates']}", flush=True)
            return result
        finally:
            llm.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scheduler-socket", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--candidate-cap", type=int, default=28800)
    parser.add_argument("--prompt-policy", choices=tuple(METHODS), default="v1022")
    args = parser.parse_args()
    states = [json.loads(line) for line in (args.dataset / "states.jsonl").read_text().splitlines()]
    if args.workers < 1 or args.repeats < 2 or len(states) * 3 * args.repeats * 4 > args.candidate_cap:
        parser.error("invalid workers/repeats or campaign exceeds candidate cap")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"states_sha256": hashlib.sha256((args.dataset / "states.jsonl").read_bytes()).hexdigest(),
                "repeats": args.repeats, "actions": list(ACTIONS), "candidate_budget": 4,
                "candidate_cap": args.candidate_cap, "max_candidates": len(states) * 3 * args.repeats * 4,
                "prompt_policy": args.prompt_policy, "prompt_sources_sha256": prompt_identity(args.prompt_policy),
                "protocol_revision": REVISION, "execution_order": "shuffled state blocks; decision-test states last"}
    save_manifest(args.output / "manifest.json", manifest)
    rng = random.Random(20261009)
    rng.shuffle(states)
    states.sort(key=lambda state: state["split"] == "test")
    collector = Collector(args.dataset, args.output, args.scheduler_socket, args.workers, args.prompt_policy)
    with ThreadPoolExecutor(args.workers) as pool:
        for state in states:
            jobs = [(state, action, repeat) for action in ACTIONS for repeat in range(args.repeats)]
            rng.shuffle(jobs)
            for result in pool.map(collector.run, jobs):
                pass
    write_json(args.output / "complete.json", manifest)


if __name__ == "__main__":
    main()
