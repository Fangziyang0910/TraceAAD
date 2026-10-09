"""Initial-program replay: does the Init instruction decide which decision structures enter the search?

usage: PYTHONPATH=. .venv/bin/python -m experiments.diagnosis_init_prompt.replay [--samples 30] [--workers 12]

All nine TSP runs of 2026-10-09 started from 72 programs that all score candidates; the
V9.14 runs, whose three best programs all plan the remaining route and take its first
step, used a different first prompt. Same model and sampling; only the prompt differs:

- N: V10.21's first Init prompt, as the search sends it.
- M: the same prompt with V9.14's instruction ("Create one complete, valid, and
     competitive initial algorithm. Avoid a placeholder or trivial baseline.").
- O: V9.14's root prompt, rebuilt from its source (commit b5bd6bbf): its template and
     docstring, its fitness line and its Idea/Code output contract.
- N1, O1: prompts N and O sampled as every run did until 2026-09-30 (temperature 1.0,
     top_p 0.95, no presence penalty; thinking disabled), instead of the current profile
     (temperature 0.7, top_p 0.8, presence penalty 1.5).

Programs are evaluated once on the training set under the current protocol (2 s CPU per
instance inside the function, 20 s per instance) on this host's P-core pool.
"""

import argparse
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
from traceaad.common.instance_evaluation import InstanceProgramEvaluator
from traceaad.common.prompts import INITIAL
from traceaad.v10_21 import Config
from traceaad.v10_21.prompts import PromptBuilder

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments_result" / "diagnosis_init_prompt"
SOCKET = "/tmp/traceaad-1000/scheduler.sock"
SEED = 730241
TASK = "tsp_construct"

V914_INSTRUCTION = ("Create one complete, valid, and competitive initial algorithm.\n"
                    "Avoid a placeholder or trivial baseline.")
V914_TASK = (
    "The Traveling Salesman Problem asks for a shortest tour that visits each node once and returns to the start. "
    "Instances are generated from node coordinates, but the constructive heuristic does not receive coordinates. "
    "At each step it receives the current node id, the destination/start node id, an array of unvisited candidate "
    "node ids, and the pairwise distance matrix, and must return the id of the next node to visit. "
    "Help me design a novel algorithm to select the next node in each step.")
V914_TARGET = '''def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    """
    Design a novel algorithm to select the next node in each step.

    Args:
    current_node: ID of the current node.
    destination_node: ID of the destination node.
    unvisited_nodes: Array of IDs of unvisited nodes.
    distance_matrix: Distance matrix of nodes.

    Return:
    ID of the next node to visit.
    """'''
V914_PROMPT = "\n".join([
    "[Task]", V914_TASK, "Fitness is this task's metric and lower is better.", "",
    "[Target Function]", V914_TARGET,
    "Keep the function name, arguments, return type, and contract unchanged.",
    "Include every required import and helper in the returned program.", "",
    "[Instruction]", V914_INSTRUCTION,
    "Output one concise Idea and one complete Python program:",
    "Idea: <one sentence, within 500 characters>", "Code:", "```python",
    "<complete executable implementation>", "```"])


class TokenCount:
    def count_prompt_tokens(self, text):
        return len(text) // 4

    count_tokens = count_prompt_tokens


def prompts():
    evaluation = training_task(TASK, condition="traceaad")[0]
    current = PromptBuilder(TokenCount(), TASK, evaluation, {}, {}, Config()).initial([])["prompt"]
    instruction = INITIAL.split("\n", 1)[1]
    assert instruction in current
    return {"N": current, "M": current.replace(instruction, V914_INSTRUCTION), "O": V914_PROMPT,
            "N1": current, "O1": V914_PROMPT}


class Study:
    def __init__(self):
        self.llms = [build_llm_client(base_url=BACKENDS[b].base_url, model=BACKENDS[b].model,
                                      no_proxy=BACKENDS[b].no_proxy, max_tokens=8192) for b in ("server3", "server3b")]
        self.old_llms = [build_llm_client(base_url=BACKENDS[b].base_url, model=BACKENDS[b].model,
                                          no_proxy=BACKENDS[b].no_proxy, max_tokens=8192, temperature=1.0, top_p=0.95)
                         for b in ("server3", "server3b")]
        for llm in self.old_llms:
            llm.extra_body["presence_penalty"] = 0.0
        self.lock, self.local = threading.Lock(), threading.local()
        self.template = str(training_task(TASK, condition="traceaad")[0].template_program)
        self.prompts = prompts()
        OUT.mkdir(parents=True, exist_ok=True)
        with open(OUT / "prompts.json", "w") as f:
            json.dump(self.prompts, f, indent=1)

    def evaluate(self, code):
        if not hasattr(self.local, "evaluator"):
            self.local.evaluator = InstanceProgramEvaluator(training_task(TASK, condition="traceaad")[0], (SEED,), "search",
                                                            timeout_seconds=INSTANCE_SECONDS, n_workers=8,
                                                            scheduler_socket=SOCKET)
        outcome = self.local.evaluator.evaluate(code, key(code))
        failure = outcome["failure"]
        return {"status": failure["kind"] if failure else "valid", "fitness": outcome["fitness"],
                "function_seconds": outcome.get("function_cpu_seconds"),
                "error": (failure["error"] or "")[-300:] if failure else None}

    def run(self, job, index):
        arm, sample = job
        record = {"arm": arm, "sample": sample}
        try:
            llms = self.old_llms if arm.endswith("1") else self.llms
            details = generate(llms[index % 2], self.prompts[arm], max_tokens=8192)
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
        print(f"{arm} {sample}: {record.get('status')} {record.get('fitness')}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--prompts-only", action="store_true")
    args = parser.parse_args(argv)
    study = Study()
    if args.prompts_only:
        for arm, text in study.prompts.items():
            print(f"===== {arm}\n{text}\n")
        return
    done = set()
    if (OUT / "results.jsonl").exists():
        done = {(r["arm"], r["sample"]) for r in map(json.loads, open(OUT / "results.jsonl"))}
    jobs = [(arm, s) for s in range(args.samples) for arm in study.prompts if (arm, s) not in done]
    print(f"{len(jobs)} jobs", flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        for future in [pool.submit(study.run, job, i) for i, job in enumerate(jobs)]:
            future.result()


if __name__ == "__main__":
    main()
