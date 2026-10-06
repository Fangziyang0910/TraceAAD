from traceaad.common.state import Facts
from pathlib import Path
"""How should Qwen think before coding? Paired Refine contexts, 6 output formats.  Resumable JSONL."""
import experiments  # noqa: F401
import glob
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import traceaad.v10_15.prompts as prompts
from core import SecureEvaluator
from experiments.infra.base import BACKENDS, build_llm_client, build_task
from traceaad.common.canonical import canonical, key
from traceaad.v10_15.config import Config
from traceaad.common.delivery import DeliveryError, SourceError, parse_response
from traceaad.common.evaluation import SeededEvaluation

OUT = os.environ.get("FORMAT_STUDY_OUT", "experiments_result/format_study/qwen_round1")
os.makedirs(OUT, exist_ok=True)
CONTENT = "its core idea, the key quantities it computes, and how they are combined into each decision"
CODE_TAIL = "Code:\n```python\n<the complete program>\n```\nWrite no comments or docstrings in the code, and nothing after the code block."
CONCISE = ("[Output Format]\nReply with a Design followed by one Python code block:\n"
           f"Design: <the design of the algorithm you will implement, in about 100-200 words of plain prose: {CONTENT}. "
           "State only the final design; leave out deliberation, alternatives, formulas and comparisons with earlier versions>\n" + CODE_TAIL)
ARMS = {
    "concise_design": (CONCISE, False),
    "long_idea": ("[Output Format]\nReply with an Idea followed by one Python code block:\n"
                  f"Idea: <about 150-250 words describing the complete algorithm in your code: {CONTENT}>\n" + CODE_TAIL, False),
    "brief_analysis": ("[Output Format]\nReply in this order:\n"
                       "Analysis: <a few sentences: what limits the current algorithm, and what change should improve it. "
                       "It will not be shown again>\n"
                       f"Design: <one or two sentences (at most 60 words) stating the core idea of the algorithm you will implement>\n"
                       + CODE_TAIL, False),
    "free_analysis": ("[Output Format]\nReply in this order:\n"
                      "Analysis: <think the problem through before coding, as much as you need. It will not be shown again>\n"
                      f"Design: <one or two sentences (at most 60 words) stating the core idea of the algorithm you will implement>\n"
                      + CODE_TAIL, False),
    "code_only": ("[Output Format]\nReply with one Python code block:\n```python\n<the complete program>\n```\n"
                  "Write no comments or docstrings in the code, and nothing after the code block.", False),
    # native thinking dropped: all 21 trials hit the 16,384-token limit before writing code
}


class Counter:
    def count_prompt_tokens(self, text):
        return len(text) // 4
    count_tokens = count_prompt_tokens


EVAL, LOCK = {}, threading.Lock()


def evaluation(task):
    with LOCK:
        if task not in EVAL:
            e, _ = build_task(task, 4)
            e.timeout_seconds = 120
            EVAL[task] = (e, SecureEvaluator(SeededEvaluation(e, measure_calls=False)))
        return EVAL[task]


def archives(task):
    for path in sorted(glob.glob(f"experiments_result/traceaad_v10_15_4/{task}/*/events.jsonl")):
        yield Path(path).parent.name, Facts(Path(path).parent).valid


CLIENTS = {}


def client(backend, thinking):
    k = (backend, thinking)
    with LOCK:
        if k not in CLIENTS:
            b = BACKENDS[backend]
            CLIENTS[k] = build_llm_client(base_url=b.base_url, model=b.model, no_proxy=b.no_proxy,
                                          max_tokens=16384 if thinking else 8192, enable_thinking=thinking)
        return CLIENTS[k]


def main(n_per_task=20):
    path = f"{OUT}/results.jsonl"
    finished = {json.loads(l)["job"] for l in open(path)} if os.path.exists(path) else set()
    rng = random.Random(3)
    jobs = []
    for task in ("tsp_construct", "online_bin_packing", "vrptw_construct"):
        e, _ = evaluation(task)
        pool = []
        for run, archive in archives(task):
            pool += [(run, archive, n) for n in archive.values() if n["parent_id"] in archive]
        for run, archive, parent in rng.sample(pool, n_per_task):
            for arm, (fmt, thinking) in ARMS.items():
                prompts.output_format = lambda f=fmt: f
                prompt = prompts.PromptBuilder(Counter(), task, e, archive, {}, Config()).build("Refine", parent)["prompt"]
                job = f"{task}|{run}|{parent['id']}|{arm}"
                if job not in finished:
                    jobs.append((job, task, archive, parent, arm, thinking, prompt))
    print(f"{len(jobs)} generations", flush=True)
    write = threading.Lock()

    def run(item):
        i, (job, task, archive, parent, arm, thinking, prompt) = item
        llm = client("server3" if i % 2 else "server3b", thinking)
        t0 = time.monotonic()
        for attempt in range(4):
            try:
                d = llm.draw_sample_with_details(prompt)
                break
            except Exception:
                time.sleep(10 * (attempt + 1))
        else:
            return
        rec = {"job": job, "task": task, "context": job.rsplit("|", 1)[0], "arm": arm,
               "seconds": time.monotonic() - t0, "output_tokens": (d.get("usage") or {}).get("completion_tokens"),
               "finish": d.get("finish_reason"), "parent_fitness": parent["fitness"]}
        e, sec = evaluation(task)
        try:
            code, design, _ = parse_response(d.get("content", ""), d.get("finish_reason"), e.template_program)
        except (DeliveryError, SourceError):
            rec["status"] = "delivery/source"
        else:
            rec["design_words"] = len(design.split())
            k = key(canonical(code))
            grand = archive.get(parent["parent_id"])
            if grand is not None and k == grand["key"]:
                rec["status"] = "undo"
            elif k in {n["key"] for n in archive.values()}:
                rec["status"] = "duplicate"
            else:
                out = sec.evaluate_program_with_details(e.template_program, source=canonical(code), seed=730241)
                v = out.result
                if isinstance(v, dict) and isinstance(v.get("score"), float):
                    rec["status"], rec["score"] = "valid", v["score"]
                else:
                    rec["status"] = out.failure_kind or "invalid"
        with write:
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    with ThreadPoolExecutor(16) as pool:
        list(pool.map(run, enumerate(jobs)))
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
