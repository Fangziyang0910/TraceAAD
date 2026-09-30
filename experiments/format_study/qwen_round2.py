"""Round 2: how much Analysis, what kind, and how long a Design?  Qwen, paired contexts, resumable.

usage: think_exp2.py refine N        (8 arms on Refine, N parents per task)
       think_exp2.py transfer N ARM... (listed arms on Explore and Crossover)
"""
import experiments  # noqa: F401
import difflib
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
from traceaad.v10_15.canonical import canonical, key
from traceaad.v10_15.config import Config
from traceaad.v10_15.delivery import DeliveryError, SourceError, parse_response
from traceaad.v10_15.evaluation import SeededEvaluation
from traceaad.v10_15.selection import choose_explore_references, choose_reference

OUT = os.environ.get("FORMAT_STUDY_OUT", "experiments_result/format_study/qwen_round2")
os.makedirs(OUT, exist_ok=True)
TAIL = ("Code:\n```python\n<the complete program>\n```\n"
        "Write no comments or docstrings in the code, and nothing after the code block.")
DESIGN = {"d25": "one sentence (at most 25 words) stating the core idea of the algorithm you will implement",
          "d60": "one or two sentences (at most 60 words) stating the core idea of the algorithm you will implement",
          "d150": "about 100-150 words of plain prose describing the algorithm you will implement: its core idea, "
                  "the key quantities it computes, and how they are combined into each decision"}
TARGET = {"Refine": "what limits the current algorithm, and what change should improve it",
          "Explore": "what limits the current algorithm, and what different idea should do better",
          "Crossover": "what the reference algorithm does well that the current algorithm lacks, and how to combine them"}


def fmt(arm, action):
    target = TARGET[action]
    analysis = {
        "design_only": None,
        "one_sentence": f"one sentence: {target}",
        "few": f"a few sentences: {target}",
        "paragraph": f"one paragraph (at most 150 words): {target}",
        "generic": "a few sentences: think about how to approach this task",
        "options": "a few sentences: list two or three possible changes and choose the most promising one",
        "target_options": f"a few sentences: {target}; weigh two or three options and choose the most promising one",
        "few_d25": f"a few sentences: {target}",
        "few_d150": f"a few sentences: {target}",
    }[arm]
    design = DESIGN["d25" if arm == "few_d25" else "d150" if arm == "few_d150" else "d60"]
    lines = ["[Output Format]", "Reply in this order:"]
    if analysis:
        lines.append(f"Analysis: <{analysis}. It will not be shown again>")
    lines.append(f"Design: <{design}>")
    return "\n".join(lines) + "\n" + TAIL


REFINE_ARMS = ["design_only", "one_sentence", "few", "paragraph", "generic", "options", "few_d25", "few_d150"]


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
            EVAL[task] = (e, SecureEvaluator(SeededEvaluation(e)))
        return EVAL[task]


def archives(task):
    for path in sorted(glob.glob(f"experiments_result/traceaad_v10_15_4/{task}/*/search.jsonl")):
        nodes = {}
        with open(path, "rb") as f:
            for raw in f:
                if raw.startswith(b'{"kind":"node"'):
                    n = json.loads(raw)["data"]
                    nodes[n["id"]] = n
        yield path.split("/")[-2], nodes


CLIENTS = {}


def client(backend):
    with LOCK:
        if backend not in CLIENTS:
            b = BACKENDS[backend]
            CLIENTS[backend] = build_llm_client(base_url=b.base_url, model=b.model, no_proxy=b.no_proxy, max_tokens=8192)
        return CLIENTS[backend]


def changed_lines(a, b):
    return sum(1 for l in difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=0)
               if l[:1] in "+-" and not l.startswith(("+++", "---")))


def run_jobs(jobs, path):
    write = threading.Lock()

    def run(item):
        i, (job, task, archive, parent, arm, action, prompt) = item
        t0 = time.monotonic()
        for attempt in range(4):
            try:
                d = client("server3" if i % 2 else "server3b").draw_sample_with_details(prompt)
                break
            except Exception:
                time.sleep(10 * (attempt + 1))
        else:
            return
        content = d.get("content", "")
        a0, d0 = content.find("Analysis:"), content.find("Design:")
        rec = {"job": job, "task": task, "context": job.rsplit("|", 1)[0], "arm": arm, "action": action,
               "seconds": time.monotonic() - t0, "output_tokens": (d.get("usage") or {}).get("completion_tokens"),
               "analysis_words": len(content[a0:d0].split()) if 0 <= a0 < d0 else 0,
               "parent_fitness": parent["fitness"]}
        e, sec = evaluation(task)
        try:
            code, design, _ = parse_response(content, d.get("finish_reason"), e.template_program)
        except (DeliveryError, SourceError):
            rec["status"] = "delivery/source"
        else:
            normal = canonical(code)
            rec["design_words"] = len(design.split())
            rec["changed_lines"] = changed_lines(parent["code"], normal)
            k = key(normal)
            grand = archive.get(parent["parent_id"])
            if grand is not None and k == grand["key"]:
                rec["status"] = "undo"
            elif k in {n["key"] for n in archive.values()}:
                rec["status"] = "duplicate"
            else:
                out = sec.evaluate_program_with_details(e.template_program, source=normal, seed=730241)
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


def build(actions, arms, n_per_task, seed, path):
    finished = {json.loads(l)["job"] for l in open(path)} if os.path.exists(path) else set()
    rng = random.Random(seed)
    jobs = []
    for task in ("tsp_construct", "online_bin_packing", "vrptw_construct"):
        e, _ = evaluation(task)
        pool = []
        for run, archive in archives(task):
            pool += [(run, archive, n) for n in archive.values() if n["parent_id"] in archive]
        for run, archive, parent in rng.sample(pool, n_per_task):
            for action in actions:
                kwargs = {}
                if action == "Explore":
                    refs, _ = choose_explore_references(parent, archive, random.Random(parent["id"]))
                    kwargs = {"references": refs, "best_score": max(n["score"] for n in archive.values())}
                elif action == "Crossover":
                    ref, _ = choose_reference(parent, archive, random.Random(parent["id"]))
                    if ref is None:
                        continue
                    kwargs = {"reference": ref}
                for arm in arms:
                    prompts.output_format = lambda _action, f=fmt(arm, action): f
                    request = prompts.PromptBuilder(Counter(), task, e, archive, Config()).build(action, parent, **kwargs)
                    if request["action"] != action:
                        continue
                    job = f"{task}|{run}|{parent['id']}|{action}|{arm}"
                    if job not in finished:
                        jobs.append((job, task, archive, parent, arm, action, request["prompt"]))
    return jobs


if __name__ == "__main__":
    mode, n = sys.argv[1], int(sys.argv[2])
    if mode == "refine":
        path = f"{OUT}/refine.jsonl"
        jobs = build(["Refine"], REFINE_ARMS, n, 5, path)
    else:
        path = f"{OUT}/transfer.jsonl"
        jobs = build(["Explore", "Crossover"], sys.argv[3:], n, 9, path)
    print(f"{mode}: {len(jobs)} generations", flush=True)
    run_jobs(jobs, path)
    print("ALL DONE", flush=True)
