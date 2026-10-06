from traceaad.common.state import Facts
from pathlib import Path
"""Does the Design's length and position affect generated code?  (DeepSeek v4.1 flash, thinking off)

Part A (producer): Refine on paired parent contexts, 7 arms =
    {first, after} x {short, medium, long} + code-only.
Part B (consumer): Explore with the same references described at three lengths.
Results are appended to JSONL files so an interrupted run resumes.
"""
import experiments  # noqa: F401  (single-threaded BLAS)
import glob
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import traceaad.v10_15.prompts as prompts
from core import SecureEvaluator
from experiments.infra.base import build_task
from traceaad.common.canonical import canonical, key
from traceaad.v10_15.config import Config
from traceaad.common.delivery import DeliveryError, SourceError, parse_response
from traceaad.common.evaluation import SeededEvaluation
from traceaad.v10_15.selection import choose_explore_references

URL = os.environ.get("FORMAT_STUDY_URL", "http://183.36.243.124:33333/v1/chat/completions")
MODEL = "deepseek-v4.1-flash"
OUT = os.environ.get("FORMAT_STUDY_OUT", "experiments_result/format_study/deepseek_length_order")
os.makedirs(OUT, exist_ok=True)

LENGTHS = {"short": "one or two sentences (at most 40 words)",
           "medium": "about 100-200 words",
           "long": "about 250-400 words, covering every component and parameter"}
CONTENT = ("its core idea, the key quantities it computes, and how they are combined into each decision")


def fmt(order, length):
    if order == "none":
        return ("[Output Format]\nReply with one Python code block:\n"
                "```python\n<the complete program>\n```\n"
                "Write no comments or docstrings in the code, and nothing after the code block.")
    size = LENGTHS[length]
    if order == "first":
        return ("[Output Format]\nReply with a Design followed by one Python code block:\n"
                f"Design: <the design of the algorithm you will implement, in {size} of plain prose: {CONTENT}. "
                "State only the final design; leave out deliberation, alternatives, formulas and comparisons with earlier versions>\n"
                "Code:\n```python\n<the complete program>\n```\n"
                "Write no comments or docstrings in the code, and nothing after the code block.")
    return ("[Output Format]\nReply with one Python code block followed by a Design:\n"
            "Code:\n```python\n<the complete program>\n```\n"
            f"Design: <{size} of plain prose describing the algorithm implemented in the code above: {CONTENT}. "
            "Describe only what the code does: no formulas, alternatives or comparisons with earlier versions>\n"
            "Write no comments or docstrings in the code.")


ARMS = [("none", None)] + [(o, l) for o in ("first", "after") for l in ("short", "medium", "long")]


def chat(prompt, max_tokens=8192):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0.7, "top_p": 0.8, "thinking": {"type": "disabled"}}
    for attempt in range(6):
        req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={
            "Content-Type": "application/json", "Authorization": "Bearer " + os.environ["FORMAT_STUDY_KEY"]})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                x = json.loads(r.read())
            c = x["choices"][0]
            return c["message"].get("content") or "", c.get("finish_reason"), (x.get("usage") or {}).get("completion_tokens")
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, ConnectionError) as exc:
            time.sleep(min(60, 5 * 2 ** attempt))
            last = exc
    raise RuntimeError(f"chat failed: {last}")


class TokenCounter:
    """Prompt sizes only decide trimming, which never triggers at these sizes."""
    def count_prompt_tokens(self, text):
        return len(text) // 4
    count_tokens = count_prompt_tokens


EVAL = {}
EVAL_LOCK = threading.Lock()


def evaluation(task):
    with EVAL_LOCK:
        if task not in EVAL:
            e, _ = build_task(task, 4)
            e.timeout_seconds = 120  # arm comparison, not a timeout study
            EVAL[task] = (e, SecureEvaluator(SeededEvaluation(e, measure_calls=False)))
        return EVAL[task]


def score(task, code):
    e, sec = evaluation(task)
    out = sec.evaluate_program_with_details(e.template_program, source=code, seed=730241)
    v = out.result
    if isinstance(v, dict) and isinstance(v.get("score"), float):
        return "valid", v["score"]
    return out.failure_kind or "invalid", None


def archives(task):
    for path in sorted(glob.glob(f"experiments_result/traceaad_v10_15_2/{task}/*/events.jsonl")):
        yield Path(path).parent.name, Facts(Path(path).parent).valid


def outcome(task, archive, parent, content, finish, tokens, design_expected):
    e, _ = evaluation(task)
    rec = {"output_tokens": tokens, "finish": finish}
    try:
        code, design, _ = parse_response(content, finish, e.template_program)
    except (DeliveryError, SourceError):
        rec["status"] = "delivery/source"
        return rec
    rec["design_words"] = len(design.split())
    rec["design_missing"] = design_expected and not design.strip()
    normal = canonical(code)
    k = key(normal)
    grand = archive.get(parent["parent_id"])
    if grand is not None and k == grand["key"]:
        rec["status"] = "undo"
        return rec
    if k in {n["key"] for n in archive.values()}:
        rec["status"] = "duplicate"
        return rec
    rec["status"], rec["score"] = score(task, normal)
    return rec


def done(path):
    if not os.path.exists(path):
        return set()
    return {json.loads(l)["job"] for l in open(path)}


def append(path, rec, lock=threading.Lock()):
    with lock:
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")


def part_a(n_per_task=20):
    path = f"{OUT}/part_a.jsonl"
    finished = done(path)
    rng = random.Random(7)
    jobs = []
    for task in ("tsp_construct", "online_bin_packing", "vrptw_construct"):
        e, _ = evaluation(task)
        pool = []
        for run, archive in archives(task):
            pool += [(run, archive, n) for n in archive.values() if n["parent_id"] in archive]
        for run, archive, parent in rng.sample(pool, n_per_task):
            for order, length in ARMS:
                prompts.output_format = lambda f=fmt(order, length): f
                builder = prompts.PromptBuilder(TokenCounter(), task, e, archive, {}, Config())
                job = f"{task}|{run}|{parent['id']}|{order}|{length}"
                if job not in finished:
                    jobs.append((job, task, archive, parent, order, length, builder.build("Refine", parent)["prompt"]))
    print(f"part A: {len(jobs)} generations to run", flush=True)

    def run(item):
        job, task, archive, parent, order, length, prompt = item
        content, finish, tokens = chat(prompt)
        rec = outcome(task, archive, parent, content, finish, tokens, order != "none")
        rec.update(job=job, task=task, context=job.rsplit("|", 2)[0], order=order, length=length,
                   parent_fitness=parent["fitness"])
        append(path, rec)

    with ThreadPoolExecutor(12) as pool:
        list(pool.map(run, jobs))


def describe(task, node, length, cache={}, lock=threading.Lock()):
    """Describe an archived program at a given length (the reference card text)."""
    ck = (task, node["key"], length)
    with lock:
        if ck in cache:
            return cache[ck]
    e, _ = evaluation(task)
    prompt = ("[Task]\n" + e.task_description.strip() + "\n\n[Program]\n```python\n" + node["code"].rstrip() + "\n```\n\n"
              f"Describe the algorithm this program implements in {LENGTHS[length]} of plain prose: {CONTENT}. "
              "Describe only what the code does: no formulas and no code. Reply with the description only.")
    text, _, _ = chat(prompt, max_tokens=1200)
    with lock:
        cache[ck] = " ".join(text.split())
    return cache[ck]


def part_b(n_per_task=20):
    path = f"{OUT}/part_b.jsonl"
    finished = done(path)
    rng = random.Random(11)
    contexts = []
    for task in ("tsp_construct", "online_bin_packing"):
        pool = []
        for run, archive in archives(task):
            pool += [(run, archive, n) for n in archive.values() if n["parent_id"] in archive]
        for run, archive, parent in rng.sample(pool, n_per_task):
            refs, _ = choose_explore_references(parent, archive, random.Random(parent["id"]))
            contexts.append((task, run, archive, parent, refs))
    print(f"part B: {len(contexts)} contexts", flush=True)
    prompts.output_format = lambda f=fmt("first", "medium"): f

    def run(item):
        (task, run, archive, parent, refs), length = item
        job = f"{task}|{run}|{parent['id']}|{length}"
        if job in finished:
            return
        cards = [{**r, "idea": describe(task, r, length)} for r in refs]
        e, _ = evaluation(task)
        builder = prompts.PromptBuilder(TokenCounter(), task, e, archive, {}, Config())
        best = max(n["score"] for n in archive.values())
        prompt = builder.build("Explore", parent, references=cards, best_score=best)["prompt"]
        content, finish, tokens = chat(prompt)
        rec = outcome(task, archive, parent, content, finish, tokens, True)
        rec.update(job=job, task=task, context=job.rsplit("|", 1)[0], length=length,
                   card_words=sum(len(c["idea"].split()) for c in cards) / max(1, len(cards)),
                   parent_fitness=parent["fitness"])
        append(path, rec)

    with ThreadPoolExecutor(12) as pool:
        list(pool.map(run, [(c, l) for c in contexts for l in LENGTHS]))


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "ab"
    if "a" in part:
        part_a(30)
    if "b" in part:
        part_b(25)
    print("done", flush=True)
