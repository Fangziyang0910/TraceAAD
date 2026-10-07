"""Fixed-parent paired study of Explore variants on V10.15-5 archives (resumable).

usage: PYTHONPATH=. uv run python -m experiments.diagnosis_v1015_5.explore_study TASK N_PER_RUN SAMPLES [ARM...]

Parents are the best distinct score classes of each V10.15-5 run's archive truncated
at attempt CUT (default 600), i.e. what Explore meets once the run has settled into a
basin.  Every arm sees the same parent and truncated archive; only the Explore text,
the [Evaluation] wording and the sampling profile vary.  Children are evaluated on the
training evaluator with a 120 s limit on an otherwise idle host; the recorded wall time
says whether they would have passed the 30 s search limit.
"""
import experiments  # noqa: F401
import glob
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import traceaad.v10_15.prompts as P
from core import SecureEvaluator
from experiments.infra.base import BACKENDS, build_llm_client, build_task
from traceaad.common.canonical import canonical, key, similarity
from traceaad.v10_15.config import Config
from traceaad.common.state import Facts
from pathlib import Path
from traceaad.common.delivery import DeliveryError, SourceError, parse_response
from traceaad.common.evaluation import SeededEvaluation
from traceaad.v10_15.selection import choose_explore_references

OUT = os.environ.get("EXPLORE_STUDY_OUT", "experiments_result/diagnosis_v1015_5/explore_study")
CUT = int(os.environ.get("EXPLORE_STUDY_CUT", "600"))
os.makedirs(OUT, exist_ok=True)

# V10.15-1/-2 Explore text (verbatim), which produced most of their TSP structure jumps.
V1_EXPLORE = """[Your Task: Explore]
Find a materially different way to solve this task better than the current algorithm.
- First identify the main limitation of the current approach: information it ignores, decisions it systematically gets wrong, or situations it cannot represent.
- Then change the core of the algorithm to remove that limitation: what it computes from the inputs, how it evaluates a choice before committing to it, or how it turns signals into a decision. Tuning parameters or making a small local edit is not enough.
- You may keep useful parts of the current program or start from scratch, but do not simply restate an idea listed above.
- The new algorithm must be complete and competitive on its own, and must stay within the time limit."""
V1_FORMAT = """[Output Format]
Reply with exactly one Idea line followed by one Python code block:
Idea: <one sentence, at most 300 characters, describing the new algorithm>
Code:
```python
<the complete program>
```
Write no comments or docstrings in the code, and nothing after the code block."""

# Proposed: a structural rewrite anchored on the current program and aimed at the search best.
RESTRUCTURE = """[Your Task: Explore]
Write an algorithm that beats the best score found so far by changing how the current algorithm makes its decisions, not by tuning it.
Identify what the current algorithm cannot capture: information it ignores, choices whose consequences it never evaluates, or situations it cannot represent. Then change its core computation to capture that: what it computes from the inputs, how it evaluates a choice before committing to it, or how it turns signals into a decision. Keep parts of the current program only where they still serve the new design."""
RESTRUCTURE_ANALYSIS = "what the current algorithm cannot capture, and what different computation should capture it"


def restructure_format():
    return "\n".join(["[Output Format]", "Reply in this order:",
                      f"Analysis: <a few sentences: {RESTRUCTURE_ANALYSIS}. It will not be shown again>",
                      "Design: <one or two sentences (at most 60 words) stating the core idea of the algorithm you will implement>",
                      "Code:\n```python\n<the complete program>\n```",
                      "Write no comments or docstrings in the code, and nothing after the code block."])


THINKING_PROFILE = {"temperature": 1.0, "top_p": 0.95, "extra_body": {"presence_penalty": 0.0}}
ARMS = {
    # name: (instruction, output format, references shown, evaluation wording, sampling overrides)
    "cur": ("cur", "cur", True, "efficient", {}),
    "v1": ("v1", "v1", False, "efficient", {}),
    "restr": ("restr", "restr", False, "efficient", {}),
    "restr_time": ("restr", "restr", False, "headroom", {}),
    "restr_time_t1": ("restr", "restr", False, "headroom", THINKING_PROFILE),
    "cur_t1": ("cur", "cur", True, "efficient", THINKING_PROFILE),
    "cur_time": ("cur", "cur", True, "headroom", {}),
    "v1_time": ("v1", "v1", False, "headroom", {}),
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
            nominal = e.timeout_seconds
            e.timeout_seconds = 120
            EVAL[task] = (e, SecureEvaluator(SeededEvaluation(e, measure_calls=False)), nominal)
        return EVAL[task]


def load_run(path):
    return Facts(Path(path).parent).valid


def truncated(nodes, cut):
    return {i: n for i, n in nodes.items() if (n.get("attempt_id") or 0) <= cut}


def parents_of(archive, n, higher):
    best = {}
    for node in sorted(archive.values(), key=lambda n: n["id"]):
        k = round(node["score"], 9)
        best.setdefault(k, node)
    classes = sorted(best, reverse=higher)
    return [best[k] for k in classes[:n]]


def build_prompt(task, e, archive, parent, arm, nominal, parent_seconds):
    instr, fmt, refs, wording, _ = ARMS[arm]
    builder = P.PromptBuilder(Counter(), task, e, archive, {}, Config())
    if wording == "headroom":
        timeout = format(nominal, "g")
        builder.common[1] = builder.common[1].replace(
            f"The whole evaluation must finish within {format(e.timeout_seconds, 'g')} seconds, so keep the computation efficient.",
            f"The whole evaluation must finish within {timeout} seconds. The current algorithm takes about "
            f"{parent_seconds:.1f} seconds, so substantially more computation per decision is affordable when it improves the result.")
    else:
        builder.common[1] = builder.common[1].replace(f"within {format(e.timeout_seconds, 'g')} seconds",
                                                      f"within {format(nominal, 'g')} seconds")
    higher = builder.higher_is_better
    best = (max if higher else min)(n["score"] for n in archive.values())
    references = choose_explore_references(parent, archive, random.Random(parent["id"]))[0] if refs else []
    sections = builder.common + [builder._current(parent)]
    sections.append(builder._reference_ideas(references, best))
    sections.append({"cur": P.EXPLORE, "v1": V1_EXPLORE, "restr": RESTRUCTURE}[instr])
    sections.append({"cur": P.output_format("Explore"), "v1": V1_FORMAT, "restr": restructure_format()}[fmt])
    return builder._render(sections), best, higher


CLIENTS = {}


def client(backend):
    with LOCK:
        if backend not in CLIENTS:
            b = BACKENDS[backend]
            CLIENTS[backend] = build_llm_client(base_url=b.base_url, model=b.model, no_proxy=b.no_proxy, max_tokens=8192)
        return CLIENTS[backend]


def main():
    task, n_per_run, samples = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    arms = sys.argv[4:] or list(ARMS)
    path = f"{OUT}/{task}.jsonl"
    done = {json.loads(l)["job"] for l in open(path)} if os.path.exists(path) else set()
    e, sec, nominal = evaluation(task)
    jobs = []
    timing = {}
    for run_path in sorted(glob.glob(f"experiments_result/traceaad_v10_15_5/{task}/*/events.jsonl")):
        run = run_path.split("/")[-2]
        archive = truncated(load_run(run_path), CUT)
        higher = P.SCORES[task][1]
        for parent in parents_of(archive, n_per_run, higher):
            pk = f"{run}|{parent['id']}"
            if pk not in timing:
                t0 = time.monotonic()
                sec.evaluate_program_with_details(parent['code'], seed=730241)
                timing[pk] = time.monotonic() - t0
            for arm in arms:
                prompt, best, higher = build_prompt(task, e, archive, parent, arm, nominal, timing[pk])
                for s in range(samples):
                    job = f"{task}|{pk}|{arm}|{s}"
                    if job not in done:
                        jobs.append(dict(job=job, task=task, run=run, archive=archive, parent=parent, arm=arm,
                                         prompt=prompt, best=best, higher=higher, parent_seconds=timing[pk]))
    print(f"{task}: {len(jobs)} generations, parents timed: {len(timing)}", flush=True)
    write = threading.Lock()
    eval_slots = threading.Semaphore(int(os.environ.get("EXPLORE_STUDY_EVAL_SLOTS", "6")))

    def run(item):
        i, j = item
        overrides = dict(ARMS[j["arm"]][4])
        t0 = time.monotonic()
        for attempt in range(4):
            try:
                d = client("server3" if i % 2 else "server3b").draw_sample_with_details(j["prompt"], **overrides)
                break
            except Exception:
                time.sleep(10 * (attempt + 1))
        else:
            return
        content = d.get("content", "")
        parent = j["parent"]
        rec = {k: j[k] for k in ("job", "task", "run", "arm", "best", "higher", "parent_seconds")}
        rec.update(parent_id=parent["id"], parent_score=parent["score"], llm_seconds=time.monotonic() - t0,
                   output_tokens=(d.get("usage") or {}).get("completion_tokens"), response=content)
        try:
            code, design, _ = parse_response(content, d.get("finish_reason"), e.template_program)
        except (DeliveryError, SourceError) as exc:
            rec["status"] = "delivery/source"
            rec["error"] = str(exc)[:300]
        else:
            normal = canonical(code)
            rec["design"] = design
            rec["sim_parent"] = similarity(normal, parent["code"])
            rec["code"] = normal
            if key(normal) in {n["key"] for n in j["archive"].values()}:
                rec["status"] = "duplicate"
            else:
                with eval_slots:
                    t1 = time.monotonic()
                    out = sec.evaluate_program_with_details(normal, seed=730241)
                    rec["eval_seconds"] = time.monotonic() - t1
                v = out.result
                if isinstance(v, dict) and isinstance(v.get("score"), float):
                    rec["status"] = "valid"
                    rec["score"] = v["score"]
                else:
                    rec["status"] = out.failure_kind or "invalid"
                    rec["error"] = (out.error or "")[:300] if hasattr(out, "error") else ""
        with write:
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    with ThreadPoolExecutor(16) as pool:
        list(pool.map(run, enumerate(jobs)))
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
