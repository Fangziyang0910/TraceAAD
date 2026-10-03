"""Counterfactual replay: does the current algorithm's time profile change what Refine writes?

usage: PYTHONPATH=. uv run python -m experiments.diagnosis_v1016_profile.replay [SAMPLES] [WORKERS]

Items are the last Refine prompts of V10.16 TSP runs whose parent took at least 18 s
of the 30 s search limit (``items_all.json``, extracted from the server3 logs). Each
prompt is resampled as logged (arm ``base``) and with one added section that reports
where the current algorithm spends its time (arm ``profile``); nothing else differs.
Children and parents are evaluated locally in one pool with a 90 s limit; analyze.py
judges whether a child would have met the 30 s search limit by its time relative to its
parent on this host, scaled to the parent's logged search time, so the verdict does not
depend on how fast this host is. Results are appended to ``results.jsonl``; reruns skip finished samples.
"""
import experiments  # noqa: F401
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from benchmarks.generated_data_config import get_generated_task_kwargs
from benchmarks.tsp_construct import TSPEvaluation
from core import SecureEvaluator
from experiments.infra.base import BACKENDS, build_llm_client
from traceaad.v10_16.delivery import DeliveryError, SourceError, parse_response
from traceaad.v10_16.evaluation import SeededEvaluation

OUT = os.environ.get("PROFILE_STUDY_OUT", "experiments_result/diagnosis_v1016_profile")
LIMIT = 90.0
MIN_SHARE = 0.02
PROFILE_HEADER = "[Where the Current Algorithm Spends Its Time]"


def training_evaluation(timeout):
    kwargs = get_generated_task_kwargs("tsp_construct", "train")
    kwargs["timeout_seconds"] = timeout
    return TSPEvaluation(**kwargs)


# ---------- profile of the parent ----------

RUNNER = """import sys, importlib.util
sys.path.insert(0, {repo!r})
import experiments
from benchmarks.generated_data_config import get_generated_task_kwargs
from benchmarks.tsp_construct import TSPEvaluation
kwargs = get_generated_task_kwargs("tsp_construct", "train")
kwargs["timeout_seconds"] = None
spec = importlib.util.spec_from_file_location("candidate", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(TSPEvaluation(**kwargs).evaluate_program(open(sys.argv[1]).read(), module.select_next_node))
"""


def profile(code, name):
    """One full training evaluation sampled by py-spy from outside the process (an
    in-process sampler only sees the lines where the GIL is released). Returns, per
    program line, the share of samples in which it is on the stack, so a line calling
    a helper includes the helper's time."""
    folder = f"{OUT}/profiles/{name}"
    os.makedirs(folder, exist_ok=True)
    with open(f"{folder}/candidate.py", "w") as f:
        f.write(code)
    with open(f"{folder}/runner.py", "w") as f:
        f.write(RUNNER.format(repo=os.getcwd()))
    raw = f"{folder}/samples.txt"
    subprocess.run(["uvx", "py-spy", "record", "--format", "raw", "--rate", "250", "--nonblocking",
                    "-o", raw, "--", sys.executable, f"{folder}/runner.py", f"{folder}/candidate.py"],
                   check=True, capture_output=True, timeout=900)
    hits, total = Counter(), 0
    for row in open(raw):
        stack, count = row.rsplit(" ", 1)
        lines = {int(n) for n in re.findall(r"\(candidate\.py:(\d+)\)", stack)}
        if lines:
            total += int(count)
            hits.update({n: int(count) for n in lines})
    return {"samples": total, "lines": {str(n): c for n, c in hits.items()}}


def profile_section(code, measured):
    lines = code.splitlines()
    total = measured["samples"]
    counts = {int(n): c for n, c in measured["lines"].items()}
    shown = [(n, counts[n] / total) for n in sorted(counts)
             if counts[n] / total >= MIN_SHARE and 1 <= n <= len(lines)]
    body = "\n".join(f"{100 * share:5.1f}% | {lines[n - 1]}" for n, share in shown)
    return (f"{PROFILE_HEADER}\n"
            "Share of the time inside the function spent on each line of the current algorithm, "
            "sampled over one full evaluation; a line that calls a helper includes the helper's time, "
            "and lines with less than 2% are omitted.\n"
            f"```\n{body}\n```")


def with_profile(prompt, section):
    for marker in ("[How the Current Algorithm Was Formed]", "[Attempts From the Current Algorithm]",
                   "[Your Task: Refine]"):
        at = prompt.find(marker)
        if at >= 0:
            return prompt[:at] + section + "\n\n" + prompt[at:]
    raise ValueError("no insertion point in the Refine prompt")


# ---------- evaluation of a child ----------

LOCAL = threading.local()


def evaluator():
    if not hasattr(LOCAL, "evaluator"):
        evaluation = training_evaluation(LIMIT)
        LOCAL.template = str(evaluation.template_program)
        LOCAL.evaluator = SecureEvaluator(SeededEvaluation(evaluation))
    return LOCAL.template, LOCAL.evaluator


def evaluate(code):
    template, secure = evaluator()
    started = time.monotonic()
    result = secure.evaluate_program_with_details(template, source=code, seed=730241)
    seconds = time.monotonic() - started
    value = result.result
    if isinstance(value, dict) and isinstance(value.get("score"), (int, float)):
        return {"status": "valid", "score": value["score"], "seconds": seconds}
    status = "timeout" if result.failure_kind == "timeout" else "runtime_error"
    return {"status": status, "score": None, "seconds": seconds,
            "error": (result.error or result.failure_kind or "")[:300]}


# ---------- study ----------

def name_of(item):
    return f"r{item['rep']}a{item['attempt_id']}"


def main():
    samples = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    items = json.load(open(f"{OUT}/items_all.json"))
    parents_path = f"{OUT}/parents.json"
    parents = json.load(open(parents_path)) if os.path.exists(parents_path) else {}
    for item in items:
        if name_of(item) not in parents:
            measured = profile(item["parent_code"], name_of(item))
            measured["section"] = profile_section(item["parent_code"], measured)
            parents[name_of(item)] = measured
            json.dump(parents, open(parents_path, "w"), indent=1)
            print(f"profiled {name_of(item)}: {measured['samples']} samples", flush=True)

    results_path = f"{OUT}/results.jsonl"
    done = set()
    if os.path.exists(results_path):
        done = {(r["item"], r["arm"], r["sample"]) for r in map(json.loads, open(results_path))}
    llms = [build_llm_client(base_url=BACKENDS[b].base_url, model=BACKENDS[b].model,
                             no_proxy=BACKENDS[b].no_proxy, max_tokens=8192)
            for b in ("server3", "server3b")]
    lock = threading.Lock()
    # The parent is timed in the same pool (first and last round), so a child's time
    # is compared with its parent's under the same host load.
    jobs = [(item, arm, s) for s in range(samples) for item in items
            for arm in (("parent", "base", "profile") if s in (0, samples - 1) else ("base", "profile"))
            if (name_of(item), arm, s) not in done]
    print(f"{len(jobs)} jobs to run", flush=True)

    def run(job, index):
        item, arm, s = job
        template = evaluator()[0]
        record = {"item": name_of(item), "arm": arm, "sample": s}
        if arm == "parent":
            record.update(evaluate(item["parent_code"]))
        else:
            prompt = item["prompt"] if arm == "base" else with_profile(item["prompt"], parents[name_of(item)]["section"])
            details = llms[index % 2].draw_sample_with_details(prompt, max_tokens=8192)
            record.update(response=details.get("content", ""), finish_reason=details.get("finish_reason"),
                          usage=details.get("usage"))
            try:
                code, idea, _ = parse_response(record["response"], record["finish_reason"], template)
            except (DeliveryError, SourceError) as exc:
                record.update(status="invalid", error=str(exc)[:300])
            else:
                record["idea"], record["code"] = idea, code
                record.update(evaluate(code))
        with lock:
            with open(results_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        print(f"{record['item']} {arm} {s}: {record['status']} {record.get('score')} "
              f"{record.get('seconds', 0):.1f}s", flush=True)

    with ThreadPoolExecutor(workers) as pool:
        for future in [pool.submit(run, job, i) for i, job in enumerate(jobs)]:
            future.result()


if __name__ == "__main__":
    main()
