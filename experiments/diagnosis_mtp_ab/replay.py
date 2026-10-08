"""Does MTP speculative decoding change what the model writes? Same prompts, two endpoints.

usage: PYTHONPATH=. uv run python -m experiments.diagnosis_mtp_ab.replay [--prompts 150] [--samples 2]

Both server3 endpoints serve the same weights with vLLM 0.31.0 and the recipe's flags;
:8000 (backend server3) adds MTP with 3 speculative tokens, :8001 (server3b) has none.
The prompts are the exact Refine, Explore and Crossover prompts the stopped TSP platform
runs sent (20261008_cpu_*), drawn evenly over the nine runs. Each prompt goes to both
endpoints ``--samples`` times with the client's usual sampling. Children are evaluated on
the training set under the current protocol (2 s CPU budget in the function, 20 s per
instance) on the local P-core pool. A child is a duplicate when its source equals a program
the run had evaluated before that prompt. Results go to ``results.jsonl``; reruns skip
finished jobs.
"""

import argparse
import gzip
import json
import random
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
from traceaad.common.state import Facts

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments_result" / "diagnosis_mtp_ab"
RUNS = "experiments_result/traceaad_v10_21/tsp_construct/20261008_cpu_*"
ARMS = {"mtp": "server3", "no_mtp": "server3b"}
SOCKET = "/tmp/traceaad-1000/scheduler.sock"
SHAPE = ("broadcast", "out of bounds", "too many indices", "AxisError", "kth(")


def rerender(rows, builder_class, config):
    """The same decision states written with another version's prompt builder."""
    llm = build_llm_client(base_url=BACKENDS["server3"].base_url, model=BACKENDS["server3"].model,
                           no_proxy=BACKENDS["server3"].no_proxy, max_tokens=8192)
    evaluation = training_task("tsp_construct", condition="traceaad")[0]
    facts = {}
    for row in rows:
        run = facts.setdefault(row["run"], Facts(ROOT.glob(f"experiments_result/*/tsp_construct/{row['run']}").__next__()))
        attempt = run.attempts[row["attempt"]]
        programs = {i: p for i, p in run.programs.items() if i < row["attempt"]}
        attempts = {i: a for i, a in run.attempts.items() if i < row["attempt"]}
        builder = builder_class(llm, "tsp_construct", evaluation, programs, attempts, config)
        reference = run.programs.get(attempt.get("reference_id")) if row["action"] == "Crossover" else None
        row["prompt"] = builder.build(row["action"], run.programs[attempt["parent_id"]], reference=reference)["prompt"]
    return rows


def prompts(count, runs=RUNS, first=None, seed=0):
    """(run, attempt id, prompt, keys the run had evaluated before it), evenly over the runs;
    ``first`` keeps only the run's first attempts."""
    rows = []
    runs = sorted(ROOT.glob(runs))
    for run in runs:
        facts = Facts(run)
        by_request = {a["request_id"]: a for a in facts.attempts.values()}
        own = []
        calls = next(run.glob("calls.jsonl*"))
        for line in (gzip.open(calls, "rt") if calls.suffix == ".gz" else open(calls)):
            call = json.loads(line)
            attempt = by_request.get(call.get("request_id"))
            if (attempt and attempt["action"] in ("Refine", "Explore", "Crossover") and not attempt.get("repair_of")
                    and (first is None or attempt["id"] <= first)):
                known = sorted(p["key"] for p in facts.programs.values() if p["id"] < attempt["id"])
                own.append({"run": run.name, "attempt": attempt["id"], "action": attempt["action"],
                            "original": attempt["status"],
                            "parent_key": facts.programs[attempt["parent_id"]]["key"],
                            "reference_key": (facts.programs[attempt["reference_id"]]["key"]
                                              if attempt.get("reference_id") else None),
                            "prompt": call["prompt"], "known": known})
        random.Random(f"{seed}:{run.name}").shuffle(own)
        rows += own[:count // len(runs)]
    return rows


class Study:
    def __init__(self, rows, samples, out=OUT):
        self.rows, self.samples = rows, samples
        self.llms = {arm: build_llm_client(base_url=BACKENDS[b].base_url, model=BACKENDS[b].model,
                                           no_proxy=BACKENDS[b].no_proxy, max_tokens=8192)
                     for arm, b in ARMS.items()}
        self.evaluation = training_task("tsp_construct", condition="traceaad")[0]
        self.template = str(self.evaluation.template_program)
        self.lock, self.local = threading.Lock(), threading.local()
        out.mkdir(parents=True, exist_ok=True)
        self.path = out / "results.jsonl"
        self.done = {(r["index"], r["arm"], r["sample"]) for r in map(json.loads, self.path.open())} \
            if self.path.exists() else set()

    def evaluate(self, code):
        if not hasattr(self.local, "evaluator"):
            self.local.evaluator = InstanceProgramEvaluator(self.evaluation, (730241,), "search",
                                                            timeout_seconds=INSTANCE_SECONDS, n_workers=8,
                                                            scheduler_socket=SOCKET)
        return self.local.evaluator.evaluate(code, key(code))

    def run(self, job):
        index, arm, sample = job
        row = self.rows[index]
        record = {"index": index, "arm": arm, "sample": sample, "run": row["run"], "attempt": row["attempt"],
                  "action": row["action"], "original": row["original"]}
        try:
            details = generate(self.llms[arm], row["prompt"], max_tokens=8192)
        except ModelCallError as exc:
            record.update(status="model_error", error=str(exc)[:300])
            return self.save(record)
        call = details["calls"][-1]
        record.update(finish_reason=details.get("finish_reason"),
                      output_tokens=(call["usage"] or {}).get("completion_tokens"))
        try:
            code, _, _ = parse_response(details.get("content", ""), details.get("finish_reason"), self.template)
        except DeliveryError as exc:
            record.update(status="delivery_failed", error=str(exc)[:300])
            return self.save(record)
        except SourceError as exc:
            record.update(status="invalid_source", error=str(exc)[:300])
            return self.save(record)
        try:
            code = canonical(code)
        except (SyntaxError, ValueError):
            pass
        source = key(code)
        if source == row["parent_key"] or source == row["reference_key"] or source in set(row["known"]):
            record.update(status="duplicate", same_as_parent=source == row["parent_key"],
                          same_as_reference=source == row["reference_key"])
            return self.save(record)
        outcome = self.evaluate(code)
        failure = outcome["failure"]
        error = (failure["error"] or "").strip().splitlines()[-1][:200] if failure else None
        record.update(status=failure["kind"] if failure else "valid", fitness=outcome["fitness"], error=error,
                      shape_error=bool(error and any(s in error for s in SHAPE)))
        return self.save(record)

    def save(self, record):
        with self.lock, self.path.open("a") as output:
            output.write(json.dumps(record) + "\n")
        return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=int, default=150)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--runs", default=RUNS, help="run directories (glob) whose prompts are replayed")
    parser.add_argument("--first", type=int, help="only each run's first attempts")
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--arms", help="label=backend pairs, e.g. a=server3,b=server3b")
    parser.add_argument("--render", choices=("v1017", "v1021", "no_attempts", "code_only"),
                        help="rewrite the states with this version's prompts, or with V10.21's minus material")
    args = parser.parse_args(argv)
    if args.arms:
        ARMS.clear()
        ARMS.update(pair.split("=") for pair in args.arms.split(","))
    rows = prompts(args.prompts, args.runs, args.first)
    if args.render == "v1017":
        from traceaad.v10_17 import Config as V1017Config
        from traceaad.v10_17.prompts import PromptBuilder as V1017Prompts
        rows = rerender(rows, V1017Prompts, V1017Config())
    elif args.render:
        from traceaad.v10_21 import Config as V1021Config
        from traceaad.v10_21.prompts import PromptBuilder as V1021Prompts

        class Reduced(V1021Prompts):
            # no_attempts: no attempt list; code_only: neither attempts nor any formation or progress history.
            def _attempts_section(self, *args, **kwargs):
                return ("", []) if args_render != "v1021" else super()._attempts_section(*args, **kwargs)

            def _formation(self, *args, **kwargs):
                return ("", []) if args_render == "code_only" else super()._formation(*args, **kwargs)

            def _progress_section(self, *args, **kwargs):
                return ("", []) if args_render == "code_only" else super()._progress_section(*args, **kwargs)

        args_render = args.render
        rows = rerender(rows, Reduced, V1021Config())
    study = Study(rows, args.samples, Path(args.out))
    (Path(args.out) / "prompts.json").write_text(json.dumps([{k: v for k, v in r.items() if k != "known"} for r in rows]))
    # Interleave the arms so both endpoints see the same load over time.
    jobs = [(i, arm, s) for s in range(args.samples) for i in range(len(rows)) for arm in ARMS
            if (i, arm, s) not in study.done]
    with ThreadPoolExecutor(args.workers) as pool:
        for record in pool.map(study.run, jobs):
            print(record["arm"], record["status"], record.get("error") or "", flush=True)


if __name__ == "__main__":
    main()
