"""C2 and C3 offline, on V12.0 runs: does completing a first version that is not a test of its Design
help, and does the help come from the reading or from a second sample?

V12.0 developed an exploration on a fixed draw of 1/8, independent of its Design and its score, so the
developed explorations are a random sample. Each first version is read against its Design and its start,
as V12.1 would. For a first version not better than its start, three arms start from the same state:

- develop: the run's own first development step (a Refine of the first version, already in the archive);
- resample: the first implementation request sent again, an independent implementation from the start;
- complete: the V12.1 completion request, only when the first version is not a test.

Each new program is evaluated as in the run, without repair, and read against the same Design and start.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import queue
import threading

from benchmarks.tasks import training_task
from core.llm import generate
from experiments.infra.base import RESULTS_ROOT, build_llm_client
from traceaad.common.canonical import canonical, key
from traceaad.common.delivery import DeliveryError, SourceError, parse_response
from traceaad.common.instance_evaluation import InstanceProgramEvaluator
from traceaad.common.prompts import ContextTooLong
from traceaad.common.selection import better
from traceaad.common.state import Facts
from traceaad.common.storage import read_json, rows as saved_rows
from traceaad.v12_0 import TraceAADV120
from traceaad.v12_0.linking import cut_proposal, normal
from traceaad.v12_1.config import Config
from traceaad.v12_1.prompts import PromptBuilder
from traceaad.v12_1.verification import contain_prompt, parse_dropped, parse_verdict, verify_prompt
from .verification_report import reading

OUT = RESULTS_ROOT / "traceaad_v12_1" / "completion_pairs"
ARMS = ("develop", "resample", "complete")
# Qwen's itemized containment replies can run past the 2048 tokens set for DeepSeek.
READING_TOKENS = 4096


class Run:
    """One V12.0 run: its facts, saved calls, prompts and evaluator, and a pool of model clients."""

    def __init__(self, run_dir, clients, counter, eval_workers, eval_lock):
        self.dir, self.clients, self.eval_lock = Path(run_dir), clients, eval_lock
        self.facts = Facts(run_dir)
        config = read_json(self.dir / "run_config.json")
        self.task, self.params = config["task"], config["method_params"]
        name = "calls.jsonl.gz" if (self.dir / "calls.jsonl.gz").exists() else "calls.jsonl"
        self.calls = {c["request_id"]: c for c in saved_rows(self.dir / name)}
        evaluation, _ = training_task(self.task, eval_workers, condition="traceaad")
        self.template = str(evaluation.template_program)
        self.prompts = PromptBuilder(counter, self.task, evaluation, self.facts.programs, self.facts.attempts,
                                     Config(max_input_tokens=self.params["max_input_tokens"]))
        self.task_section = self.prompts.common[0]
        self.evaluator = InstanceProgramEvaluator(
            evaluation, tuple(self.params["evaluation_seeds"]), "search", measure_calls=TraceAADV120.MEASURE_CALLS,
            timeout_seconds=self.params["eval_timeout_seconds"], n_workers=eval_workers)
        self.eval_workers = eval_workers

    def cases(self):
        """The developed explorations with a valid first version, in order."""
        for record in sorted(self.facts.explorations.values(), key=lambda r: r["id"]):
            attempt = self.facts.attempts[record["proposal_attempt"]]
            if record["developed"] and record["proposed_id"] is not None and normal(attempt.get("idea")):
                yield record

    def ask(self, prompt, tokens, **kwargs):
        with self.clients.client() as llm:
            details = generate(llm, prompt, **kwargs)
        usage = details.get("usage") or {}
        tokens[0] += usage.get("prompt_tokens") or 0
        tokens[1] += usage.get("completion_tokens") or 0
        return details

    def read(self, design, start, code, tokens):
        """V12.1's two readings, in separate requests; the replies are kept for reading by hand."""
        verdict = {"replies": []}

        def ask(prompt):
            details = self.ask(prompt, tokens, max_tokens=READING_TOKENS, temperature=0.0)
            verdict["replies"].append({"text": details.get("content", ""), "finish_reason": details.get("finish_reason")})
            return reparse(verdict)
        ask(verify_prompt(self.task_section, design, code))
        if verdict["implemented"] is not False:
            ask(contain_prompt(self.task_section, design, start["code"], code))
        return verdict

    def program(self, code):
        """Evaluate a delivered program as the search would; an existing program is not evaluated again."""
        try:
            code = canonical(code)
        except (SyntaxError, ValueError):
            code = code.rstrip() + "\n"
        known = self.facts.by_key.get(key(code))
        if known is not None:
            return {"status": "valid" if known["valid"] else "failed", "fitness": known["fitness"],
                    "same_as": known["id"]}, code
        with self.eval_lock:
            outcome = self.evaluator.evaluate(code, key(code), timeout_seconds=self.params["eval_timeout_seconds"],
                                              n_workers=self.eval_workers)
        failed = outcome["failure"] is not None or outcome["fitness"] is None
        return {"status": "failed" if failed else "valid", "fitness": None if failed else outcome["fitness"],
                "failure": (outcome["failure"] or {}).get("kind"), "same_as": None}, code

    def arm(self, prompt, record, design, tokens):
        """Generate once from ``prompt``, evaluate, and read the program against the Design and the start."""
        details = self.ask(prompt, tokens, max_tokens=self.params["output_tokens"])
        try:
            code, _, _ = parse_response(details.get("content", ""), details.get("finish_reason"), self.template)
        except DeliveryError as exc:
            return {"status": "undelivered", "error": str(exc)[:200]}
        except SourceError as exc:
            return {"status": "failed", "failure": "invalid_source", "error": str(exc)[:200]}
        result, code = self.program(code)
        return self.judged(result, code, record, design, tokens)

    def judged(self, result, code, record, design, tokens):
        if result["status"] == "valid":
            start, first = (self.facts.programs[record[k]] for k in ("start_id", "proposed_id"))
            result.update(code=code, beats_start=better(result["fitness"], start["fitness"]),
                          beats_first=better(result["fitness"], first["fitness"]),
                          **self.read(design, start, code, tokens))
        return result

    def develop(self, record, design, tokens):
        """The run's own first development step, without its repair."""
        if not record["development_attempts"]:
            return {"status": "missing"}
        attempt = self.facts.attempts[record["development_attempts"][0]]
        program = self.facts.programs.get(attempt.get("program_id"))
        if attempt["status"] not in ("valid", "duplicate") or program is None:
            return {"status": "failed", "failure": attempt["status"]}
        result = {"status": "valid", "fitness": program["fitness"],
                  "same_as": program["id"] if attempt["status"] == "duplicate" else None}
        return self.judged(result, program["code"], record, design, tokens)

    def case(self, record):
        attempt = self.facts.attempts[record["proposal_attempt"]]
        design = normal(attempt["idea"])
        start, first = (self.facts.programs[record[k]] for k in ("start_id", "proposed_id"))
        tokens = [0, 0]
        verdict = self.read(design, start, first["code"], tokens)
        row = {"run": self.dir.name, "task": self.task, "exploration": record["id"], "start": start["id"],
               "first": first["id"], "design": design, "first_beats_start": better(first["fitness"], start["fitness"]),
               "start_fitness": start["fitness"], "first_fitness": first["fitness"], **verdict,
               "develop_best_beats_start": record["best_id"] is not None
               and better(self.facts.programs[record["best_id"]]["fitness"], start["fitness"])}
        if not row["first_beats_start"]:
            calls = [self.calls[i] for i in attempt["call_ids"]]
            implement = next(c for c in calls if c.get("stage") == "implement")
            proposal = cut_proposal(next(c for c in calls if c.get("stage") == "propose")["response"])
            # The run's task section is the completion's: the prompts are rebuilt from the same evaluation.
            assert implement["prompt"].startswith(self.prompts._render(self.prompts.common)), self.dir
            jobs = {"develop": lambda: self.develop(record, design, tokens),
                    "resample": lambda: self.arm(implement["prompt"], record, design, tokens)}
            if verdict["reading"] in ("missing", "dropped"):
                link = attempt.get("link") or {}
                earlier = self.facts.programs.get(link.get("linked_id")) if link.get("shown") else None
                base = first if verdict["kept"] is not False else start
                try:
                    prompt = self.prompts.complete(base, proposal or f"Design: {design}", verdict, earlier,
                                                   version=None if base is first else first)["prompt"]
                    jobs["complete"] = lambda: self.arm(prompt, record, design, tokens)
                except ContextTooLong:
                    row["complete"] = {"status": "too_long"}
            with ThreadPoolExecutor(len(jobs)) as pool:
                row.update(zip(jobs, pool.map(lambda job: job(), jobs.values())))
        row["input_tokens"], row["output_tokens"] = tokens
        return row


def reparse(verdict):
    """The two readings from their saved replies; a reply cut at the output limit has no judgment."""
    texts = [r["text"] if r["finish_reason"] == "stop" else "" for r in verdict["replies"]]
    verdict.update(zip(("implemented", "missing"), parse_verdict(texts[0])))
    verdict.update(zip(("kept", "dropped"), parse_dropped(texts[1]) if len(texts) > 1 else (None, None)))
    verdict["reading"] = reading(verdict)
    return verdict


class Clients:
    """At most ``slots`` requests in flight per endpoint."""

    def __init__(self, endpoints, slots, max_tokens):
        self.pool = queue.Queue()
        for endpoint in endpoints:
            for _ in range(slots):
                self.pool.put(build_llm_client(base_url=endpoint["base_url"], model=endpoint["model"],
                                               no_proxy=endpoint.get("no_proxy", ""), max_tokens=max_tokens,
                                               enable_thinking=False))

    class _Lease:
        def __init__(self, pool):
            self.pool = pool

        def __enter__(self):
            self.llm = self.pool.get()
            return self.llm

        def __exit__(self, *exc):
            self.pool.put(self.llm)

    def client(self):
        return self._Lease(self.pool)


def summary(rows):
    def share(subset, test):
        return round(sum(map(test, subset)) / len(subset), 3) if subset else None

    def readings(subset):
        return {kind: share(subset, lambda r: r["reading"] == kind) for kind in ("test", "missing", "dropped")}

    def arm(subset, name):
        results = [r[name] for r in subset if name in r]
        valid = [a for a in results if a["status"] == "valid"]
        return {"n": len(results), "valid": share(results, lambda a: a["status"] == "valid"),
                "beats_start": share(results, lambda a: a.get("beats_start", False)),
                "beats_first": share(results, lambda a: a.get("beats_first", False)),
                "tests": share(results, lambda a: a.get("reading") == "test"),
                "test_beats_start": share(results, lambda a: a.get("reading") == "test" and a.get("beats_start")),
                "missing_if_valid": share(valid, lambda a: a.get("reading") == "missing"),
                "same_as_existing": share(results, lambda a: a.get("same_as") is not None)}

    def paired(subset, a, b, field):
        both = [r for r in subset if a in r and b in r]
        hit = lambda r, arm: (r[arm].get("reading") == "test") if field == "tests" else bool(r[arm].get(field))
        return {"n": len(both), "both": sum(hit(r, a) and hit(r, b) for r in both),
                f"{a}_only": sum(hit(r, a) and not hit(r, b) for r in both),
                f"{b}_only": sum(hit(r, b) and not hit(r, a) for r in both)}

    def group(subset):
        judged = [r for r in subset if r["reading"] is not None]
        worse = [r for r in judged if not r["first_beats_start"]]
        result = {"first_versions": len(subset), "judged": len(judged), "readings": readings(judged),
                  "readings_if_worse": readings(worse),
                  "readings_if_better": readings([r for r in judged if r["first_beats_start"]])}
        for name, members in (("test", [r for r in worse if r["reading"] == "test"]),
                              ("not_test", [r for r in worse if r["reading"] != "test"])):
            result[f"worse_{name}"] = {"n": len(members), **{a: arm(members, a) for a in ARMS},
                                       "develop_best_of_3_beats_start": share(members,
                                                                              lambda r: r["develop_best_beats_start"])}
        not_test = [r for r in worse if r["reading"] != "test"]
        result["paired_not_test"] = {f"{a}_vs_{b}_{field}": paired(not_test, a, b, field)
                                     for a, b in (("complete", "resample"), ("complete", "develop"))
                                     for field in ("beats_start", "tests")}
        return result

    tasks = sorted({r["task"] for r in rows})
    return {"all": group(rows), **{task: group([r for r in rows if r["task"] == task]) for task in tasks}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, nargs="+")
    parser.add_argument("--slots", type=int, default=9, help="requests in flight per endpoint")
    parser.add_argument("--eval-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="cases per run")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args(argv)
    if not (args.run_dir or args.summary_only):
        parser.error("--run-dir is required unless --summary-only")
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "rows.jsonl"
    done = [json.loads(line) for line in path.open()] if path.exists() else []
    for row in done:
        for verdict in [row, *(row[a] for a in ARMS if "replies" in row.get(a, {}))]:
            reparse(verdict)
    if not args.summary_only:
        seen = {(r["run"], r["exploration"]) for r in done}
        settings = read_json(args.run_dir[0] / "run_config.json")
        endpoints = (settings.get("resource_scheduler") or {}).get("endpoints") or [settings["llm"]]
        clients = Clients(endpoints, args.slots, settings["method_params"]["output_tokens"])
        counter = Clients(endpoints[:1], 1, 1).pool.get()
        eval_lock, write_lock = threading.Lock(), threading.Lock()
        runs = [Run(d, clients, counter, args.eval_workers, eval_lock) for d in args.run_dir]
        queues = [[r for r in run.cases() if (run.dir.name, r["id"]) not in seen][:args.limit] for run in runs]
        # Interleave the runs, so that a partial result covers every task.
        jobs = [(run, q[i]) for i in range(max(map(len, queues), default=0)) for run, q in zip(runs, queues)
                if i < len(q)]
        print(f"{len(jobs)} cases to read, {len(done)} done", flush=True)

        def work(job):
            run, record = job
            try:
                row = run.case(record)
            except Exception as exc:  # one case's failure is reported, not fatal to the others
                print(f"{run.dir.name} exploration {record['id']}: {type(exc).__name__}: {exc}", flush=True)
                return
            with write_lock, path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
                done.append(row)
                print(f"{len(done)} {run.task} {record['id']} {row['reading']}", flush=True)

        with ThreadPoolExecutor(len(endpoints) * args.slots) as pool:
            list(pool.map(work, jobs))
    result = summary(done)
    (args.out / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["all"], indent=2))


if __name__ == "__main__":
    main()
