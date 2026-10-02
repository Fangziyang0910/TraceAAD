"""The population is the search's experience: programs and the generation events between them.

A program is identified by its normalized code and is evaluated once; failed
programs are programs too. Every model generation is an event that starts from
a program (or from nothing, at initialization) and ends in a program that is
new or already known, or in no program at all. Starting points are chosen by
training quality weighted by the experience of each program, and every
generation sees the measured outcomes around its starting point.
"""

from dataclasses import asdict
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import time

from core import SecureEvaluator
from traceaad.v10_13.storage import write_json

from .canonical import canonical, key
from .config import Config
from .delivery import DeliveryError, SourceError, extract_idea, parse_response
from .evaluation import SeededEvaluation, fingerprint, protocol_identity
from .history import verdict
from .prompts import ContextTooLong, PromptBuilder, idea_view
from .selection import OPERATORS, better, choose_reference, sample_parent
from .state import Facts

FAILURES = ("invalid_source", "runtime_error", "invalid_output", "timeout")
TRIED_OUT = 15  # attempts without improvement after which a program almost never improves (diagnostics)


def now():
    return datetime.now().isoformat(timespec="seconds")


def _tuple_tree(value):
    return tuple(_tuple_tree(v) for v in value) if isinstance(value, list) else value


def _service_error(exc):
    status = getattr(exc, "status_code", None)
    return (status == 429 or isinstance(status, int) and status >= 500 or
            isinstance(exc, (ConnectionError, TimeoutError, OSError)) or
            any(word in type(exc).__name__.lower() for word in ("connection", "timeout", "ratelimit")))


def clean_traceback(text):
    """Drop the frames of the call counter, which wraps the candidate during evaluation."""
    lines, kept, skip = (text or "").splitlines(), [], False
    for index, line in enumerate(lines):
        if skip:
            skip = False
            continue
        following = lines[index + 1] if index + 1 < len(lines) else ""
        if line.lstrip().startswith("File ") and ("probe.py" in line or "_traceaad_probe_" in following):
            skip = True
            continue
        kept.append(line)
    return "\n".join(kept)


def failing_line(error, code):
    """The program line where a runtime error was raised: the innermost frame in the program.

    The evaluated text is the program followed by the call counter, so frames
    past the program's last line belong to the counter.
    """
    lines = (code or "").splitlines()
    numbers = [int(n) for n in re.findall(r'File "<string>", line (\d+)', error or "") if 0 < int(n) <= len(lines)]
    return lines[numbers[-1] - 1].strip()[:160] if numbers else None


class TraceAADV1016:
    METHOD = "v1016"

    def __init__(self, *, evaluation, llm, run_dir, config=None, task=None,
                 selection_evaluation=None):
        self.config = config or Config()
        if (getattr(llm, "chars_per_token", None) is not None or
                not callable(getattr(llm, "count_prompt_tokens", None)) or
                not callable(getattr(llm, "count_tokens", None))):
            raise ValueError("V10.16 requires the serving tokenizer, not character estimates")
        self.llm, self.task = llm, task
        self.template = str(evaluation.template_program)
        self.facts = Facts(run_dir)
        self.run_dir = Path(run_dir)
        self.archive = self.facts.tables["node"]  # valid programs: the candidate starting points
        self.attempts_table = self.facts.tables["attempt"]
        self.programs = {}  # every program, valid or failed, by id
        self.key_index = {}  # normalized code key -> program id
        for node in self.archive.values():
            self.programs[node["id"]] = node
            self.key_index[node["key"]] = node["id"]
        for attempt in self.attempts_table.values():
            if attempt["status"] in FAILURES and attempt.get("program_id") == attempt["id"]:
                self._add_failed(attempt)
        self.prompts = PromptBuilder(llm, task, evaluation, self.programs, self.attempts_table, self.config)
        self.protocol, self.environment = protocol_identity(evaluation, self.config.evaluation_seeds, "search")
        self.seeded = SeededEvaluation(evaluation)
        self.evaluator = SecureEvaluator(self.seeded)
        self.selection_seeded = self.selection_evaluator = self.selection_protocol = None
        if selection_evaluation is not None:
            if str(selection_evaluation.template_program) != self.template:
                raise ValueError("selection evaluator has a different task interface")
            self.selection_protocol, selection_environment = protocol_identity(
                selection_evaluation, self.config.evaluation_seeds, "selection")
            search_data = getattr(evaluation, "_datasets", None)
            selection_data = getattr(selection_evaluation, "_datasets", None)
            if (selection_environment == self.environment or
                    search_data is not None and selection_data is not None and
                    fingerprint(search_data) == fingerprint(selection_data)):
                raise ValueError("selection evaluation must use a distinct frozen dataset/protocol")
            self.selection_seeded = SeededEvaluation(selection_evaluation)
            self.selection_evaluator = SecureEvaluator(self.selection_seeded)
        self.parent_rng = random.Random(f"v10.16:{self.config.seed}:parent")
        self.action_rng = random.Random(f"v10.16:{self.config.seed}:action")
        self.reference_rng = random.Random(f"v10.16:{self.config.seed}:reference")
        self.phase = "roots"
        self.attempts = 0
        self.init_attempts = 0
        self.model_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.evaluation_calls = 0
        self.service_failures = 0
        self.too_long = set()
        self.finalists = []
        self.selection_results = []
        self.pending = None
        self.started_at = now()
        self.elapsed = 0.0
        self._clock = time.monotonic()
        repo = Path(__file__).resolve().parents[2]
        implementation_files = [*Path(__file__).parent.glob("*.py"),
                                repo / "core" / "llm.py", repo / "core" / "evaluate.py",
                                repo / "traceaad" / "v10_13" / "parsing.py",
                                repo / "traceaad" / "v10_13" / "storage.py"]
        source_identity = {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(set(implementation_files))}
        self.identity = json.loads(json.dumps({
            "config": asdict(self.config), "task": task, "protocol": self.protocol,
            "selection_protocol": self.selection_protocol,
            "implementation": fingerprint(source_identity),
            "model": {name: getattr(llm, name, None) for name in
                      ("model", "temperature", "top_p", "enable_thinking", "chars_per_token", "stop", "extra_body")},
        }))
        if self.facts.state:
            self._restore(self.facts.state)

    # ---------- state ----------

    def _state(self):
        return {"identity": self.identity, "phase": self.phase, "attempts": self.attempts,
                "init_attempts": self.init_attempts, "model_calls": self.model_calls,
                "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "evaluation_calls": self.evaluation_calls, "service_failures": self.service_failures,
                "too_long": sorted(self.too_long),
                "finalists": self.finalists, "selection_results": self.selection_results,
                "pending": self.pending, "started_at": self.started_at,
                "elapsed": self.elapsed + time.monotonic() - self._clock,
                "rng": {"parent": self.parent_rng.getstate(), "action": self.action_rng.getstate(),
                        "reference": self.reference_rng.getstate()}}

    def _save(self):
        self.facts.checkpoint(self._state())

    def _restore(self, state):
        if state["identity"] != self.identity:
            raise ValueError("resume configuration, model, source or evaluation protocol changed")
        if state.get("pending"):
            raise RuntimeError("uncertain external result; refusing to replay a charged request/evaluation")
        for name in ("phase", "attempts", "init_attempts", "model_calls", "input_tokens",
                     "output_tokens", "evaluation_calls", "service_failures", "finalists",
                     "selection_results", "started_at", "elapsed"):
            setattr(self, name, state[name])
        self.too_long = set(state["too_long"])
        for name in ("parent", "action", "reference"):
            getattr(self, name + "_rng").setstate(_tuple_tree(state["rng"][name]))
        self._clock = time.monotonic()

    def _add_failed(self, attempt):
        """A failed program, rebuilt from the attempt that first produced its code."""
        parent = self.programs.get(attempt["parent_id"])
        program = {"id": attempt["id"], "key": attempt["key"], "code": attempt["code"],
                   "parent_id": attempt["parent_id"], "action": attempt["executed_action"],
                   "reference_id": attempt["reference_id"], "idea": attempt["idea"],
                   "depth": parent["depth"] + 1 if parent else 0,
                   "repaired": attempt["repair_of"] is not None, "valid": False,
                   "failure": {"kind": attempt["status"], "error": attempt["error"],
                               "seconds": attempt.get("seconds"), "calls": attempt.get("calls"),
                               "function_seconds": attempt.get("function_seconds"),
                               "call_running": attempt.get("call_running", False),
                               "line": failing_line(attempt["error"], attempt["code"])
                               if attempt["status"] in ("runtime_error", "invalid_output") else None}}
        self.programs[program["id"]] = program
        self.key_index[program["key"]] = program["id"]
        return program

    # ---------- model and evaluator ----------

    def _generate(self, request):
        """Service errors are logged and retried without creating attempts."""
        failures = 0
        while True:
            rid = self.model_calls + 1
            self.model_calls = rid
            self.pending = {"kind": "generation", "request_id": rid}
            self._save()
            self.facts.add("request", {"id": rid, **request, "created_at": now(),
                                       "output_tokens": self.config.output_tokens})
            started = time.monotonic()
            try:
                details = self.llm.draw_sample_with_details(request["prompt"], max_tokens=self.config.output_tokens)
                error = None
            except Exception as exc:
                details = {}
                error = f"{type(exc).__name__}: {exc}"
                transient = _service_error(exc)
            usage = details.get("usage") or {}
            self.input_tokens += usage.get("prompt_tokens") if isinstance(usage.get("prompt_tokens"), int) else request["input_tokens"]
            self.output_tokens += usage.get("completion_tokens") if isinstance(usage.get("completion_tokens"), int) else (
                self.config.output_tokens if not error else 0)
            self.facts.record_call({"request_id": rid, "prompt": request["prompt"],
                                    "response": details.get("content", ""),
                                    "finish_reason": details.get("finish_reason"), "usage": usage,
                                    "error": error, "seconds": time.monotonic() - started,
                                    "model": details.get("model")})
            if not error:
                self.pending = {"kind": "delivered_generation", "request_id": rid}
                self._save()
                return rid, details
            self.pending = None
            self._save()
            if not transient:
                raise RuntimeError(f"non-service model failure: {error}")
            self.service_failures += 1
            self._save()
            failures += 1
            if failures >= 3:
                raise RuntimeError(f"model service unavailable after 3 calls: {error}")
            time.sleep(min(2 ** failures, 8))

    def _evaluate(self, code, source_key, *, role="search"):
        """Mean score over seeds, with the calls and time measured inside the function.

        On failure the measurement of the failed seed is returned: for a
        timeout it says how far the evaluation got before the limit.
        """
        seeded, evaluator = ((self.seeded, self.evaluator) if role == "search"
                             else (self.selection_seeded, self.selection_evaluator))
        protocol = self.protocol if role == "search" else self.selection_protocol
        values, ids, seconds, calls, inside = [], [], 0.0, 0, 0.0
        previous_pending = self.pending
        for seed in self.config.evaluation_seeds:
            eid = len(self.facts.tables["evaluation"]) + 1
            self.pending = {"kind": "evaluation", "id": eid, "role": role, "key": source_key}
            self._save()
            started = time.monotonic()
            seeded.reset()
            result = evaluator.evaluate_program_with_details(self.template, source=code, seed=seed)
            elapsed = time.monotonic() - started
            measured = seeded.measured(until=started + elapsed)
            if result.failure_kind == "timeout" and evaluator._evaluator.timeout_seconds is not None:
                # The evaluation process is stopped after the limit; time inside
                # a running call ends at the limit, not when stopping finished.
                measured["function_seconds"] = min(measured["function_seconds"],
                                                   float(evaluator._evaluator.timeout_seconds))
            self.evaluation_calls += 1
            value = result.result
            valid = (isinstance(value, dict) and type(value.get("score")) in (int, float)
                     and math.isfinite(value["score"]))
            self.facts.add("evaluation", {"id": eid, "key": source_key, "role": role,
                "protocol": protocol, "seed": seed, "valid": valid,
                "score": value["score"] if valid else None, "failure_kind": result.failure_kind,
                "error_type": result.error_type, "error": result.error,
                "traceback": result.traceback, "seconds": elapsed,
                "cpu_seconds": result.cpu_seconds, **measured})
            self.pending = previous_pending
            self._save()
            ids.append(eid)
            if not valid:
                kind = ("timeout" if result.failure_kind == "timeout" else
                        "invalid_output" if result.failure_kind == "invalid_result" else "runtime_error")
                error = clean_traceback(result.traceback or result.error or result.failure_kind or "invalid evaluation")
                return None, ids, kind, error, {"seconds": elapsed, **measured}
            measured.pop("call_running")
            values.append(float(value["score"]))
            seconds += elapsed
            calls += measured["calls"]
            inside += measured["function_seconds"]
        n = len(self.config.evaluation_seeds)
        return statistics.fmean(values), ids, None, None, {
            "seconds": seconds / n, "calls": round(calls / n), "function_seconds": inside / n}

    # ---------- one generation event ----------

    def _attempt(self, request, *, parent=None, action="Init", reference=None,
                 selection=None, repair_of=None):
        rid, details = self._generate(request)
        self.attempts += 1
        if self.phase == "roots":
            self.init_attempts += 1
        aid = self.attempts
        self.pending = {"kind": "candidate", "id": aid, "request_id": rid}
        self._save()
        response = details.get("content", "")
        finish = details.get("finish_reason")
        idea = extract_idea(response) if isinstance(response, str) else ""
        executed = request["action"]
        record = {"id": aid, "request_id": rid, "parent_id": parent["id"] if parent else None,
                  "action": action, "executed_action": executed,
                  "reference_id": reference["id"] if reference else None,
                  "repair_of": repair_of, "repaired": repair_of is not None,
                  "selection": selection, "history_edge_ids": request["history_edge_ids"],
                  "reference_history_edge_ids": request["reference_history_edge_ids"],
                  "attempt_ids_shown": request.get("attempt_ids", []),
                  "progress_ids_shown": request.get("progress_ids", []),
                  "trims": request["trims"], "status": None, "idea": idea,
                  "idea_display": idea_view(idea), "raw_code": None, "completed_code": None,
                  "code": None, "key": None, "program_id": None, "new_program": False,
                  "score": None, "fitness": None, "seconds": None, "calls": None,
                  "function_seconds": None, "evaluation_ids": [], "error": None, "created_at": now()}
        normal = None
        try:
            code, idea, delivery = parse_response(response, finish, self.template)
            record["delivery"] = delivery
        except DeliveryError as exc:
            record.update(status="delivery_failed", error=str(exc))
        except SourceError as exc:
            # An unusable program is still a program: it can be repaired, and
            # its code identifies it when the same text comes back.
            shown = exc.code or exc.submitted_code or ""
            try:
                normal = canonical(shown)
            except (SyntaxError, ValueError):
                normal = shown.rstrip() + "\n"
            record.update(status="invalid_source", error=str(exc), raw_code=exc.submitted_code,
                          completed_code=exc.code, code=normal, key=key(normal))
        else:
            record["idea"], record["idea_display"] = idea, idea_view(idea)
            record["raw_code"], record["completed_code"] = delivery["submitted_code"], code
            normal = canonical(code)
            record["code"], record["key"] = normal, key(normal)
        if record["key"] is not None and record["key"] in self.key_index:
            # Identical code has a known result: the event links to the known program.
            known = self.programs[self.key_index[record["key"]]]
            record["program_id"] = known["id"]
            if reference is not None and known["id"] == reference["id"]:
                record["status"] = "copied_reference"
            elif known["valid"]:
                record["status"] = "duplicate"
            else:
                record["status"] = "known_failure"
            record["error"] = None
        elif record["status"] == "invalid_source":
            record.update(program_id=aid, new_program=True)
        elif record["key"] is not None:
            if record["key"] not in self.facts.tables["artifact"]:
                self.facts.add("artifact", {"id": record["key"], "code": normal, "environment": self.environment})
            fitness, eval_ids, failure, error, measured = self._evaluate(normal, record["key"])
            record.update(evaluation_ids=eval_ids, program_id=aid, new_program=True, **measured)
            if failure:
                record.update(status=failure, error=error)
            else:
                score = fitness if self.prompts.higher_is_better else -fitness
                node = {"id": aid, "key": record["key"], "code": normal, "fitness": fitness,
                        "score": score, "parent_id": parent["id"] if parent else None,
                        "action": executed, "reference_id": reference["id"] if reference else None,
                        "idea": idea, "depth": parent["depth"] + 1 if parent else 0,
                        "repaired": repair_of is not None, "valid": True,
                        "eval_seconds": measured["seconds"], "calls": measured["calls"],
                        "function_seconds": measured["function_seconds"], "attempt_id": aid}
                self.facts.add("node", node)
                self.programs[aid] = node
                self.key_index[node["key"]] = aid
                record.update(status="valid", score=score, fitness=fitness,
                              same_as_parent=bool(parent and parent.get("valid")) and
                              verdict(parent["fitness"], fitness, True) == "same score")
        self.facts.add("attempt", record)
        failed = self._add_failed(record) if record["status"] in FAILURES and record["new_program"] else None
        self.facts._append({"kind": "candidate", "candidate_id": aid, "budget_used": aid,
                            "status": record["status"], "fitness": record["fitness"],
                            "evaluation_id": self.evaluation_calls})
        self.pending = None
        self._save()
        if (failed is not None and repair_of is None and self.attempts < self.config.budget
                and (self.phase != "roots" or self.init_attempts < self.config.init_attempt_limit)):
            try:
                repair_request = self.prompts.repair(failed, self._error_text(failed))
            except ContextTooLong:
                return None
            return self._attempt(repair_request, parent=failed, action="Repair", selection=selection,
                                 repair_of=aid)
        return self.archive.get(record["program_id"]) if record["status"] == "valid" else None

    def _error_text(self, failed):
        failure = failed["failure"]
        if failure["kind"] == "timeout":
            text = "The evaluation " + self.prompts.failure(failed) + "."
            source = self.programs.get(failed["parent_id"])
            if source is not None and source.get("valid") and source.get("calls") is not None:
                text += (f" The algorithm it was developed from completes the evaluation with {source['calls']} "
                         f"calls in about {max(source.get('eval_seconds') or 0.0, 0.1):.1f} s, "
                         f"{self.prompts._seconds(source.get('function_seconds') or 0.0)} inside the function.")
            return text
        if failure["kind"] == "invalid_source":
            return f"The program could not be used: {failure['error']}"
        if failure["kind"] == "runtime_error":
            return "\n".join((failure["error"] or "").splitlines()[-15:])[-1500:]
        return failure["error"] or "Invalid output"

    # ---------- phases ----------

    def _is_root(self, node):
        """An initial algorithm: no valid program precedes it on its formation path
        (a repaired initial program starts from its failed first version)."""
        parent = self.programs.get(node["parent_id"])
        while parent is not None:
            if parent["valid"]:
                return False
            parent = self.programs.get(parent["parent_id"])
        return True

    def _roots(self):
        roots = [node for node in self.archive.values() if self._is_root(node)]
        if (len(roots) >= self.config.roots or
                self.init_attempts >= self.config.init_attempt_limit or self.attempts >= self.config.budget):
            self.phase = "search" if roots else "no_valid_root"
            self._save()
            return
        self._attempt(self.prompts.initial(roots))

    def _search(self):
        if self.attempts >= self.config.budget:
            self.phase = "freeze"
            self._save()
            return
        eligible = [node for node in self.archive.values() if node["id"] not in self.too_long]
        if not eligible:
            self.phase = "freeze"
            self._save()
            return
        parent, selection = sample_parent(eligible, self.attempts_table, self.programs, self.parent_rng)
        sampled = self.action_rng.choices(list(OPERATORS), [.45, .30, .25])[0]
        action = sampled
        reference, reference_selection = None, None
        flags = []
        if action == "Crossover":
            reference, reference_selection = choose_reference(parent, self.archive, self.reference_rng)
            if reference is None:
                action = "Refine"
                flags.append("crossover_fallback")
        try:
            request = self.prompts.build(action, parent, reference=reference)
        except ContextTooLong:
            self.too_long.add(parent["id"])
            self._save()
            return
        if request["action"] == "Refine" and action == "Crossover":
            reference = None
            flags.append("crossover_context_fallback")
        request.update(sampled_action=sampled, fallbacks=flags, parent_id=parent["id"],
                       reference_id=reference["id"] if reference else None, selection=selection,
                       reference_selection=reference_selection)
        self._attempt(request, parent=parent, action=request["action"], reference=reference,
                      selection=selection)

    def _ranking(self):
        return [n["id"] for n in sorted(self.archive.values(), key=lambda n: (-n["fitness"], n["id"]))]

    def _freeze(self):
        self.finalists = self._ranking()[:self.config.final_candidates]
        write_json(self.run_dir / "finalists.json", {
            "protocol": self.protocol, "selection_protocol": self.selection_protocol,
            "frozen_at": now(), "candidates": [self.archive[i] for i in self.finalists]})
        self.phase = "selection" if self.selection_evaluator else "search_complete"
        self._save()

    def _select(self):
        """Evaluate finalists on the selection set; a finalist that fails there is
        replaced by the next program in the training ranking, up to as many
        replacements as there are finalists."""
        if len(self.selection_results) < len(self.finalists):
            node = self.archive[self.finalists[len(self.selection_results)]]
            self.pending = {"kind": "selection_candidate", "node_id": node["id"]}
            self._save()
            fitness, ids, failure, error, measured = self._evaluate(node["code"], node["key"], role="selection")
            self.selection_results.append({"node_id": node["id"], "fitness": fitness, "evaluation_ids": ids,
                                           "failure": failure, "error": error, **measured})
            if failure and len(self.finalists) < 2 * self.config.final_candidates:
                remaining = [i for i in self._ranking() if i not in self.finalists]
                if remaining:
                    self.finalists.append(remaining[0])
            self.pending = None
            self._save()
            return
        valid = [r for r in self.selection_results if r["fitness"] is not None]
        if not valid:
            self.phase = "selection_failed"
        else:
            selected = max(valid, key=lambda r: (r["fitness"], -self.finalists.index(r["node_id"])))
            node = self.archive[selected["node_id"]]
            (self.run_dir / "best_program.py").write_text(node["code"], encoding="utf-8")
            write_json(self.run_dir / "selection.json", {
                "selection_protocol": self.selection_protocol, "results": self.selection_results,
                "finalists": self.finalists, "selected_node": node["id"], "selected_key": node["key"]})
            self.phase = "finished"
        self._save()

    # ---------- reporting ----------

    def _summary(self, status, error=None):
        best = None
        if self.archive:
            best = max(self.archive.values(), key=lambda n: (n["fitness"], -n["id"]))
        if self.phase == "finished":
            valid = [r for r in self.selection_results if r["fitness"] is not None]
            selected = max(valid, key=lambda r: (r["fitness"], -self.finalists.index(r["node_id"])))
            best = self.archive[selected["node_id"]]
        best_export = {**best, "selection_fitness": next((r["fitness"] for r in self.selection_results
                       if r["node_id"] == best["id"]), None)} if best else None
        diagnostics = self._diagnostics()
        write_json(self.run_dir / "diagnostics.json", diagnostics)
        summary = {"status": status, "phase": self.phase, "method": self.METHOD,
                   "budget": self.config.budget, "budget_used": self.attempts,
                   "init_attempts": self.init_attempts, "num_nodes": len(self.archive),
                   "num_failed_programs": sum(not p["valid"] for p in self.programs.values()),
                   "num_roots": sum(self._is_root(n) for n in self.archive.values()),
                   "model_calls": self.model_calls, "evaluation_calls": self.evaluation_calls,
                   "search_evaluations": sum(e["role"] == "search" for e in self.facts.tables["evaluation"].values()),
                   "selection_evaluations": sum(e["role"] == "selection" for e in self.facts.tables["evaluation"].values()),
                   "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                   "service_failures": self.service_failures, "too_long": sorted(self.too_long),
                   "started_at": self.started_at, "finished_at": now(),
                   "seconds": self.elapsed + time.monotonic() - self._clock,
                   "best": best_export, "finalists": self.finalists,
                   "diagnostics": diagnostics, "error": error}
        self.facts.save_summary(summary)
        return summary

    def _diagnostics(self):
        attempts = sorted(self.attempts_table.values(), key=lambda a: a["id"])
        counts = dict(Counter(a["status"] for a in attempts))
        repairs = {a["repair_of"]: a for a in attempts if a["repair_of"] is not None}
        actions = {}
        for name in (*OPERATORS, "Repair"):
            proposed = [a for a in attempts if a["executed_action"] == name]
            new = [a for a in proposed if a["status"] == "valid"]
            started = [(a, self.programs.get(a["parent_id"])) for a in new]
            improved = sum(1 for a, p in started if p is not None and p["valid"] and better(a["fitness"], p["fitness"]))
            actions[name] = {"attempts": len(proposed), "new_valid": len(new), "improved_over_start": improved,
                             "improvement_per_attempt": improved / len(proposed) if proposed else None}
        frontier, frontiers = -math.inf, Counter()
        explore_late = 0
        for node in sorted(self.archive.values(), key=lambda n: n["id"]):
            if node["fitness"] > frontier:
                frontiers[node["action"]] += 1
                explore_late += node["action"] == "Explore" and node["id"] > 300
            frontier = max(frontier, node["fitness"])
        # How much of the budget went to programs that had already been tried
        # many times without improving (the waste experience should remove).
        tried, improved, on_tried_out = Counter(), Counter(), 0
        operator_attempts = [a for a in attempts if a["repair_of"] is None and a["action"] in OPERATORS]
        for a in operator_attempts:
            source = self.programs.get(a["parent_id"])
            if source is None or not source["valid"]:
                continue
            on_tried_out += tried[source["id"]] >= TRIED_OUT and improved[source["id"]] == 0
            tried[source["id"]] += 1
            final = repairs.get(a["id"], a)
            result = self.programs.get(final.get("program_id"))
            if final["status"] == "valid" and result is not None and better(result["fitness"], source["fitness"]):
                improved[source["id"]] += 1
        quarters = []
        for q in range(4):
            group = [a for a in operator_attempts
                     if q * self.config.budget // 4 < a["id"] <= (q + 1) * self.config.budget // 4]
            quarters.append({"attempts": len(group),
                             "timeout": sum(a["status"] == "timeout" for a in group) / len(group) if group else None,
                             "error": sum(a["status"] in {"runtime_error", "invalid_output", "invalid_source"}
                                          for a in group) / len(group) if group else None})
        search = [e for e in self.facts.tables["evaluation"].values() if e["role"] == "search"]
        cpu = sorted(e["cpu_seconds"] for e in search
                     if e["valid"] and isinstance(e.get("cpu_seconds"), (int, float)))
        best = max(self.archive.values(), key=lambda n: (n["fitness"], -n["id"])) if self.archive else None
        return {"status_counts": counts, "actions": actions,
                "repair_success": sum(a["status"] == "valid" for a in repairs.values()),
                "repair_attempts": len(repairs),
                "crossover_copy_rate": counts.get("copied_reference", 0) / actions["Crossover"]["attempts"]
                if actions["Crossover"]["attempts"] else None,
                "new_frontiers_by_action": dict(frontiers),
                "explore_new_frontiers_after_300": explore_late,
                "attempts_on_tried_out_programs": on_tried_out / len(operator_attempts) if operator_attempts else None,
                "failure_rates_by_quarter": quarters,
                "failed_programs": sum(not p["valid"] for p in self.programs.values()),
                "valid_cpu_seconds": {"median": cpu[len(cpu) // 2], "p95": cpu[int(len(cpu) * .95)],
                                      "max": cpu[-1]} if cpu else None,
                "best_training_depth": best["depth"] if best else None,
                "too_long_nodes": len(self.too_long)}

    def run(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        try:
            while self.phase in {"roots", "search", "freeze", "selection"}:
                if self.phase == "roots":
                    self._roots()
                elif self.phase == "search":
                    self._search()
                elif self.phase == "freeze":
                    self._freeze()
                else:
                    self._select()
        except RuntimeError as exc:
            if "model service unavailable" in str(exc):
                return self._summary("service_unavailable", str(exc))
            raise
        return self._summary(self.phase)
