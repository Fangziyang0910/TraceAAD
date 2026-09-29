"""Immediate quality competition; formation paths affect generation only."""

from dataclasses import asdict
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

from core import SecureEvaluator
from traceaad.v10_13.storage import write_json

from .canonical import canonical, key
from .config import Config
from .delivery import DeliveryError, SourceError, extract_idea, parse_response
from .evaluation import SeededEvaluation, fingerprint, protocol_identity
from .prompts import ContextTooLong, PromptBuilder, idea_view
from .selection import choose_reference, sample_parent
from .state import Facts


def now():
    return datetime.now().isoformat(timespec="seconds")


def _tuple_tree(value):
    return tuple(_tuple_tree(v) for v in value) if isinstance(value, list) else value


def _service_error(exc):
    status = getattr(exc, "status_code", None)
    return (status == 429 or isinstance(status, int) and status >= 500 or
            isinstance(exc, (ConnectionError, TimeoutError, OSError)) or
            any(word in type(exc).__name__.lower() for word in ("connection", "timeout", "ratelimit")))


class TraceAADV1015:
    METHOD = "v1015"

    def __init__(self, *, evaluation, llm, run_dir, config=None, task=None,
                 selection_evaluation=None):
        self.config = config or Config()
        if (getattr(llm, "chars_per_token", None) is not None or
                not callable(getattr(llm, "count_prompt_tokens", None)) or
                not callable(getattr(llm, "count_tokens", None))):
            raise ValueError("V10.15 requires the serving tokenizer, not character estimates")
        self.llm, self.task = llm, task
        self.template = str(evaluation.template_program)
        self.facts = Facts(run_dir)
        self.run_dir = Path(run_dir)
        self.archive = self.facts.tables["node"]
        self.prompts = PromptBuilder(llm, task, evaluation, self.archive, self.config)
        self.protocol, self.environment = protocol_identity(evaluation, self.config.evaluation_seeds, "search")
        self.evaluator = SecureEvaluator(SeededEvaluation(evaluation))
        self.selection_evaluator = None
        self.selection_protocol = None
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
            self.selection_evaluator = SecureEvaluator(SeededEvaluation(selection_evaluation))
        self.parent_rng = random.Random(f"v10.15:{self.config.seed}:parent")
        self.action_rng = random.Random(f"v10.15:{self.config.seed}:action")
        self.reference_rng = random.Random(f"v10.15:{self.config.seed}:reference")
        self.phase = "roots"
        self.attempts = 0
        self.init_attempts = 0
        self.model_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.evaluation_calls = 0
        self.service_failures = 0
        self.failed_keys = set()
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

    def _state(self):
        return {"identity": self.identity, "phase": self.phase, "attempts": self.attempts,
                "init_attempts": self.init_attempts, "model_calls": self.model_calls,
                "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "evaluation_calls": self.evaluation_calls, "service_failures": self.service_failures,
                "failed_keys": sorted(self.failed_keys), "too_long": sorted(self.too_long),
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
        self.failed_keys = set(state["failed_keys"])
        self.too_long = set(state["too_long"])
        for name in ("parent", "action", "reference"):
            getattr(self, name + "_rng").setstate(_tuple_tree(state["rng"][name]))
        self._clock = time.monotonic()

    def _generate(self, request):
        """Service errors are logged and retried without creating candidate attempts."""
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
        evaluator = self.evaluator if role == "search" else self.selection_evaluator
        protocol = self.protocol if role == "search" else self.selection_protocol
        values, ids = [], []
        previous_pending = self.pending
        for seed in self.config.evaluation_seeds:
            eid = len(self.facts.tables["evaluation"]) + 1
            self.pending = {"kind": "evaluation", "id": eid, "role": role, "key": source_key}
            self._save()
            started = time.monotonic()
            result = evaluator.evaluate_program_with_details(self.template, source=code, seed=seed)
            self.evaluation_calls += 1
            value = result.result
            valid = (isinstance(value, dict) and type(value.get("score")) in (int, float)
                     and math.isfinite(value["score"]))
            self.facts.add("evaluation", {"id": eid, "key": source_key, "role": role,
                "protocol": protocol, "seed": seed, "valid": valid,
                "score": value["score"] if valid else None, "failure_kind": result.failure_kind,
                "error_type": result.error_type, "error": result.error,
                "traceback": result.traceback, "seconds": time.monotonic() - started})
            self.pending = previous_pending
            self._save()
            ids.append(eid)
            if not valid:
                kind = ("timeout" if result.failure_kind == "timeout" else
                        "invalid_output" if result.failure_kind == "invalid_result" else "runtime_error")
                return None, ids, kind, result.traceback or result.error or result.failure_kind or "invalid evaluation"
            values.append(float(value["score"]))
        return statistics.fmean(values), ids, None, None

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
        record = {"id": aid, "request_id": rid, "parent_id": parent["id"] if parent else None,
                  "action": action, "executed_action": request["action"],
                  "reference_id": reference["id"] if reference else None,
                  "repair_of": repair_of, "repaired": repair_of is not None,
                  "selection": selection, "history_edge_ids": request["history_edge_ids"],
                  "trims": request["trims"], "status": None, "idea": idea,
                  "idea_display": idea_view(idea), "raw_code": None, "completed_code": None,
                  "code": None,
                  "key": None, "score": None, "fitness": None,
                  "evaluation_ids": [], "error": None, "created_at": now()}
        failed_code = None
        try:
            code, idea, delivery = parse_response(response, finish, self.template)
            record["delivery"] = delivery
        except DeliveryError as exc:
            record.update(status="delivery_failed", error=str(exc))
        except SourceError as exc:
            failed_code = exc.code
            record.update(status="invalid_source", error=str(exc), raw_code=exc.submitted_code,
                          completed_code=failed_code)
        else:
            record["idea"] = idea
            record["idea_display"] = idea_view(idea)
            record["raw_code"] = delivery["submitted_code"]
            record["completed_code"] = code
            normal = canonical(code)
            source_key = key(normal)
            record["code"], record["key"] = normal, source_key
            valid = {node["key"] for node in self.archive.values()}
            if reference is not None and source_key == reference["key"]:
                record["status"] = "copied_reference"
            elif source_key in valid:
                record["status"] = "duplicate"
            elif source_key in self.failed_keys:
                record["status"] = "known_failure"
            else:
                if source_key not in self.facts.tables["artifact"]:
                    self.facts.add("artifact", {"id": source_key, "code": normal,
                                                "environment": self.environment})
                fitness, eval_ids, failure, error = self._evaluate(normal, source_key)
                record["evaluation_ids"] = eval_ids
                if failure:
                    record.update(status=failure, error=error)
                    self.failed_keys.add(source_key)
                    failed_code = normal
                else:
                    score = fitness if self.prompts.higher_is_better else -fitness
                    node = {"id": aid, "key": source_key, "code": normal, "fitness": fitness,
                            "score": score, "parent_id": parent["id"] if parent else None,
                            "action": action, "reference_id": reference["id"] if reference else None,
                            "idea": idea, "depth": parent["depth"] + 1 if parent else 0,
                            "repaired": repair_of is not None, "attempt_id": aid}
                    self.facts.add("node", node)
                    record.update(status="valid", score=score, fitness=fitness, node_id=aid)
        self.facts.add("attempt", record)
        self.facts._append({"kind": "candidate", "candidate_id": aid, "budget_used": aid,
                            "status": record["status"], "fitness": record["fitness"],
                            "evaluation_id": self.evaluation_calls})
        self.pending = None
        self._save()
        if (record["status"] in {"invalid_source", "runtime_error", "invalid_output", "timeout"}
                and repair_of is None and self.attempts < self.config.budget
                and (self.phase != "roots" or self.init_attempts < self.config.init_attempt_limit)):
            if record["status"] == "timeout":
                error_text = f"The evaluation did not finish within {self.prompts.evaluation.timeout_seconds:g} seconds."
            elif record["status"] == "invalid_source":
                error_text = f"The program could not be used: {record['error']}"
            elif record["status"] == "runtime_error":
                error_text = "\n".join((record["error"] or "").splitlines()[-15:])[-1500:]
            else:
                error_text = record["error"] or "Invalid output"
            try:
                shown_code = failed_code or record["raw_code"] or ""
                try:
                    shown_code = canonical(shown_code)
                except SyntaxError:
                    pass
                repair_request = self.prompts.repair(shown_code,
                                                     idea, error_text, parent=parent)
            except ContextTooLong:
                return self.archive.get(aid)
            return self._attempt(repair_request, parent=parent, action=action,
                                 reference=reference, selection=selection, repair_of=aid)
        return self.archive.get(aid)

    def _roots(self):
        roots = [node for node in self.archive.values() if node["parent_id"] is None]
        if (len(roots) >= self.config.roots or
                self.init_attempts >= self.config.init_attempt_limit or self.attempts >= self.config.budget):
            self.phase = "search" if roots else "no_valid_root"
            self._save()
            return
        request = self.prompts.initial(roots)
        self._attempt(request)

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
        parent, selection = sample_parent(eligible, self.parent_rng)
        sampled = self.action_rng.choices(["Refine", "Explore", "Crossover"], [.45, .30, .25])[0]
        action = sampled
        reference, reference_selection = None, None
        flags = []
        if action == "Crossover":
            reference, reference_selection = choose_reference(parent, self.archive, self.reference_rng)
            if reference is None:
                action = "Refine"
                flags.append("crossover_fallback")
        best_score = max(self.archive.values(), key=lambda n: n["fitness"])["score"]
        try:
            request = self.prompts.build(action, parent, reference=reference, best_score=best_score)
        except ContextTooLong:
            self.too_long.add(parent["id"])
            self._save()
            return
        if request["action"] == "Refine" and action == "Crossover":
            reference = None
            flags.append("crossover_context_fallback")
        request["sampled_action"] = sampled
        request["fallbacks"] = flags
        request["parent_id"] = parent["id"]
        request["reference_id"] = reference["id"] if reference else None
        request["selection"] = selection
        request["reference_selection"] = reference_selection
        self._attempt(request, parent=parent, action=request["action"], reference=reference,
                      selection=selection)

    def _freeze(self):
        ordered = sorted(self.archive.values(), key=lambda n: (-n["fitness"], n["id"]))
        self.finalists = [n["id"] for n in ordered[:self.config.final_candidates]]
        write_json(self.run_dir / "finalists.json", {
            "protocol": self.protocol, "selection_protocol": self.selection_protocol,
            "frozen_at": now(), "candidates": [self.archive[i] for i in self.finalists]})
        self.phase = "selection" if self.selection_evaluator else "search_complete"
        self._save()

    def _select(self):
        if len(self.selection_results) < len(self.finalists):
            node = self.archive[self.finalists[len(self.selection_results)]]
            self.pending = {"kind": "selection_candidate", "node_id": node["id"]}
            self._save()
            fitness, ids, failure, error = self._evaluate(node["code"], node["key"], role="selection")
            self.selection_results.append({"node_id": node["id"], "fitness": fitness,
                                           "evaluation_ids": ids, "failure": failure, "error": error})
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
                "selected_node": node["id"], "selected_key": node["key"]})
            self.phase = "finished"
        self._save()

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
                   "num_roots": sum(n["parent_id"] is None for n in self.archive.values()),
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
        attempts = list(self.facts.tables["attempt"].values())
        counts = dict(Counter(a["status"] for a in attempts))
        actions = {}
        for name in ("Refine", "Explore", "Crossover"):
            proposed = [a for a in attempts if a["action"] == name]
            new = [a for a in proposed if a["status"] == "valid" and a["parent_id"] is not None]
            improved = same = worse = 0
            for attempt in new:
                parent = self.archive[attempt["parent_id"]]
                delta = attempt["fitness"] - parent["fitness"]
                tolerance = 1e-9 * max(1.0, abs(parent["score"]))
                if delta > tolerance:
                    improved += 1
                elif delta < -tolerance:
                    worse += 1
                else:
                    same += 1
            actions[name] = {"attempts": len(proposed), "valid": len(new),
                             "improved": improved, "worse": worse, "same_score": same,
                             "improvement_per_attempt": improved / len(proposed) if proposed else None,
                             "improvement_per_valid": improved / len(new) if new else None}
        repaired = [a for a in attempts if a["repair_of"] is not None]
        best = max(self.archive.values(), key=lambda n: (n["fitness"], -n["id"])) if self.archive else None
        frontier = -math.inf
        explore_frontiers = 0
        for node in sorted(self.archive.values(), key=lambda n: n["id"]):
            if node["action"] == "Explore" and node["fitness"] > frontier:
                explore_frontiers += 1
            frontier = max(frontier, node["fitness"])
        ordered = sorted(self.archive.values(), key=lambda n: n["id"])
        quartiles = [ordered[i * len(ordered) // 4:(i + 1) * len(ordered) // 4]
                     for i in range(4)]
        return {"status_counts": counts, "actions": actions,
                "repair_success": sum(a["status"] == "valid" for a in repaired),
                "repair_attempts": len(repaired),
                "crossover_copy_rate": counts.get("copied_reference", 0) / actions["Crossover"]["attempts"]
                if actions["Crossover"]["attempts"] else None,
                "explore_new_frontiers": explore_frontiers,
                "best_training_depth": best["depth"] if best else None,
                "mean_code_chars_by_node_quartile": [statistics.fmean(len(n["code"]) for n in group)
                                                     if group else None for group in quartiles],
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
