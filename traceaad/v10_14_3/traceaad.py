"""V10.14-3: all-source development with complete trajectory information."""

from dataclasses import asdict
from datetime import datetime
import json
import math
from pathlib import Path
import random
import statistics
import time
import traceback

from core import SecureEvaluator
from traceaad.v10_13.storage import write_json
from .config import Config
from .edits import changed_symbols, close_trace, numeric_parameters, observed_change, parse_response, source_id, symbols, validate_source
from .evaluation import SeededEvaluation, fingerprint, protocol_identity, training_probes
from .frontier import Frontier
from .prompts import ContextError, PromptBuilder, source_diff
from .state import Facts, Ledger
from .selection import select_parent, unique_archive, opportunity_key, syntax_id


def now():
    return datetime.now().isoformat(timespec="seconds")


class TraceAADV10143:
    METHOD = "v1014_3"

    def __init__(self, *, evaluation, llm, run_dir, config=None, task=None,
                 selection_evaluation=None):
        self.config = config or Config()
        if getattr(llm, "chars_per_token", None) is not None or not callable(getattr(llm, "count_tokens", None)):
            raise ValueError("V10.14-3 requires the serving tokenizer, not character estimates")
        self.parent_counts = {}
        self.llm, self.task = llm, task
        self.facts = Facts(run_dir)
        self.run_dir = self.facts.path.parent
        self.template = str(evaluation.template_program)
        self.probes = training_probes(evaluation, task)
        self.protocol, self.environment = protocol_identity(
            evaluation, self.config.evaluation_seeds, self.probes, "search")
        self.evaluator = SecureEvaluator(SeededEvaluation(evaluation))
        self.probe_evaluator = SecureEvaluator(SeededEvaluation(evaluation, self.probes, diagnostic_only=True))
        self.selection_evaluator = None
        self.selection_protocol = None
        if selection_evaluation is not None:
            if str(selection_evaluation.template_program) != self.template:
                raise ValueError("selection evaluator has a different task interface")
            self.selection_protocol, selection_environment = protocol_identity(
                selection_evaluation, self.config.evaluation_seeds, [], "selection")
            search_data = getattr(evaluation, "_datasets", None)
            selection_data = getattr(selection_evaluation, "_datasets", None)
            if selection_environment == self.environment or (search_data is not None and
                    selection_data is not None and fingerprint(search_data) == fingerprint(selection_data)):
                raise ValueError("selection evaluation must use a distinct frozen dataset/protocol")
            self.selection_evaluator = SecureEvaluator(SeededEvaluation(selection_evaluation))
        self.anchors = self.facts.tables["anchor"]
        task_text = evaluation.task_description.strip()
        notes = getattr(evaluation, "design_notes", "")
        if notes:
            task_text += "\n\nTask execution details:\n" + notes
        task_text += (f"\nEvaluation timeout per run: {evaluation.timeout_seconds}; quality is maximized. "
                      "Preserve the interface, feasibility and finite outputs; do not access evaluation internals.")
        self.prompts = PromptBuilder(llm, task_text, self.template, self.facts, self.config)
        self.rng = random.Random(self.config.seed)
        self.ledger = Ledger(self.config.budget, self.config.max_evaluations)
        self.frontier = Frontier(self.config, self.anchors)
        self.phase = "roots"
        self.init_index = 0
        self.bootstrap = []
        self.bootstrap_index = 0
        self.session = None
        self.session_count = 0
        self.repairs = {}
        self.checked_closures = {}
        self.finalists = []
        self.selection_results = []
        self.pending = None
        self.elapsed = 0.
        self.started_at = now()
        self.service_failures = 0
        self._clock = time.monotonic()
        self.identity = {"config": asdict(self.config), "task": task, "protocol": self.protocol,
                         "selection_protocol": self.selection_protocol,
                         "implementation": fingerprint({str(p): source_id(p.read_text()) for p in [*Path(__file__).parent.glob("*.py"), Path(__file__).parents[1] / "v10_13" / "storage.py", Path(__file__).parents[1] / "v10_13" / "parsing.py", Path(__file__).parents[2] / "core" / "llm.py"]}),
                         "model": {k: getattr(llm, k, None) for k in
                                   ("model", "temperature", "top_p", "enable_thinking", "chars_per_token", "stop", "extra_body")}}
        # JSON round-trip makes tuple/list representation identical at resume.
        self.identity = json.loads(json.dumps(self.identity))
        if self.facts.state:
            self._restore(self.facts.state)

    def _state(self):
        return {"identity": self.identity, "ledger": asdict(self.ledger), "parent_counts": self.parent_counts,
                "frontier": self.frontier.regions, "phase": self.phase,
                "init_index": self.init_index, "bootstrap": self.bootstrap,
                "bootstrap_index": self.bootstrap_index, "session": self.session,
                "session_count": self.session_count, "repairs": self.repairs,
                "checked_closures": self.checked_closures, "finalists": self.finalists,
                "selection_results": self.selection_results, "pending": self.pending,
                "rng": self.rng.getstate(), "elapsed": self.elapsed + time.monotonic()-self._clock,
                "started_at": self.started_at, "service_failures": self.service_failures}

    def _save(self):
        self.facts.checkpoint(self._state())

    def _restore(self, state):
        if state["identity"] != self.identity:
            raise ValueError("resume configuration, model, source or evaluation protocol changed")
        if state.get("pending"):
            raise RuntimeError("external result is uncertain; refusing to replay a charged request/evaluation")
        self.ledger = Ledger(**state["ledger"])
        self.parent_counts = state["parent_counts"]
        self.frontier = Frontier(self.config, self.anchors, state["frontier"])
        for key in ("phase", "init_index", "bootstrap", "bootstrap_index", "session", "session_count",
                    "repairs", "checked_closures", "finalists", "selection_results", "elapsed",
                    "started_at", "service_failures"):
            setattr(self, key, state[key])
        rng = state["rng"]
        self.rng.setstate((rng[0], tuple(rng[1]), rng[2]))

    @property
    def caps(self):
        return {"trial": math.floor(self.ledger.search_limit * self.config.trial_fraction),
                "recheck": math.floor(self.ledger.search_limit * self.config.recheck_fraction)}

    def _time_available(self):
        return (self.config.max_seconds is None or
                self.elapsed + time.monotonic()-self._clock < self.config.max_seconds)

    def _can_start(self, channel, length):
        if not self._time_available():
            return False
        if self.config.max_total_tokens is not None:
            calls = length - 1 if channel == "recheck" else length
            worst = calls * (self.config.max_input_tokens + self.config.output_tokens)
            if self.ledger.tokens + worst > self.config.max_total_tokens:
                return False
        return self.ledger.can_reserve(channel, length, len(self.config.evaluation_seeds), self.caps)

    def _begin(self, channel, length, anchor=None, region=None, closure=None):
        self.ledger.reserve(channel, length, len(self.config.evaluation_seeds), self.caps)
        self.session_count += 1
        ref = anchor["fitness"] if anchor else None
        if region is not None:
            ref = max(ref, self.anchors[self.frontier.regions[region]["champion"]]["fitness"])
        cursor = anchor
        while cursor:
            ref = max(ref, cursor["fitness"])
            cursor = self.anchors.get(cursor["parent_id"])
        self.session = {"id": self.session_count, "channel": channel, "stage": self.phase, "length": length, "step": 0,
            "origin": anchor["id"] if anchor else None, "working": anchor["id"] if anchor else None,
            "origin_region": region, "reference_quality": ref, "best_new": None,
            "hypothesis": "", "repair": None, "attempt_ids": [], "closure": closure,
            "comparison_id": None, "status": "active"}
        if channel == "trial" and anchor:
            r = self.frontier.regions[region]
            if r["challenger"] == anchor["id"]:
                r["challenger"] = None
                self.session["hypothesis"] = self._hypothesis(anchor)
            else:
                self.session["discover"] = True
            if anchor["artifact_id"] not in r["tried"]:
                r["tried"].append(anchor["artifact_id"])
        if closure:
            revision = self.facts.tables["revision"][closure["revision_id"]]
            self.session["hypothesis"] = (
                f"Revalidate retaining historical revision {revision['id']} in current anchor {anchor['id']}; "
                "its original effect alone does not establish its current value.")
        self._save()

    def _hypothesis(self, anchor):
        revision = self.facts.tables["revision"].get(anchor.get("revision_id"))
        if not revision:
            return "Develop one concrete change to the current decision mechanism and test its consequences."
        return (f"Continue testing actual revision {revision['id']} touching {', '.join(revision['symbols'])}. "
                f"Candidate's optional rationale (unverified): {anchor.get('idea_metadata', {}).get('display', anchor.get('idea', ''))}\n"
                "Correct dependencies or calibrate this change before proposing an unrelated redesign.")

    def _finish_session(self, reason="completed"):
        session = self.session
        session["status"] = reason
        session["unused_reservation"] = self.ledger.release()
        session["net_gain"] = max(0., session["best_new"]-session["reference_quality"]) if (
            session["best_new"] is not None and session["reference_quality"] is not None) else 0.
        session["global_gain"] = sum(self.facts.tables["attempt"][i].get("global_gain", 0.)
                                     for i in session["attempt_ids"])
        if session["channel"] == "recheck" and len(session["attempt_ids"]) > 1:
            followup = self.facts.tables["attempt"][session["attempt_ids"][-1]]
            parent = self.anchors[followup["parent_id"]]
            session["followup_gain"] = (max(0., followup["fitness"]-parent["fitness"])
                if followup.get("is_new") and followup.get("fitness") is not None else 0.)
        session["finished_at"] = now()
        if session["id"] not in self.facts.tables["session"]:
            self.facts.add("session", dict(session))
        if session["channel"] == "init":
            if session["stage"] == "roots":
                self.init_index += 1
            elif session["stage"] == "bootstrap":
                self.bootstrap_index += 1
        self.session = None
        self._save()

    def _donor(self, anchor):
        candidates = [a for a in unique_archive(self.anchors)
                      if opportunity_key(a) != opportunity_key(anchor)]
        if self.config.behavior_eligibility_gate:
            candidates = [a for a in candidates if not a["profile"] or a["profile"] != anchor["profile"]]
        # Uniform reference opportunity over all distinct valid sources; no
        # same-probe veto. The parent still follows the frozen quality/count rule.
        return self.rng.choice(candidates) if candidates else None

    def _contract(self, anchor):
        donor = self._donor(anchor)
        actions = ["Refine", "Pivot"]
        if numeric_parameters(self.facts.code(anchor)):
            actions.append("Tune")
        if donor is not None:
            actions.append("Transfer")
        action = self.rng.choice(actions)
        return ("Refine", "Transfer", donor) if action == "Transfer" else (action, "None", None)

    def _generate(self, request):
        rid = self.ledger.calls + 1
        self.ledger.calls += 1
        self.pending = {"kind": "generation", "request_id": rid, "candidate_id": self.ledger.candidates}
        self._save()
        self.facts.add("request", {"id": rid, "candidate_id": self.ledger.candidates,
            "session_id": self.session["id"], **request, "model_config": self.identity["model"],
            "output_tokens": self.config.output_tokens, "created_at": now()})
        started = time.monotonic()
        try:
            details = self.llm.draw_sample_with_details(request["prompt"], max_tokens=self.config.output_tokens)
            content = details.get("content", "")
            error = None
            self.service_failures = 0
        except Exception as exc:
            details, content, error = {}, "", f"{type(exc).__name__}: {exc}"
            self.service_failures += 1
        usage = details.get("usage") or {}
        actual_tokens = usage.get("total_tokens")
        if not isinstance(actual_tokens, int) or actual_tokens < 0:
            actual_tokens = request["input_tokens"] + self.config.output_tokens
            accounting = "conservative_bound_missing_usage"
        else:
            accounting = "reported_usage"
        self.ledger.tokens += actual_tokens
        self.facts.record_call({"request_id": rid, "candidate_id": self.ledger.candidates,
            "prompt": request["prompt"], "response": content, "finish_reason": details.get("finish_reason"),
            "usage": usage, "accounted_tokens": actual_tokens, "token_accounting": accounting,
            "model": details.get("model"), "error": error, "seconds": time.monotonic()-started})
        return rid, content, details.get("finish_reason"), error

    def _evaluate(self, artifact_id, *, role="search"):
        evaluator = self.evaluator if role == "search" else self.selection_evaluator
        protocol = self.protocol if role == "search" else self.selection_protocol
        # Only complete successful panels are reusable. Invalid attempts stay facts.
        cached = [a for a in self.anchors.values() if a["artifact_id"] == artifact_id and a["protocol"] == protocol]
        if cached and role == "search":
            a = cached[0]
            return {key: a[key] for key in ("fitness", "scores", "profile", "scenes", "evaluation_ids")}, None, True
        scores, ids, profile, scenes = [], [], [], []
        code = self.facts.tables["artifact"][artifact_id]["code"]
        for position, seed in enumerate(self.config.evaluation_seeds):
            if role == "search" and not self._time_available():
                return None, "wall-clock budget reached before complete panel", False
            if role == "search":
                self.ledger.consume_evaluation()
            eid = len(self.facts.tables["evaluation"]) + 1
            self.pending = {"kind": "evaluation", "evaluation_id": eid, "artifact_id": artifact_id, "role": role}
            self._save()
            started = time.monotonic()
            result = evaluator.evaluate_program_with_details(self.template, source=code, seed=seed,
                include_probes=False)
            value = result.result
            valid = isinstance(value, dict) and isinstance(value.get("score"), (int, float)) and math.isfinite(value["score"])
            error = result.error if not valid else None
            if valid and position == 0 and role == "search" and self.probes:
                # Diagnosis has its own timeout and cannot invalidate a valid score.
                probe_started = time.monotonic()
                probe_result = self.probe_evaluator.evaluate_program_with_details(
                    self.template, source=code, seed=seed, include_probes=True)
                if isinstance(probe_result.result, dict):
                    value.update({k: v for k, v in probe_result.result.items() if k != "score"})
                else:
                    value.update(profile=[], scenes=[], probe_error=probe_result.error or probe_result.failure_kind,
                                 probe_calls=None, probe_seconds=time.monotonic()-probe_started)
            self.facts.add("evaluation", {"id": eid, "artifact_id": artifact_id, "protocol": protocol,
                "role": role, "seed": seed, "result": value if valid else None,
                "valid": valid, "failure_kind": result.failure_kind, "error": error,
                "error_type": result.error_type, "traceback": result.traceback,
                "seconds": time.monotonic()-started, "created_at": now()})
            ids.append(eid)
            if not valid:
                return None, error or result.failure_kind or "invalid evaluation", False
            scores.append(value["score"])
            if position == 0:
                profile, scenes = value["profile"], value["scenes"]
        return {"fitness": statistics.fmean(scores), "scores": scores, "profile": profile,
                "scenes": scenes, "evaluation_ids": ids}, None, False

    def _attempt(self, *, anchor=None, request=None, source=None, repair_of=None):
        self.ledger.consume_candidate()
        if anchor:
            key = opportunity_key(anchor)
            self.parent_counts[key] = self.parent_counts.get(key, 0) + 1
        cid = self.ledger.candidates
        session = self.session
        region = session["origin_region"]
        record = {"id": cid, "session_id": session["id"], "channel": session["channel"],
                  "parent_id": anchor["id"] if anchor else None, "repair_of": repair_of,
                  "protocol": self.protocol, "status": "pending", "created_at": now(),
                  "scope": request["scope"] if request else "TRACE_RECHECK",
                  "reference_mode": request["reference_mode"] if request else "None",
                  "donor_id": request["donor_id"] if request else None}
        record["global_best_before"] = max((a["fitness"] for a in self.anchors.values()), default=None)
        record["global_gain"] = 0.
        self.pending = {"kind": "candidate", "candidate_id": cid}
        self._save()
        code, idea, error = source, "", None
        delivery = {}
        record["selection"] = session.get("selection")
        record["selection_applied_to_parent"] = anchor is not None and bool(session.get("selection"))
        if request:
            record["sampled_action"] = request.get("sampled_action", "Init" if anchor is None else "Refine")
            record["executed_action"] = "Repair" if repair_of else ("Transfer" if request["reference_mode"] == "Transfer" else request["scope"])
            record["context_mode"] = request["context_mode"]
        if request:
            rid, text, finish, error = self._generate(request)
            record["request_id"] = rid
            if error:
                record["status"] = "service_error"
            else:
                try:
                    edit_base = (self.facts.tables["attempt"][repair_of]["code"] if repair_of
                                 else self.facts.code(anchor) if anchor else None)
                    code, idea = parse_response(text, finish, self.template,
                        base=edit_base, mode=request["output_mode"], metadata=delivery,
                        count_tokens=self.llm.count_tokens, idea_limit=self.config.idea_tokens)
                except (SyntaxError, ValueError, TypeError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    code = getattr(exc, "code", None)
                    record["status"] = "delivery_failed"
        else:
            try:
                validate_source(code, self.template)
            except (SyntaxError, ValueError, TypeError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                record["status"] = "delivery_failed"
        idea = delivery.pop("idea_raw", idea)
        new_anchor = None
        if not error:
            aid = self.facts.artifact(code, self.environment)
            record["artifact_id"] = aid
            is_new = not any(a["artifact_id"] == aid for a in self.anchors.values())
            outcome, error, cached = self._evaluate(aid)
            record["cached"] = cached
            if outcome:
                new_anchor = {"id": cid, "artifact_id": aid, "parent_id": anchor["id"] if anchor else None,
                    "operator": record["scope"], "reference_id": record["donor_id"], "idea": idea,
                    "syntax_id": syntax_id(code),
                    "idea_metadata": delivery.get("idea", {}), "delivery": delivery,
                    "attempt_id": cid, "protocol": self.protocol, "origin_region": region,
                    "revision_id": cid if anchor and aid != anchor["artifact_id"] else None,
                    "is_new": is_new, **outcome}
                if new_anchor["revision_id"]:
                    diff = source_diff(self.facts.code(anchor), code)
                    self.facts.add("revision", {"id": cid, "parent": anchor["id"], "child": cid,
                        "old_artifact": anchor["artifact_id"], "new_artifact": aid,
                        "old_score": anchor["fitness"], "new_score": outcome["fitness"],
                        "idea": idea, "idea_metadata": delivery.get("idea", {}),
                        "protocol": self.protocol, "diff": diff, "symbols": changed_symbols(self.facts.code(anchor), code),
                        "interpretation": "observed original-background difference, not a universal contribution"})
                    record["diff"] = diff
                self.facts.add("anchor", new_anchor)
                record.update(status="ok" if is_new else "duplicate", fitness=outcome["fitness"],
                              anchor_id=cid, evaluation_ids=outcome["evaluation_ids"], is_new=is_new)
                if is_new and record["global_best_before"] is not None:
                    record["global_gain"] = max(0., outcome["fitness"]-record["global_best_before"])
                if is_new:
                    session["best_new"] = max(outcome["fitness"], session["best_new"] if session["best_new"] is not None else -math.inf)
            else:
                record["status"] = "evaluation_failed"
        record["error"] = error
        if code is not None:
            record["code"] = code
        record["idea"] = idea
        record["delivery"] = delivery
        record["idea_metadata"] = delivery.get("idea", {})
        if anchor and code and "diff" not in record:
            record["diff"] = source_diff(self.facts.code(anchor), code)
        if anchor and code:
            change = observed_change(self.facts.code(anchor), code)
            record["observed_syntax_change"] = change
            record["contract_deviation"] = (
                "Tune produced a structural syntax change" if record["scope"] == "Tune" and change == "structural_syntax_change" else
                "Pivot produced no structural syntax change" if record["scope"] == "Pivot" and change in
                {"identical_source", "presentation_only", "numeric_literals_only"} else None)
        if region is not None:
            r = self.frontier.regions[region]
            if session["channel"] == "main" and record["status"] != "service_error":
                r["stagnation"] += 1
            if session["channel"] == "trial":
                r["trial_used"] += 1
                if new_anchor and new_anchor["artifact_id"] not in r["tried"]:
                    r["tried"].append(new_anchor["artifact_id"])
        if new_anchor:
            self.frontier.admit(new_anchor)
            if session["channel"] == "main" and region is not None:
                source_region = self.frontier.regions[region]
                if new_anchor["fitness"] >= source_region["checkpoint"] + self.config.delta:
                    source_region["checkpoint"], source_region["stagnation"] = new_anchor["fitness"], 0
            if new_anchor["is_new"]:
                session["working"] = new_anchor["id"]
                if session["channel"] == "trial" and not session["hypothesis"]:
                    session["hypothesis"] = self._hypothesis(new_anchor)
        # Delivery failures without a complete source are facts, not editable programs.
        if error and code and anchor and not repair_of and record["status"] != "service_error" and request:
            if session["channel"] == "trial":
                session["repair"] = cid
                if not session["hypothesis"]:
                    session["hypothesis"] = (f"Test and repair the concrete change in failed attempt {cid}; "
                        f"affected symbols: {', '.join(symbols(code))}. "
                        "Its validity and quality are unresolved; a repair must address its recorded failure.")
            elif session["channel"] == "main":
                self.repairs[str(anchor["id"])] = cid
        session["attempt_ids"].append(cid)
        session["step"] += 1
        self.facts.add("attempt", record)
        self.pending = None
        # Compatibility projection for monitors; immutable facts remain separate.
        self.facts._append({"kind": "candidate", "candidate_id": cid, "budget_used": cid,
            "evaluation_id": self.ledger.evaluations, "status": record["status"],
            "operator": record["scope"], "node_id": new_anchor["id"] if new_anchor else None,
            "fitness": new_anchor["fitness"] if new_anchor else None,
            "node": self._export(new_anchor) if new_anchor else None, "state": self._state()})
        self._save()
        self._summary("running")
        return new_anchor, record

    def _export(self, anchor):
        return {**anchor, "code": self.facts.code(anchor),
                "evaluation_id": anchor["evaluation_ids"][-1]}

    def _closure_options(self):
        options = []
        active = [(i, r, anchor_id) for i, r in enumerate(self.frontier.regions)
                  for anchor_id in (r["champion"], r["challenger"]) if anchor_id is not None]
        for region_id, region, anchor_id in active:
            current = self.anchors.get(anchor_id)
            if current is None or current["parent_id"] is None:
                continue
            middle = self.anchors[current["parent_id"]]
            if middle["parent_id"] is None or not middle.get("revision_id"):
                continue
            old = self.anchors[middle["parent_id"]]
            if any(a["protocol"] != self.protocol for a in (old, middle, current)):
                continue
            revision_id = middle["revision_id"]
            # Select only evidence that fits the current decision packet. This
            # preview is NOT recorded as a model request; the trigger is logged.
            try:
                packet = self.prompts.build(current)
            except ContextError:
                continue
            if f"revision:{revision_id}" not in packet["evidence_ids"]:
                continue
            key = fingerprint([old["artifact_id"], middle["artifact_id"], current["artifact_id"], self.protocol])
            if key in self.checked_closures:
                continue
            started = time.monotonic()
            try:
                source, check = close_trace(*(self.facts.code(a) for a in (old, middle, current)))
            except ValueError as exc:
                self.checked_closures[key] = "not_reconstructable"
                self.facts._append({"kind": "closure_check", "key": key,
                    "anchors": [old["id"], middle["id"], current["id"]],
                    "status": "not_reconstructable", "error": str(exc),
                    "seconds": time.monotonic()-started})
                continue
            closure = {"key": key, "anchors": [old["id"], middle["id"], current["id"]],
                       "revision_id": revision_id, "source": source, "check": check,
                       "trigger_evidence_ids": packet["evidence_ids"],
                       "reconstruction_seconds": time.monotonic()-started}
            # Compare only if its provenance can also fit the subsequent request.
            options.append((region_id, closure))
        options.sort(key=lambda item: (
            -int(self.frontier.regions[item[0]]["stagnation"] > 0),
            -self.anchors[item[1]["anchors"][2]]["fitness"],
            -abs(self.anchors[item[1]["anchors"][1]]["fitness"] - self.anchors[item[1]["anchors"][0]]["fitness"]),
            -item[1]["anchors"][2]))
        return options

    def _schedule_search(self):
        choices, weights = [], []
        c = self.config
        if self._can_start("main", 1):
            choices.append("main")
            weights.append(1-c.trial_fraction-c.recheck_fraction)
        if c.trial_fraction and self._can_start("trial", c.trial_length):
            choices.append("trial")
            weights.append(c.trial_fraction/c.trial_length)
        closures = []
        if c.recheck_fraction and self._can_start("recheck", 2):
            closures = self._closure_options()
            if closures:
                choices.append("recheck")
                weights.append(c.recheck_fraction/2)
        if not choices:
            return False
        channel = self.rng.choices(choices, weights=weights)[0]
        if channel == "recheck":
            region, closure = closures[0]
            self._begin(channel, 2, self.anchors[closure["anchors"][2]], region, closure)
        elif channel == "trial":
            region = self.frontier.trial(self.rng)
            r = self.frontier.regions[region]
            selected = r["challenger"] if r["challenger"] is not None else r["champion"]
            self._begin(channel, c.trial_length, self.anchors[selected], region)
        else:
            pool = self.anchors
            if c.behavior_eligibility_gate:
                ids = {r["champion"] for r in self.frontier.regions if r["champion"] is not None}
                pool = {i: self.anchors[i] for i in ids}
            anchor, selection = select_parent(pool, self.parent_counts, c.exploration_constant, c.parent_policy)
            self._begin(channel, 1, anchor, self.frontier.region_for(anchor))
            self.session["selection"] = selection
            self._save()
        return True

    def _comparison(self, closure, counterfactual):
        existing = next((r for r in self.facts.tables["comparison"].values()
                         if r.get("closure_key") == closure["key"]), None)
        if existing:
            return existing
        old, middle, current = [self.anchors[i] for i in closure["anchors"]]
        cid = len(self.facts.tables["comparison"]) + 1
        record = {"id": cid, "closure_key": closure["key"], "protocol": self.protocol, "historical_revision": closure["revision_id"],
            "anchors": [old["id"], middle["id"], current["id"], counterfactual["id"] if counterfactual else None],
            "source_hashes": closure["check"]["snapshots"], "edits": closure["check"],
            "construction": "actual parent is current p11; inverse A; B(p00) is an identity check only",
            "status": "invalid" if counterfactual is None else "measured",
            "scope": "specific code bundles, fixed data, numeric scale and seed panel; not a universal causal effect"}
        if counterfactual:
            panels = [a["scores"] for a in (old, middle, current, counterfactual)]
            before = [b-a for a, b in zip(panels[0], panels[1])]
            after = [a-b for a, b in zip(panels[2], panels[3])]
            interaction = [b-a for a, b in zip(before, after)]
            tol = self.config.comparison_tolerance
            verdict = ("positive_on_fixed_panel" if all(x > tol for x in after) else
                       "negative_on_fixed_panel" if all(x < -tol for x in after) else "inconclusive")
            record.update(scores=panels, evaluation_ids=[a["evaluation_ids"] for a in (old, middle, current, counterfactual)],
                delta_old=statistics.fmean(before), delta_current=statistics.fmean(after),
                interaction=statistics.fmean(interaction), paired_current=after, verdict=verdict,
                seeds=list(self.config.evaluation_seeds),
                uncertainty="fixed-panel observation; a single seed is not statistical evidence of generality")
        self.facts.add("comparison", record)
        return record

    def _step_session(self):
        s = self.session
        if s["id"] in self.facts.tables["session"]:
            self._finish_session(self.facts.tables["session"][s["id"]]["status"])
            return
        if not self._time_available():
            self._finish_session("time_limit")
            return
        if s["channel"] == "recheck" and s["comparison_id"] is None:
            closure = s["closure"]
            current = self.anchors[s["origin"]]
            if s["step"] == 0:
                counterfactual, _ = self._attempt(anchor=current, source=closure["source"])
            else:
                counterfactual = self.anchors.get(s["attempt_ids"][0])
            comparison = self._comparison(closure, counterfactual)
            self.checked_closures[closure["key"]] = comparison["id"]
            s["comparison_id"] = comparison["id"]
            s["direct_recheck_gain"] = max(0., counterfactual["fitness"]-current["fitness"]) if counterfactual else None
            if counterfactual:
                self.frontier.reactivate(current, comparison["id"])
            # Both feedback/no-feedback branches start from exactly the same
            # deterministic better-of-current-and-counterfactual rule.
            s["working"] = (counterfactual["id"] if counterfactual and counterfactual["fitness"] > current["fitness"]
                            else current["id"])
            self._save()
            if counterfactual is None:
                self._finish_session("invalid_comparison")
            return
        if s["channel"] == "recheck" and self.facts.tables["comparison"][s["comparison_id"]]["status"] == "invalid":
            self._finish_session("invalid_comparison")
            return
        anchor = self.anchors.get(s["working"])
        failed_id = s.pop("repair", None) or (self.repairs.pop(str(anchor["id"]), None)
                                             if anchor and s["channel"] == "main" else None)
        failed = self.facts.tables["attempt"].get(failed_id)
        if anchor:
            scope, reference_mode, donor = (("Refine", "None", None) if s["stage"] == "bootstrap"
                                            else self._contract(anchor))
            sampled_action = "Transfer" if reference_mode == "Transfer" else scope
            if failed:
                scope, reference_mode, donor = "Refine", "None", None
        else:
            scope, reference_mode, donor = "Init", "None", None
            sampled_action = "Init"
        comparison = self.facts.tables["comparison"].get(s["comparison_id"])
        independent = (scope == "Pivot" and not failed and s["channel"] == "main"
                       and self.config.pivot_context == "independent")
        if independent:
            # Selection is recorded as scheduling context only: no parent code,
            # rationale, score, probes, history or donor reaches this request.
            s["scheduled_parent_id"] = anchor["id"]
            s["origin"] = s["working"] = s["origin_region"] = None
            s["reference_quality"] = None
            s["hypothesis"] = ""
            anchor, donor, comparison = None, None, None
            reference_mode = "None"
        roots = []
        if not anchor and not independent:
            n_independent = {"independent": self.config.root_attempts, "sequential": 1,
                             "hybrid": (self.config.root_attempts+1)//2}[self.config.init_mode]
            if self.init_index >= n_independent:
                unique = {}
                for a in self.anchors.values():
                    if a["parent_id"] is None:
                        unique.setdefault(a["artifact_id"], a)
                roots = sorted(unique.values(), key=lambda a: -a["fitness"])
        try:
            request = self.prompts.build(anchor, scope=scope, reference_mode=reference_mode,
                donor=donor, hypothesis=s["hypothesis"], failed=failed, comparison=comparison, roots=roots)
        except ContextError as exc:
            self.facts._append({"kind": "context_failure", "session_id": s["id"], "error": str(exc)})
            # A non-fitting required request is not silently sampled again.
            self._finish_session("context_limit")
            self.phase = "freeze"
            self._save()
            return
        request["sampled_action"] = sampled_action
        request["executed_action"] = "Repair" if failed else ("Transfer" if request["reference_mode"] == "Transfer" else request["scope"])
        child, _ = self._attempt(anchor=anchor, request=request, repair_of=failed_id)
        if s["channel"] == "recheck":
            s["followup_gain"] = max(0., child["fitness"]-anchor["fitness"]) if child and child["is_new"] else 0.
            self._save()

    def _initialize(self):
        if self.phase == "roots":
            if self.init_index < self.config.root_attempts and self._can_start("init", 1):
                self._begin("init", 1)
                self._step_session()
                if self.session:
                    self._finish_session()
                self._save()
                return
            unique = {}
            for a in self.anchors.values():
                if a["parent_id"] is None:
                    unique.setdefault(opportunity_key(a), a["id"])
            self.bootstrap = list(unique.values())
            self.phase = "bootstrap"
            self._save()
        if self.phase == "bootstrap":
            if self.bootstrap_index < len(self.bootstrap) and self._can_start("init", 1):
                self._begin("init", 1, self.anchors[self.bootstrap[self.bootstrap_index]])
                # Bootstrap is exactly one Refine, not a random main action.
                anchor = self.anchors[self.bootstrap[self.bootstrap_index]]
                try:
                    request = self.prompts.build(anchor, scope="Refine")
                    self._attempt(anchor=anchor, request=request)
                except ContextError as exc:
                    self.facts._append({"kind": "context_failure", "error": str(exc), "session_id": self.session["id"]})
                self._finish_session()
                self._save()
                return
            if not self.anchors:
                self.phase = "no_valid_root"
            else:
                self.frontier.freeze()
                self.ledger.search_limit = self.config.budget-self.ledger.candidates
                self.phase = "search"
            self._save()

    def _freeze_finalists(self):
        distinct = {}
        for a in sorted(self.anchors.values(), key=lambda a: (-a["fitness"], a["id"])):
            distinct.setdefault(opportunity_key(a), a["id"])
        self.finalists = list(distinct.values())[:self.config.final_candidates]
        self.phase = "selection" if self.selection_evaluator else "search_complete"
        write_json(self.run_dir / "finalists.json", {"protocol": self.protocol, "frozen_at": now(),
                   "selection_protocol": self.selection_protocol,
                   "candidates": [self._export(self.anchors[i]) for i in self.finalists]})
        self._save()

    def _select(self):
        for aid in self.finalists[len(self.selection_results):]:
            outcome, error, _ = self._evaluate(self.anchors[aid]["artifact_id"], role="selection")
            self.selection_results.append({"anchor_id": aid, "outcome": outcome, "error": error})
            self.pending = None
            self._save()
        valid = [r for r in self.selection_results if r["outcome"]]
        if not valid:
            self.phase = "selection_failed"
        else:
            winner = max(valid, key=lambda r: (r["outcome"]["fitness"], -self.finalists.index(r["anchor_id"])))
            anchor = self.anchors[winner["anchor_id"]]
            (self.run_dir / "best_program.py").write_text(self.facts.code(anchor), encoding="utf-8")
            write_json(self.run_dir / "selection.json", {"selection_protocol": self.selection_protocol,
                "results": self.selection_results, "selected_anchor": anchor["id"],
                "selected_source_sha256": source_id(self.facts.code(anchor))})
            self.phase = "finished"
        self._save()

    def _summary(self, status, error=None):
        best = max(self.anchors.values(), key=lambda a: (a["fitness"], -a["id"])) if self.anchors else None
        selected = None
        if self.phase == "finished":
            valid = [r for r in self.selection_results if r["outcome"]]
            selected = max(valid, key=lambda r: (r["outcome"]["fitness"], -self.finalists.index(r["anchor_id"])))
            best = self.anchors[selected["anchor_id"]]
        exported = self._export(best) if best else None
        if selected:
            exported["selection_fitness"] = selected["outcome"]["fitness"]
        self.facts.save_summary({"status": status, "phase": self.phase, "method": self.METHOD,
            "budget_basis": "attempted candidates including initialization, invalid outputs and exact reconstructions",
            "budget": self.config.budget, "budget_used": self.ledger.candidates,
            "ledger": asdict(self.ledger), "num_nodes": len(self.anchors),
            "num_roots": sum(a["parent_id"] is None for a in self.anchors.values()),
            "started_at": self.started_at, "finished_at": now(), "best": exported,
            "selection_evaluations": sum(e["role"] == "selection" for e in self.facts.tables["evaluation"].values()),
            "finalists": self.finalists, "descriptor": "diagnostic_training_probes" if self.probes else "unpartitioned_quality",
            "unique_sources": len(unique_archive(self.anchors)), "parent_counts": self.parent_counts,
            "error": error})

    def run(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if not self.facts.state:
            self._save()
        try:
            while self.phase in {"roots", "bootstrap", "search", "freeze", "selection"}:
                if self.service_failures >= 3:
                    raise RuntimeError("three service failures; stop without interpreting them as algorithm failures")
                if self.session:
                    if self.session["step"] < self.session["length"]:
                        self._step_session()
                    if self.session and self.session["step"] >= self.session["length"]:
                        self._finish_session()
                    continue
                if self.phase in {"roots", "bootstrap"}:
                    self._initialize()
                elif self.phase == "search":
                    if not self._schedule_search():
                        self.phase = "freeze"
                        self._save()
                elif self.phase == "freeze":
                    self._freeze_finalists()
                elif self.phase == "selection":
                    self._select()
            self._summary(self.phase)
            return self.facts.load_summary()
        except BaseException as exc:
            self._summary("interrupted" if isinstance(exc, KeyboardInterrupt) else "error", traceback.format_exc())
            raise
