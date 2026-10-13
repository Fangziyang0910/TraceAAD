"""Execution, budget, evaluation and selection for TraceAAD."""

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import random
import time

from .storage import write_json
from core.llm import generate, ModelCallError
from core.scheduling import SchedulerError
from .canonical import canonical, key
from .config import REVISION
from .delivery import DeliveryError, SourceError, extract_idea, parse_response
from .evaluation import ProgramEvaluator, fingerprint
from .history import final_attempt
from .prompts import ContextTooLong
from .selection import better, choose_reference, sample_parent
from .state import Facts, Progress


def now():
    return datetime.now().isoformat(timespec="seconds")


def _tuple_tree(value):
    return tuple(_tuple_tree(v) for v in value) if isinstance(value, list) else value


class Search:
    Evaluator = ProgramEvaluator
    RECORD_EXPLORATIONS = False
    # Steps whose proposals open an exploration when explorations are recorded.
    EXPLORING = ("Explore",)
    MEASURE_CALLS = True
    REPLACE_FAILED_FINALISTS = True
    AUTO_REPAIR = True

    def __init__(self, *, evaluation, llm, run_dir, config=None, task=None,
                 selection_evaluation=None):
        self.config, self.llm, self.task = config or self.Config(), llm, task
        if (getattr(llm, "chars_per_token", None) is not None or
                not callable(getattr(llm, "count_prompt_tokens", None)) or
                not callable(getattr(llm, "count_tokens", None))):
            raise ValueError("search requires serving token counts")
        self.run_dir, self.template = Path(run_dir), str(evaluation.template_program)
        self.facts = Facts(run_dir)
        self.programs, self.attempts_table = self.facts.programs, self.facts.attempts
        self.prompts = self.PromptBuilder(llm, task, evaluation, self.programs, self.attempts_table, self.config)
        self.training = self.Evaluator(evaluation, self.config.evaluation_seeds, "search", measure_calls=self.MEASURE_CALLS,
                                       **self._evaluation_options())
        self.selection = (self.Evaluator(selection_evaluation, self.config.evaluation_seeds, "selection", measure_calls=self.MEASURE_CALLS,
                                        **self._evaluation_options())
                          if selection_evaluation is not None else None)
        if self.selection:
            if str(selection_evaluation.template_program) != self.template:
                raise ValueError("selection has a different task interface")
            search_data = getattr(evaluation, "_datasets", None)
            selection_data = getattr(selection_evaluation, "_datasets", None)
            if (self.training.environment == self.selection.environment or
                    search_data is not None and selection_data is not None
                    and fingerprint(search_data) == fingerprint(selection_data)):
                raise ValueError("selection must use distinct fixed data")
        self.identity = json.loads(json.dumps({"revision": REVISION, "config": asdict(self.config),
            "task": task, "protocol": self.training.protocol,
            "selection_protocol": self.selection.protocol if self.selection else None}))
        # A resumed run continues under the current configuration and evaluator.
        saved = self.facts.state or {}
        if (saved and getattr(self.training, 'function_budget_scope', None) == 'mean_instance'
                and saved.get('identity', {}).get('revision') != REVISION):
            raise ValueError('mean CPU budget requires a new batch; saved evaluation revision differs')
        if (getattr(evaluation, 'interface', None) and saved
                and saved.get('identity', {}).get('protocol') != self.training.protocol):
            raise ValueError('complete-solver resume requires the same task interface and evaluation protocol')
        self.progress = Progress(**{name: saved[name] for name in Progress.__dataclass_fields__ if name in saved})
        self.parent_rng = random.Random(f"{self.METHOD}:{self.config.seed}:parent")
        self.action_rng = random.Random(f"{self.METHOD}:{self.config.seed}:action")
        self.reference_rng = random.Random(f"{self.METHOD}:{self.config.seed}:reference")
        for name, state in saved.get("rng", {}).items():
            getattr(self, name + "_rng").setstate(_tuple_tree(state))
        self._clock = time.monotonic()

    @property
    def phase(self):
        return self.progress.phase

    def _evaluation_options(self):
        return {}

    @property
    def archive(self):
        return self.facts.valid

    @property
    def attempts(self):
        return len(self.attempts_table)

    @property
    def evaluation_calls(self):
        return len(self.facts.evaluations)

    def _state(self):
        return {**asdict(self.progress), "identity": self.identity,
                "elapsed": self.progress.elapsed + time.monotonic() - self._clock,
                "attempts": self.attempts,
                "rng": {name: getattr(self, name + "_rng").getstate()
                        for name in ("parent", "action", "reference")}}

    def _save(self, **record):
        self.facts.commit(self._state(), **record)

    def _call(self, calls, prompt, input_tokens, stage=None, **kwargs):
        """Generate and account for a request, preserving earlier stages if it fails."""
        try:
            details, failed = generate(self.llm, prompt, **kwargs), None
            made = details["calls"]
        except ModelCallError as exc:
            details, made, failed = {}, exc.calls, exc
        for call in made:
            self.progress.model_calls += 1
            call["request_id"] = self.progress.model_calls
            if stage is not None:
                call["stage"] = stage
            usage = call["usage"]
            self.progress.input_tokens += usage.get("prompt_tokens", input_tokens) or input_tokens
            self.progress.output_tokens += usage.get("completion_tokens",
                kwargs["max_tokens"] if not call["error"] else 0) or 0
            self.progress.service_failures += bool(call["error"])
        calls.extend(made)
        if failed:
            self._save(calls=calls)
            if failed.transient:
                raise RuntimeError(f"model service unavailable after {len(calls)} calls: {failed}") from failed
            raise failed
        return details

    def _generate(self, request):
        calls = []
        details = self._call(calls, request["prompt"], request["input_tokens"],
                             max_tokens=self.config.output_tokens)
        return calls[-1]["request_id"], details, calls

    def _parse_candidate(self, request, details, parent):
        return parse_response(details.get("content", ""), details.get("finish_reason"), self.template)


    def _attempt(self, request, *, parent=None, reference=None, repair_of=None):
        rid, details, calls = self._generate(request)
        aid = self.attempts + 1
        if self.phase == "roots":
            self.progress.init_attempts += 1
        response = details.get("content", "")
        attempt = {**request, "id": aid, "request_id": rid,
                   "parent_id": parent["id"] if parent else None,
                   "reference_id": reference["id"] if reference else None,
                   "repair_of": repair_of, "status": None, "program_id": None,
                   "idea": extract_idea(response) if isinstance(response, str) else "",
                   "error": None, "calls": calls, "created_at": now()}
        code, failure, evaluations = None, None, []
        try:
            code, attempt["idea"], attempt["delivery"] = self._parse_candidate(request, details, parent)
        except DeliveryError as exc:
            attempt.update(status="delivery_failed", error=str(exc))
        except SourceError as exc:
            code = exc.code
            failure = {"kind": "invalid_source", "error": str(exc), "line": None,
                       "seconds": None, "calls": None, "function_seconds": None, "call_running": False}
        program = None
        if code is not None:
            try:
                code = canonical(code)
            except (SyntaxError, ValueError):
                code = code.rstrip() + "\n"
            source_key = key(code)
            known = self.facts.by_key.get(source_key)
            if known is not None:
                status = ("copied_reference" if reference and known["id"] == reference["id"] else
                          "duplicate" if known["valid"] else "known_failure")
                attempt.update(status=status, program_id=known["id"])
            else:
                measured = {"seconds": None, "wall_seconds": None, "calls": None, "function_seconds": None}
                fitness = None
                if failure is None:
                    try:
                        outcome = self.training.evaluate(code, source_key, **self._evaluation_options())
                    except SchedulerError:
                        self._save(calls=calls)
                        raise
                    evaluations = outcome["evaluations"]
                    fitness, failure = outcome["fitness"], outcome["failure"]
                    measured = {name: outcome[name] for name in ("seconds", "calls", "function_seconds")}
                    measured["wall_seconds"] = outcome["seconds"]
                    # Prompts describe per-instance computation, not the shorter
                    # batch wall time obtained by running instances concurrently.
                    measured["seconds"] = outcome.get("instance_seconds", outcome["seconds"])
                program = {"id": aid, "key": source_key, "code": code,
                           "fitness": fitness, "score": fitness,
                           "parent_id": attempt["parent_id"], "action": request["action"],
                           "reference_id": attempt["reference_id"], "idea": attempt["idea"],
                           "depth": parent["depth"] + 1 if parent else 0, "repaired": repair_of is not None,
                           "valid": failure is None, "failure": failure,
                           "eval_seconds": measured["seconds"], "eval_wall_seconds": measured["wall_seconds"],
                           "calls": measured["calls"],
                           "function_seconds": measured["function_seconds"], "attempt_id": aid}
                attempt.update(status=failure["kind"] if failure else "valid", program_id=aid)
        needs_repair = (program is not None and not program["valid"] and repair_of is None
                        and (self.AUTO_REPAIR or self.phase == "roots")
                        and aid < self.config.budget
                        and (self.phase != "roots" or self.progress.init_attempts < self.config.init_attempt_limit))
        self.progress.repair_id = aid if needs_repair else None
        self._save(attempt=attempt, program=program, evaluations=evaluations)
        if needs_repair:
            return self._repair()
        return program if program is not None and program["valid"] else None

    def _repair(self):
        failed = self.programs[self.progress.repair_id]
        try:
            request = self.prompts.repair(failed)
        except ContextTooLong:
            self.progress.repair_id = None
            self._save()
            return None
        return self._attempt(request, parent=failed, repair_of=failed["id"])

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
        if self.progress.repair_id is not None:
            self._repair()
            return
        roots = [node for node in self.archive.values() if self._is_root(node)]
        if (len(roots) >= self.config.roots or
                self.progress.init_attempts >= self.config.init_attempt_limit or self.attempts >= self.config.budget):
            self.progress.phase = "search" if roots else "no_valid_root"
            self._save()
            return
        try:
            request = self.prompts.initial(roots)
        except ContextTooLong:
            self.progress.phase = "search" if roots else "no_valid_root"
            self._save()
            return
        self._attempt(request)

    def _choose_parent(self, eligible):
        return sample_parent(eligible, self.attempts_table, self.programs, self.parent_rng)

    def _ordinary_search(self, sampled=None):
        eligible = [node for node in self.archive.values() if node["id"] not in self.progress.too_long]
        if not eligible:
            self.progress.phase = "freeze"
            self._save()
            return
        parent, selection = self._choose_parent(eligible)
        if sampled is None:
            sampled = self.action_rng.choices(list(self.config.operators), list(self.config.operators.values()))[0]
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
            try:
                request = self.prompts.build("Explore", parent)
            except ContextTooLong:
                self.progress.too_long.append(parent["id"])
                self._save()
                return
            reference, reference_selection = None, None
            flags.append("explore_context_fallback")
        if request["action"] == "Refine" and action == "Crossover":
            reference = None
            flags.append("crossover_context_fallback")
        if self.RECORD_EXPLORATIONS:
            request['exploration'] = ({'id': len(self.facts.explorations) + 1, 'step': 0}
                                      if request['action'] in self.EXPLORING else None)
        request.update(sampled_action=sampled, fallbacks=flags, parent_id=parent["id"],
                       reference_id=reference["id"] if reference else None, selection=selection,
                       reference_selection=reference_selection)
        self._attempt(request, parent=parent, reference=reference)

    def _open_exploration(self):
        """The exploration still in progress, read from the attempts (so a resumed run continues it).

        Returns its proposal, the new program it produced (None if it produced
        none), its development attempts and the best version reached so far.
        """
        tagged = [a for a in self.attempts_table.values() if a.get("exploration")]
        if not tagged:
            return None
        eid = max(a["exploration"]["id"] for a in tagged)
        if eid in self.facts.explorations:
            return None
        attempts = sorted((a for a in tagged if a["exploration"]["id"] == eid), key=lambda a: a["id"])
        proposal, development = attempts[0], attempts[1:]
        final = final_attempt(proposal, self.attempts_table)
        proposed = self.archive.get(final["program_id"]) if final["status"] == "valid" else None
        best = proposed
        for attempt in development:
            final = final_attempt(attempt, self.attempts_table)
            reached = self.archive.get(final["program_id"]) if final["status"] == "valid" else None
            if reached is not None and better(reached["fitness"], best["fitness"]):
                best = reached
        return {"id": eid, "proposal": proposal, "proposed": proposed, "development": development, "best": best}

    def _ranking(self):
        return [n["id"] for n in sorted(self.archive.values(), key=lambda n: (n["fitness"], n["id"]))]

    def _freeze(self):
        self.progress.finalists = self._ranking()[:self.config.final_candidates if self.selection else 1]
        if self.selection:
            self.progress.phase = "selection"
            self._save()
        elif self.progress.finalists:
            self._finish(self.archive[self.progress.finalists[0]], "training")
        else:
            self.progress.phase = "no_valid_root"
            self._save()

    def _finish(self, node, criterion):
        """Freeze one winner before any held-out evaluation."""
        self.progress.selected_id = node['id']
        (self.run_dir / 'best_program.py').write_text(node['code'], encoding='utf-8')
        write_json(self.run_dir / 'selection.json', {
            'criterion': criterion,
            'selection_protocol': self.selection.protocol if self.selection else None,
            'results': self.progress.selection_results, 'finalists': self.progress.finalists,
            'selected_node': node['id'], 'selected_key': node['key']})
        self.progress.phase = 'finished'
        self._save()

    def _select(self):
        """Evaluate finalists on the selection set; a finalist that fails there is
        replaced by the next program in the training ranking, up to as many
        replacements as there are finalists."""
        if len(self.progress.selection_results) < len(self.progress.finalists):
            node = self.archive[self.progress.finalists[len(self.progress.selection_results)]]
            outcome = self.selection.evaluate(node["code"], node["key"], **self._evaluation_options())
            failure = outcome["failure"]
            self.progress.selection_results.append({"node_id": node["id"],
                                           **{k: v for k, v in outcome.items() if k != "evaluations"}})
            if (failure and self.REPLACE_FAILED_FINALISTS
                    and len(self.progress.finalists) < 2 * self.config.final_candidates):
                remaining = [i for i in self._ranking() if i not in self.progress.finalists]
                if remaining:
                    self.progress.finalists.append(remaining[0])
            self._save(evaluations=outcome["evaluations"])
            return
        valid = [r for r in self.progress.selection_results if r["fitness"] is not None]
        if not valid:
            self.progress.phase = "selection_failed"
        else:
            selected = min(valid, key=lambda r: (r["fitness"], self.progress.finalists.index(r["node_id"])))
            self._finish(self.archive[selected['node_id']], 'validation')
            return
        self._save()

    def _summary(self, status, error=None):
        best = (self.archive.get(self.progress.selected_id) if self.phase == "finished" else
                min(self.archive.values(), key=lambda n: (n["fitness"], n["id"]), default=None))
        if best is not None:
            best = {**{k: v for k, v in best.items() if k not in {"code", "failure"}},
                    "selection_fitness": next((r["fitness"] for r in self.progress.selection_results
                    if r["node_id"] == best["id"]), None)}
            if self.phase == "finished" and self.selection is None and best["selection_fitness"] is not None:
                best.update(fitness=best["selection_fitness"], score=best["selection_fitness"])
        summary = {"status": status, "phase": self.phase, "method": self.METHOD, "revision": REVISION,
                   "final_selection": "validation" if self.selection else "training",
                   "budget": self.config.budget, "budget_used": self.attempts, "budget_axis": "候选尝试",
                   "init_attempts": self.progress.init_attempts, "num_nodes": len(self.archive),
                   "num_failed_programs": len(self.programs) - len(self.archive),
                   "num_roots": sum(self._is_root(n) for n in self.archive.values()),
                   "model_calls": self.progress.model_calls, "evaluation_calls": self.evaluation_calls,
                   "search_evaluations": sum(e["role"] == "search" for e in self.facts.evaluations),
                   "selection_evaluations": sum(e["role"] == "selection" for e in self.facts.evaluations),
                   "input_tokens": self.progress.input_tokens, "output_tokens": self.progress.output_tokens,
                   "service_failures": self.progress.service_failures, "too_long": self.progress.too_long,
                   "started_at": self.progress.started_at, "finished_at": now(),
                   "seconds": self.progress.elapsed + time.monotonic() - self._clock,
                   "best": best, "finalists": self.progress.finalists, "error": error}
        self.facts.save_summary(summary)
        return summary

    def run(self):
        if self.progress.phase in {"finished", "search_complete", "selection_failed", "no_valid_root"}:
            previous = self.facts.load_summary()
            if previous:
                return previous
        self.run_dir.mkdir(parents=True, exist_ok=True)
        try:
            while self.progress.phase in {"roots", "search", "freeze", "selection"}:
                if self.progress.phase == "roots":
                    self._roots()
                elif self.progress.phase == "search":
                    self._search()
                elif self.progress.phase == "freeze":
                    self._freeze()
                else:
                    self._select()
        except SchedulerError as exc:
            return self._summary("service_unavailable", str(exc))
        except RuntimeError as exc:
            if "model service unavailable" in str(exc):
                return self._summary("service_unavailable", str(exc))
            raise
        return self._summary(self.progress.phase)
