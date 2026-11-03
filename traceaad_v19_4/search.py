"""Sequential algorithm evolution: parent choice by reading local trajectories, short-term and long-term reflection."""

from contextlib import nullcontext
from dataclasses import asdict, replace
import hashlib
import inspect
from pathlib import Path
import random
import time

from .config import Config
from .evaluation import Evaluator
from .knowledge import ExperiencePool
from .models import Attempt, Candidate, Evaluation, FormatError, ModelReply, Program, ResponseEvent, SearchResult, parse_experience, parse_reflection, parse_selection
from .prompts import PromptBuilder
from .schemas import response_format
from .selection import candidates, choose, proposals, source_key
from .storage import Store
from .tokens import TokenCounter


SCHEMA = 1
METHOD = "traceaad_v19_4"
ROLES = {"select": "selector", "reflect": "reflector", "long_term": "long_term_reflector"}


def _tuples(value):
    return tuple(_tuples(x) for x in value) if isinstance(value, list) else value


def _source_hash(function):
    try:
        path = inspect.getsourcefile(function)
        if path and Path(path).is_file():
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except (TypeError, OSError):
        pass
    return None


class TraceAAD:
    def __init__(self, task, model, *, config=None, run_dir=None, resume=False, token_counter=None,
                 evaluator=None):
        self.task, self.model = task, model
        self.store = Store(run_dir, resume=resume)
        try:
            state = self.store.read("state.json") if resume else None
            if state and (state.get("schema_version") != SCHEMA or state.get("method") != METHOD):
                raise ValueError("V19.4 requires its own checkpoint schema and method")
            self.config = config or (Config(**state["config"]) if state else Config())
            self.count = token_counter or TokenCounter(self.config.tokenizer)
            self.prompts = PromptBuilder(task, self.config, self.count)
            self.evaluator = evaluator or Evaluator(task)
            self.rng = random.Random(self.config.seed)
            self.archive = {}
            self.attempts = []
            self.responses = []
            self.pool = ExperiencePool()
            self.evaluations_used = self.model_calls = self.consecutive_unevaluated = 0
            identity = getattr(model, "identity", None)
            self.compatibility = {
                "package": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(Path(__file__).parent.glob("*.py"))},
                "task": task.fingerprint(), "evaluator_source": _source_hash(task.evaluate_program),
                "evaluation_adapter_source": _source_hash(type(self.evaluator)),
                "model_adapter_source": _source_hash(type(model)),
                "task_sources": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                 for p in task.metadata.get("source_hashes", {})},
                "model": identity() if callable(identity) else {
                    "type": type(model).__module__ + "." + type(model).__qualname__,
                    "model": getattr(model, "model", None)},
                "tokenizer": getattr(self.count, "identity", {
                    "type": type(self.count).__module__ + "." + type(self.count).__qualname__}),
            }
            if state:
                self._restore(state)
            else:
                if task.example_code is not None:
                    self.archive[0] = Program(0, task.example_code, task.example_design, None, None, None)
                    self.store.program(self.archive[0])
                self.store.write("run_config.json", {
                    "method": METHOD, "checkpoint_schema": SCHEMA,
                    "config": asdict(self.config), "task": task.description(),
                    "compatibility": self.compatibility})
                self.store.snapshot(extra_sources=self.compatibility["task_sources"])
                self._save()
        except BaseException:
            self.store.close()
            raise

    def _save(self):
        self.store.write("state.json", {
            "method": METHOD, "schema_version": SCHEMA, "config": asdict(self.config),
            "compatibility": self.compatibility,
            "archive": [p.to_dict() for p in self.archive.values()],
            "attempts": [asdict(a) for a in self.attempts], "experiences": self.pool.to_list(),
            "experience_versions": self.pool.versions,
            "responses": [asdict(e) for e in self.responses],
            "evaluations_used": self.evaluations_used, "model_calls": self.model_calls,
            "consecutive_unevaluated": self.consecutive_unevaluated, "rng_state": self.rng.getstate()})

    def _flush_call(self, record):
        public = {k: v for k, v in record.items() if not k.startswith("_")}
        if "_payload" in record:
            self.store.write(record["path"], {**record["_payload"], **public,
                **({"response": record["_response"]} if "_response" in record else {})})
        elif self.store.path is not None:
            self.store.update(record["path"], public)

    def _restore(self, state):
        if (state["config"] != asdict(self.config)
                or state["compatibility"] != self.compatibility):
            raise ValueError("Resume requires matching task, model, prompts, tokenizer, source and configuration")
        self.archive = {p["id"]: Program.from_dict(p) for p in state["archive"]}
        self.attempts = [Attempt(**a) for a in state["attempts"]]
        self.responses = [ResponseEvent(**event) for event in state["responses"]]
        self.pool = ExperiencePool(state["experiences"], state["experience_versions"])
        self.evaluations_used, self.model_calls = state["evaluations_used"], state["model_calls"]
        self.consecutive_unevaluated = state["consecutive_unevaluated"]
        self.rng.setstate(_tuples(state["rng_state"]))
        for attempt in self.attempts:
            if attempt.completed:
                continue
            for record in attempt.calls:
                if record["status"] == "running":
                    record["status"] = "interrupted"
                    attempt.interruption = "A reserved request ended without a persisted reply"
                    self._flush_call(record)
                elif "_payload" in record:
                    self._flush_call(record)
            for record in attempt.evaluations:
                if record["status"] == "running":
                    record["status"] = "interrupted"
                    attempt.interruption = "A reserved evaluation ended without an outcome"
                program = self.archive[record["program_id"]]
                self.store.write(record["path"], {**record, "attempt_id": attempt.id,
                    "parent_id": attempt.parent_id, "edited_from": program.edited_from,
                    "candidate": {"code": program.code, "analysis": program.analysis, "design": program.design}})
        for program in self.archive.values():
            self.store.program(program)
        for event in self.responses:
            self.store.write(f"responses/{event.evaluation_id:06d}.json", asdict(event))
        self._save()

    @property
    def best(self):
        return min((p for p in self.archive.values() if p.valid),
                   key=lambda p: (p.evaluation.fitness, p.id), default=None)

    @property
    def stop_reason(self):
        if self.evaluations_used >= self.config.evaluation_budget:
            return "evaluation_budget_exhausted"
        if self.consecutive_unevaluated >= self.config.max_unevaluated:
            return "consecutive_unevaluated_limit"
        return None

    def _advance(self, attempt, phase):
        attempt.phase = phase
        for call in attempt.calls:
            if call["status"] != "running":
                call.pop("_payload", None)
                call.pop("_response", None)
        self._save()

    def _new_attempt(self):
        initialized = sum(a.operator == "Init" for a in self.attempts)
        phase = "generate"
        if initialized < self.config.initial_size or not candidates(self.archive):
            parent_id, operator, selection = 0, "Init", {}
        elif self.config.selection_mode == "trajectory":
            shown = proposals(self.archive, self.rng, self.config.candidate_count)
            fallback = self.rng.choice(shown)
            self.rng.shuffle(shown)
            parent_id, operator = fallback.id, "Develop"
            selection = {"selection_mode": "trajectory", "candidate_ids": [p.id for p in shown],
                         "fallback_id": fallback.id}
            phase = "select" if len(shown) > 1 else "generate"
        else:
            parent, selection = choose(self.archive, self.responses, self.rng,
                mode=self.config.selection_mode, window=self.config.response_window,
                strength=self.config.response_strength, floor=self.config.parent_floor)
            parent_id, operator = parent.id, "Develop"
        attempt = Attempt(len(self.attempts) + 1, parent_id, operator, selection, phase=phase)
        if operator == "Develop" and phase == "generate":
            self._retrieve(attempt)
        self.attempts.append(attempt)
        self._save()
        return attempt

    def _retrieve(self, attempt):
        query = self.prompts.query(self.archive, self.attempts, self.archive[attempt.parent_id])
        retrieved = self.pool.retrieve(query, self.config.experience_k, self.rng)
        attempt.selection.update(query=query, retrieved_experience_ids=[e["id"] for e in retrieved],
                                 pending_retrieved=retrieved)

    def _select(self, attempt):
        """Read the candidates' local trajectories side by side and develop the chosen one."""
        ids = attempt.selection["candidate_ids"]
        response = self._request(attempt, "select", lambda: self.prompts.selection(self.archive, self.attempts, ids))
        if response is not None and "choice_status" not in attempt.selection:
            try:
                assessment, choice = parse_selection(response, ids)
            except FormatError as exc:
                attempt.selection["choice_status"] = "format_failure"
                attempt.calls[-1]["format_failure"] = str(exc)
                self._flush_call(attempt.calls[-1])
            else:
                attempt.parent_id = choice
                attempt.selection.update(choice_status="chosen", assessment=assessment)
        elif response is None and "choice_status" not in attempt.selection:
            attempt.selection["choice_status"] = attempt.calls[-1]["status"]
        self._retrieve(attempt)
        self._advance(attempt, "generate")

    def _generation_prompt(self, attempt):
        if attempt.operator == "Init":
            prompt = self.prompts.initialization(self.archive)
        else:
            prompt = self.prompts.generation(self.archive, self.attempts,
                self.archive[attempt.parent_id], attempt.selection.get("pending_retrieved", []))
        attempt.context = prompt.metadata
        attempt.provided_experiences = prompt.provided
        self._save()
        return prompt

    def _request(self, attempt, kind, build):
        if attempt.calls and attempt.calls[-1]["kind"] == kind:
            record = attempt.calls[-1]
            if record["status"] == "success":
                return record["_response"]
            if record["status"] in {"error", "interrupted"}:
                return None
        prompt = build()
        role = ROLES.get(kind, "generator")
        limit = self.config.max_tokens
        with getattr(self.model, "request_context", nullcontext)():
            self.model_calls += 1
            record = {"id": self.model_calls, "kind": kind, "role": role,
                      "stage": kind if kind in ROLES or kind == "generate" else "repair",
                      "status": "running", "path": f"calls/{self.model_calls:06d}.json",
                      "_payload": {"attempt_id": attempt.id, "parent_id": attempt.parent_id,
                          "operator": attempt.operator, "prompt": prompt.text, "context": prompt.metadata,
                          "temperature": self.config.temperature, "max_tokens": limit,
                          "timeout": self.config.model_timeout, "response_format": response_format(role)}}
            attempt.calls.append(record)
            if kind == "reflect":
                attempt.reflection_context = prompt.metadata
                attempt.reflection_status = "requested"
            elif kind == "long_term":
                attempt.long_term_context = prompt.metadata
                attempt.long_term_status = "requested"
            self._save()
            self._flush_call(record)
            started = time.monotonic()
            try:
                reply = self.model.complete(prompt.text, temperature=self.config.temperature,
                    max_tokens=limit, timeout=self.config.model_timeout, response_format=response_format(role))
                if isinstance(reply, str):
                    reply = ModelReply(reply)
                if not isinstance(reply, ModelReply) or not isinstance(reply.content, str):
                    raise TypeError("Model.complete must return text or ModelReply")
            except Exception as exc:
                record.update(status="error", error=f"{type(exc).__name__}: {exc}",
                              seconds=time.monotonic() - started,
                              metadata=getattr(self.model, "last_metadata", {}))
                self._save()
                self._flush_call(record)
                return None
            record.update(status="success", _response=reply.content, seconds=time.monotonic() - started,
                          model=reply.model, finish_reason=reply.finish_reason, usage=reply.usage,
                          metadata=reply.metadata)
            self._save()
            self._flush_call(record)
            return reply.content

    def _parse(self, attempt, response):
        if response is None:
            return None
        try:
            return Candidate.parse(response)
        except FormatError as exc:
            attempt.calls[-1]["format_failure"] = str(exc)
            self._save()
            self._flush_call(attempt.calls[-1])
            return None

    def _evaluate(self, attempt):
        if attempt.evaluations and attempt.evaluations[-1]["phase"] == attempt.candidate_phase:
            # An interrupted reservation has already spent its evaluation slot.
            return
        candidate = Candidate(**attempt.pending_candidate)
        if self.evaluations_used >= self.config.evaluation_budget:
            raise RuntimeError("No evaluation budget remains")
        with getattr(self.evaluator, "evaluation_context", nullcontext)():
            pid = max(self.archive, default=0) + 1
            edited_from = attempt.version_ids[-1] if attempt.candidate_phase == "repair" else attempt.parent_id
            program = Program(pid, candidate.code, candidate.design, attempt.operator, attempt.parent_id,
                None, candidate.analysis, attempt.id, edited_from, attempt.candidate_phase)
            self.archive[pid] = program
            attempt.version_ids.append(pid)
            self.evaluations_used += 1
            record = {"id": self.evaluations_used, "program_id": pid, "phase": attempt.candidate_phase,
                      "status": "running", "path": f"evaluations/{self.evaluations_used:06d}.json",
                      "resources": getattr(self.evaluator, "last_dispatch", {})}
            attempt.evaluations.append(record)
            self._save()
            self.store.write(record["path"], {**record, "attempt_id": attempt.id,
                "parent_id": attempt.parent_id, "edited_from": edited_from, "candidate": asdict(candidate)})
            self.store.program(program)
            started = time.monotonic()
            evaluation = self.evaluator.evaluate(candidate.code)
            if not isinstance(evaluation, Evaluation):
                raise TypeError("Evaluator must return Evaluation")
            previous = None
            if evaluation.status == "success":
                key = source_key(program.code)
                previous = next((p.id for p in self.archive.values() if p.valid and p.id < pid
                                 and source_key(p.code) == key), None)
            program = replace(program, evaluation=evaluation, duplicate_of=previous)
            self.archive[pid] = program
            record.update(status=evaluation.status, evaluation=asdict(evaluation),
                          seconds=time.monotonic() - started)
            if attempt.operator != "Init":
                parent = self.archive[attempt.parent_id]
                improved = bool(program.valid and evaluation.fitness < parent.evaluation.fitness)
                novel = bool(program.valid and previous is None)
                response = int(novel and improved)
                event = ResponseEvent(record["id"], attempt.id, parent.id, pid, program.valid, novel, improved,
                                      response, parent.evaluation.fitness - evaluation.fitness if response else 0.)
                self.responses.append(event)
            attempt.program_id = pid
            self._save()
            self.store.update(record["path"], record)
            self.store.program(program)
            if attempt.operator != "Init":
                self.store.write(f"responses/{record['id']:06d}.json", asdict(event))

    def _outcome(self, attempt):
        return next((self.archive[pid] for pid in reversed(attempt.version_ids)
                     if self.archive[pid].evaluation is not None), None)

    def _learn(self, attempt):
        outcome = self._outcome(attempt)
        if not attempt.outcome_applied:
            improved = None
            if outcome is not None:
                parent = self.archive.get(attempt.parent_id)
                improved = outcome.valid and parent is not None and parent.valid and outcome.evaluation.fitness < parent.evaluation.fitness
                attempt.program_id = outcome.id
                attempt.outcome_fitness = outcome.evaluation.fitness
                attempt.duplicate_of = outcome.duplicate_of
            if any(call["kind"] == "generate" for call in attempt.calls):
                self.pool.record_usage([e["id"] for e in attempt.provided_experiences],
                                       attempt_id=attempt.id, improved=improved)
            attempt.outcome_applied = True
        phase = "reflect" if (outcome is not None and attempt.operator != "Init"
                              and self.config.reflection_mode != "none" and self._new_outcome(attempt)) else "finish"
        if phase == "finish":
            attempt.reflection_status = ("not_applicable" if attempt.operator == "Init" else "no_evaluation" if outcome is None
                                         else "disabled" if self.config.reflection_mode == "none" else "no_new_source")
        self._advance(attempt, phase)

    def _new_outcome(self, attempt):
        """A failure or a new source tested an idea; a repeated source tested nothing new."""
        return any(not v.valid or v.duplicate_of is None
                   for v in map(self.archive.get, attempt.version_ids) if v.evaluation is not None)

    def _reflect(self, attempt):
        """Short-term reflection: a local hint kept on the attempt and visible in later trajectories."""
        response = self._request(attempt, "reflect", lambda: self.prompts.reflection(self.archive, self.attempts, attempt))
        if response is not None and not attempt.reflection_applied:
            try:
                attempt.reflection = parse_reflection(response)
            except FormatError as exc:
                attempt.reflection_status = "format_failure"
                attempt.calls[-1]["format_failure"] = str(exc)
                self._flush_call(attempt.calls[-1])
            else:
                attempt.reflection_applied = True
                attempt.reflection_status = "completed"
        elif response is None and attempt.reflection_status == "requested":
            attempt.reflection_status = attempt.calls[-1]["status"]
        due = len(self._unmerged_reflections()) >= self.config.long_term_interval
        self._advance(attempt, "long_term" if due else "finish")

    def _unmerged_reflections(self):
        merged = self.pool.reflected_attempt_ids
        return [a for a in self.attempts if a.reflection and a.id not in merged]

    def _long_term(self, attempt):
        """Long-term reflection rebuilds the global pools from the most recent short-term reflections."""
        recent = [a for a in self.attempts if a.reflection][-self.config.long_term_window:]
        response = self._request(attempt, "long_term", lambda: self.prompts.long_term(self.archive, self.attempts, recent))
        if response is not None and attempt.long_term_status == "requested":
            try:
                lessons, attempt.discarded_experiences = parse_experience(response)
            except FormatError as exc:
                attempt.long_term_status = "format_failure"
                attempt.calls[-1]["format_failure"] = str(exc)
                self._flush_call(attempt.calls[-1])
            else:
                self.pool.replace(lessons, {"attempt_id": attempt.id, "reflection_attempt_ids":
                                            attempt.long_term_context["reflection_attempt_ids"]})
                attempt.long_term_status = "completed"
        elif response is None and attempt.long_term_status == "requested":
            attempt.long_term_status = attempt.calls[-1]["status"]
        self._advance(attempt, "finish")

    def _finish(self, attempt):
        attempt.completed = True
        attempt.phase = "completed"
        attempt.pending_candidate = None
        attempt.selection.pop("pending_retrieved", None)
        self.consecutive_unevaluated = 0 if attempt.evaluations else self.consecutive_unevaluated + 1
        self._save()
        self.store.write(f"attempts/{attempt.id:06d}.json", asdict(attempt))

    def step(self):
        pending = next((a for a in self.attempts if not a.completed), None)
        if pending is None and self.stop_reason:
            return None
        attempt = pending or self._new_attempt()
        while not attempt.completed:
            phase = attempt.phase
            if phase in {"generate", "format", "program"}:
                if phase == "generate":
                    build = lambda: self._generation_prompt(attempt)
                elif phase == "format":
                    original = attempt.pending_candidate["original_response"]
                    failure = attempt.pending_candidate["format_failure"]
                    build = lambda: self.prompts.repair_format(original, failure)
                else:
                    build = lambda: self.prompts.repair_program(self._outcome(attempt))
                response = self._request(attempt, phase, build)
                candidate = self._parse(attempt, response)
                if candidate is not None:
                    attempt.pending_candidate = asdict(candidate)
                    attempt.candidate_phase = "repair" if phase == "program" else "generation"
                    self._advance(attempt, "evaluate")
                elif phase == "generate" and response is not None and not attempt.repair_used:
                    attempt.repair_used = True
                    attempt.pending_candidate = {"original_response": response,
                        "format_failure": attempt.calls[-1]["format_failure"]}
                    self._advance(attempt, "format")
                else:
                    self._advance(attempt, "learn")
            elif phase == "evaluate":
                self._evaluate(attempt)
                outcome = self._outcome(attempt)
                if (outcome is not None and not outcome.valid and not attempt.repair_used
                        and self.evaluations_used < self.config.evaluation_budget
                        and attempt.evaluations[-1]["status"] != "interrupted"):
                    attempt.repair_used = True
                    self._advance(attempt, "program")
                else:
                    self._advance(attempt, "learn")
            elif phase == "select":
                self._select(attempt)
            elif phase == "learn":
                self._learn(attempt)
            elif phase == "reflect":
                self._reflect(attempt)
            elif phase == "long_term":
                self._long_term(attempt)
            elif phase == "finish":
                self._finish(attempt)
            else:
                raise ValueError(f"Unsupported attempt phase: {phase}")
        return attempt

    def run(self, *, on_attempt=None):
        while (attempt := self.step()) is not None:
            if on_attempt:
                on_attempt(attempt)
        result = SearchResult("finished" if self.best is not None else "failed", self.stop_reason,
            self.best, self.evaluations_used, self.model_calls, len(self.attempts), len(candidates(self.archive)))
        self.store.write("result.json", result.to_dict())
        self.store.write("experiences.json", {"entries": self.pool.to_list(), "versions": self.pool.versions})
        if self.best is not None and self.store.path is not None:
            (self.store.path / "best.py").write_text(self.best.code)
        return result

    def close(self):
        self.store.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
