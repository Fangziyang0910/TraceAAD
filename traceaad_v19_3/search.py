"""Sequential algorithm evolution with post-evaluation trajectory reflection."""

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
from .models import Attempt, Candidate, Evaluation, FormatError, ModelReply, Program, SearchResult, parse_lessons
from .prompts import ContextOverflow, PromptBuilder
from .schemas import response_format
from .selection import candidates, choose, source_key
from .storage import Store
from .tokens import TokenCounter


SCHEMA = 1
METHOD = "traceaad_v19_3"
REVISION = "global_trajectory_reflection_20261101"


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
                raise ValueError("V19.3 requires its own checkpoint schema and method")
            self.config = config or (Config(**state["config"]) if state else Config())
            self.count = token_counter or TokenCounter(self.config.tokenizer)
            self.prompts = PromptBuilder(task, self.config, self.count)
            self.evaluator = evaluator or Evaluator(task)
            self.rng = random.Random(self.config.seed)
            self.archive = {}
            self.attempts = []
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
                    "method": METHOD, "mechanism_revision": REVISION, "checkpoint_schema": SCHEMA,
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
        if state["config"] != asdict(self.config) or state["compatibility"] != self.compatibility:
            raise ValueError("Resume requires matching task, model, prompts, tokenizer, source and configuration")
        self.archive = {p["id"]: Program.from_dict(p) for p in state["archive"]}
        self.attempts = [Attempt(**a) for a in state["attempts"]]
        self.pool = ExperiencePool(state["experiences"])
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

    def root(self, pid):
        seen = set()
        while pid in self.archive and self.archive[pid].parent_id not in (None, 0):
            if pid in seen:
                raise ValueError("Cyclic ancestry")
            seen.add(pid)
            pid = self.archive[pid].parent_id
        return pid

    def _advance(self, attempt, phase):
        attempt.phase = phase
        for call in attempt.calls:
            if call["status"] != "running":
                call.pop("_payload", None)
                call.pop("_response", None)
        self._save()

    def _new_attempt(self):
        initialized = sum(p.operator == "Init" for p in candidates(self.archive))
        if initialized < self.config.initial_size:
            parent_id, operator, selection = 0, "Init", {}
        else:
            parent, selection = choose(self.archive, self.rng, self.config.parent_floor)
            parent_id, operator = parent.id, "Develop"
            query = self.prompts.query(self.archive, self.attempts, parent)
            retrieved = self.pool.retrieve(query, self.config.experience_k, self.rng)
            selection.update(query=query, retrieved_experience_ids=[e["id"] for e in retrieved],
                             pending_retrieved=retrieved)
        attempt = Attempt(len(self.attempts) + 1, parent_id, operator, selection)
        self.attempts.append(attempt)
        self._save()
        return attempt

    def _generation_prompt(self, attempt):
        if attempt.operator == "Init":
            prompt = self.prompts.initialization(self.archive)
        else:
            prompt = self.prompts.generation(self.archive, self.attempts,
                self.archive[attempt.parent_id], attempt.selection.get("pending_retrieved", []))
            target_root = self.root(attempt.parent_id)
            for entry in prompt.provided:
                entry["same_lineage"] = [o.get("root_id") == target_root for o in entry["origins"]]
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
        try:
            prompt = build()
        except ContextOverflow as exc:
            attempt.failure = str(exc)
            if kind == "reflect":
                attempt.reflection_status = "context_failure"
            self._save()
            return None
        role = "reflector" if kind == "reflect" else "generator"
        limit = self.config.reflection_max_tokens if role == "reflector" else self.config.max_tokens
        with getattr(self.model, "request_context", nullcontext)():
            self.model_calls += 1
            record = {"id": self.model_calls, "kind": kind, "role": role,
                      "stage": "reflect" if kind == "reflect" else "generate" if kind == "generate" else "repair",
                      "status": "running", "path": f"calls/{self.model_calls:06d}.json",
                      "_payload": {"attempt_id": attempt.id, "parent_id": attempt.parent_id,
                          "operator": attempt.operator, "prompt": prompt.text, "context": prompt.metadata,
                          "temperature": self.config.temperature, "max_tokens": limit,
                          "timeout": self.config.model_timeout, "response_format": response_format(role)}}
            attempt.calls.append(record)
            if kind == "reflect":
                attempt.reflection_context = prompt.metadata
                attempt.reflection_status = "requested"
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
            attempt.program_id = pid
            self._save()
            self.store.update(record["path"], record)
            self.store.program(program)

    def _outcome(self, attempt):
        return next((self.archive[pid] for pid in reversed(attempt.version_ids)
                     if self.archive[pid].evaluation is not None), None)

    def _learn(self, attempt):
        outcome = self._outcome(attempt)
        if not attempt.utility_applied:
            if outcome is not None:
                parent = self.archive.get(attempt.parent_id)
                improved = outcome.valid and parent is not None and parent.valid and outcome.evaluation.fitness < parent.evaluation.fitness
                self.pool.feedback([e["id"] for e in attempt.provided_experiences],
                                   attempt_id=attempt.id, improved=improved)
                attempt.program_id = outcome.id
                attempt.outcome_fitness = outcome.evaluation.fitness
                attempt.duplicate_of = outcome.duplicate_of
            attempt.utility_applied = True
        phase = "reflect" if (outcome is not None and attempt.operator != "Init"
                              and self.config.reflection_mode != "none") else "finish"
        if phase == "finish":
            attempt.reflection_status = "not_applicable" if attempt.operator == "Init" else "disabled" if outcome else "no_evaluation"
        self._advance(attempt, phase)

    def _reflect(self, attempt):
        response = self._request(attempt, "reflect", lambda: self.prompts.reflection(
            self.archive, self.attempts, attempt, self._outcome(attempt)))
        if response is not None and not attempt.reflection_applied:
            try:
                lessons, discarded = parse_lessons(response, self.config.max_lessons)
            except FormatError as exc:
                attempt.reflection_status = "format_failure"
                attempt.calls[-1]["format_failure"] = str(exc)
                self._flush_call(attempt.calls[-1])
            else:
                attempt.reflection_raw_count = len(lessons) + len(discarded)
                attempt.reflection_accepted_count = len(lessons)
                origin = {"attempt_id": attempt.id, "program_id": attempt.program_id,
                          "root_id": self.root(attempt.parent_id),
                          "visible_attempt_ids": list(dict.fromkeys(
                              attempt.reflection_context.get("main_attempt_ids", [])
                              + attempt.reflection_context.get("alternative_attempt_ids", [])))}
                attempt.experience_ids = self.pool.add(lessons, origin)
                attempt.discarded_lessons = discarded
                attempt.reflection_applied = True
                attempt.reflection_status = "completed"
        elif response is None and attempt.reflection_status == "requested":
            attempt.reflection_status = attempt.calls[-1]["status"]
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
            elif phase == "learn":
                self._learn(attempt)
            elif phase == "reflect":
                self._reflect(attempt)
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
        self.store.write("experiences.json", self.pool.to_list())
        if self.best is not None and self.store.path is not None:
            (self.store.path / "best.py").write_text(self.best.code)
        return result

    def close(self):
        self.store.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
