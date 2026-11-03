"""Sequential Base search with one shared repair opportunity per attempt."""

from dataclasses import asdict
from contextlib import nullcontext
import random
import time

from .config import Config
from .evaluation import Evaluator
from .models import Attempt, Candidate, Evaluation, FormatError, ModelReply, Program, SearchResult
from .prompts import PromptBuilder
from .selection import choose
from .storage import Store
from .task import Task


def _tuples(value):
    return tuple(_tuples(v) for v in value) if isinstance(value, list) else value


class TraceAAD:
    def __init__(self, task: Task, model, *, config: Config | None = None,
                 run_dir=None, resume=False):
        self.task = task
        self.model = model
        self.store = Store(run_dir, resume=resume)
        try:
            state = self.store.read("state.json") if resume else None
            self.config = config or (Config(**state["config"]) if state else Config())
            self.prompts = PromptBuilder(task, self.config)
            self.evaluator = Evaluator(task)
            self.rng = random.Random(self.config.seed)
            self.archive: dict[int, Program] = {}
            self.attempts: list[Attempt] = []
            self.evaluations_used = 0
            self.model_calls = 0
            self.consecutive_unevaluated = 0
            self.next_program_id = 1
            if state:
                self._restore(state)
            else:
                if task.example_code is not None:
                    example = Program(0, task.example_code, task.example_design, None, None, None, None)
                    self.archive[0] = example
                    self.store.program(example)
                self.store.write("run_config.json", {
                    "method": "traceaad_v19_1", "config": asdict(self.config),
                    "task": task.description(),
                    "model": getattr(model, "model", type(model).__name__),
                })
                self.store.snapshot()
                self._save()
        except BaseException:
            self.store.close()
            raise

    def _save(self):
        self.store.write("state.json", {
            "schema_version": 2,
            "config": asdict(self.config),
            "task_fingerprint": self.task.fingerprint(),
            "archive": [self.archive[k].to_dict() for k in sorted(self.archive)],
            "attempts": [asdict(attempt) for attempt in self.attempts],
            "evaluations_used": self.evaluations_used,
            "model_calls": self.model_calls,
            "consecutive_unevaluated": self.consecutive_unevaluated,
            "next_program_id": self.next_program_id,
            "rng_state": self.rng.getstate(),
        })

    def _restore(self, state):
        if state["schema_version"] != 2:
            raise ValueError("Unsupported checkpoint schema")
        if state["config"] != asdict(self.config) or state["task_fingerprint"] != self.task.fingerprint():
            raise ValueError("Resume requires the same task and search configuration")
        self.archive = {p["id"]: Program.from_dict(p) for p in state["archive"]}
        self.attempts = [Attempt(**a) for a in state["attempts"]]
        self.evaluations_used = state["evaluations_used"]
        self.model_calls = state["model_calls"]
        self.consecutive_unevaluated = state["consecutive_unevaluated"]
        self.next_program_id = state["next_program_id"]
        self.rng.setstate(_tuples(state["rng_state"]))
        for attempt in self.attempts:
            if attempt.completed:
                continue
            attempt.interruption = "The previous process ended during this attempt"
            for record in (*attempt.calls, *attempt.evaluations):
                if record["status"] == "running":
                    record["status"] = "interrupted"
                    self.store.update(record["path"], {"status": "interrupted"})
            self._finish(attempt)
        # A run may have ended between a checkpoint and an auxiliary artifact write.
        for program in self.archive.values():
            self.store.program(program)

    @property
    def best(self) -> Program | None:
        return min((p for p in self.archive.values() if p.valid),
                   key=lambda p: (p.evaluation.fitness, p.id), default=None)

    @property
    def stop_reason(self) -> str | None:
        if self.evaluations_used >= self.config.evaluation_budget:
            return "evaluation_budget_exhausted"
        if self.consecutive_unevaluated >= self.config.max_unevaluated:
            return "consecutive_unevaluated_limit"
        return None

    def _model_call(self, attempt: Attempt, prompt: str, repair_kind=None) -> str | None:
        # A resource queue is outside both request accounting and request timeout.
        with getattr(self.model, "request_context", nullcontext)():
            return self._request_model(attempt, prompt, repair_kind)

    def _request_model(self, attempt: Attempt, prompt: str, repair_kind=None) -> str | None:
        self.model_calls += 1
        number = self.model_calls
        record = {"id": number, "stage": "repair" if repair_kind else "generate",
                  "repair_kind": repair_kind, "status": "running",
                  "path": f"calls/{number:06d}.json"}
        attempt.calls.append(record)
        self.store.write(record["path"], {
            **record, "attempt_id": attempt.id, "parent_id": attempt.parent_id,
            "operator": attempt.operator, "reference_id": attempt.reference_id,
            "prompt": prompt, "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens, "timeout": self.config.model_timeout,
        })
        self._save()
        started = time.monotonic()
        try:
            reply = self.model.complete(prompt, temperature=self.config.temperature,
                                        max_tokens=self.config.max_tokens,
                                        timeout=self.config.model_timeout)
            if isinstance(reply, str):
                reply = ModelReply(reply)
            if not isinstance(reply, ModelReply) or not isinstance(reply.content, str):
                raise TypeError("Model.complete must return text or ModelReply")
        except Exception as exc:
            record.update(status="error", error=f"{type(exc).__name__}: {exc}",
                          seconds=time.monotonic() - started,
                          metadata=getattr(self.model, "last_metadata", {}))
            self.store.update(record["path"], record)
            self._save()
            return None
        record.update(status="success", seconds=time.monotonic() - started,
                      model=reply.model, finish_reason=reply.finish_reason, usage=reply.usage,
                      metadata=reply.metadata)
        self.store.update(record["path"], {**record, "response": reply.content})
        self._save()
        return reply.content

    def _parse(self, attempt: Attempt, response: str) -> Candidate | None:
        try:
            return Candidate.parse(response)
        except FormatError as exc:
            record = attempt.calls[-1]
            record["format_failure"] = str(exc)
            self.store.update(record["path"], {"format_failure": str(exc)})
            self._save()
            return None

    def _evaluate(self, attempt: Attempt, candidate: Candidate):
        with getattr(self.evaluator, "evaluation_context", nullcontext)():
            return self._evaluate_reserved(attempt, candidate)

    def _evaluate_reserved(self, attempt: Attempt, candidate: Candidate):
        if self.evaluations_used >= self.config.evaluation_budget:
            raise RuntimeError("No evaluation budget remains")
        self.evaluations_used += 1
        record = {"id": self.evaluations_used, "status": "running",
                  "repair": attempt.repair_used,
                  "resources": getattr(self.evaluator, "last_dispatch", {}),
                  "path": f"evaluations/{self.evaluations_used:06d}.json"}
        attempt.evaluations.append(record)
        self.store.write(record["path"], {
            **record, "attempt_id": attempt.id, "parent_id": attempt.parent_id,
            "operator": attempt.operator, "reference_id": attempt.reference_id,
            "candidate": asdict(candidate),
        })
        self._save()
        started = time.monotonic()
        evaluation = self.evaluator.evaluate(candidate.code)
        record.update(status=evaluation.status, evaluation=asdict(evaluation),
                      seconds=time.monotonic() - started)
        if attempt.program_id is None:
            program = Program(self.next_program_id, candidate.code, candidate.design,
                              attempt.operator, attempt.parent_id, attempt.reference_id, evaluation,
                              analysis=candidate.analysis)
            self.next_program_id += 1
            attempt.program_id = program.id
            self.archive[program.id] = program
        else:
            program = self.archive[attempt.program_id]
            program.code, program.design, program.evaluation = candidate.code, candidate.design, evaluation
            program.analysis = candidate.analysis
        record["program_id"] = program.id
        self.store.update(record["path"], record)
        self._save()
        self.store.program(program)

    def _finish(self, attempt: Attempt):
        if attempt.completed:
            return
        attempt.completed = True
        if attempt.evaluations:
            self.consecutive_unevaluated = 0
        else:
            self.consecutive_unevaluated += 1
        self._save()
        self.store.write(f"attempts/{attempt.id:06d}.json", asdict(attempt))

    def step(self) -> Attempt | None:
        """Complete one generation attempt, including at most one repair."""
        if self.stop_reason is not None:
            return None
        initialized = sum(p.valid and p.operator == "Init" for p in self.archive.values())
        selection = {}
        if initialized < self.config.initial_size:
            parent_id, operator, reference_id = 0, "Init", None
            prompt = self.prompts.initialization(self.archive)
        else:
            parent, operator, reference, selection = choose(
                self.archive, self.attempts, self.config, self.rng
            )
            parent_id, reference_id = parent.id, reference.id if reference else None
            prompt = self.prompts.evolution(self.archive, parent, operator, reference)
        attempt = Attempt(
            len(self.attempts) + 1,
            parent_id,
            operator,
            reference_id,
            selection,
        )
        self.attempts.append(attempt)
        self._save()
        response = self._model_call(attempt, prompt)
        if response is not None:
            candidate = self._parse(attempt, response)
            if candidate is None:
                attempt.repair_used = True
                prompt = self.prompts.repair_format(response, attempt.calls[-1]["format_failure"])
                response = self._model_call(attempt, prompt, "format")
                candidate = self._parse(attempt, response) if response is not None else None
            if candidate is not None:
                self._evaluate(attempt, candidate)
                program = self.archive[attempt.program_id]
                if (not program.valid and not attempt.repair_used
                        and self.evaluations_used < self.config.evaluation_budget):
                    attempt.repair_used = True
                    prompt = self.prompts.repair_program(program)
                    response = self._model_call(attempt, prompt, program.evaluation.status)
                    repaired = self._parse(attempt, response) if response is not None else None
                    if repaired is not None:
                        self._evaluate(attempt, repaired)
        self._finish(attempt)
        return attempt

    def run(self, *, on_attempt=None) -> SearchResult:
        """Run to a stopping condition; optionally report each completed attempt."""
        try:
            while (attempt := self.step()) is not None:
                if on_attempt is not None:
                    on_attempt(attempt)
            result = SearchResult(
                "finished" if self.best is not None else "failed", self.stop_reason, self.best,
                self.evaluations_used, self.model_calls, len(self.attempts),
                sum(p.valid for p in self.archive.values()),
            )
            self.store.write("result.json", result.to_dict())
            if result.best is not None and self.store.path is not None:
                (self.store.path / "best.py").write_text(result.best.code)
            return result
        finally:
            self.close()

    def close(self):
        self.store.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

