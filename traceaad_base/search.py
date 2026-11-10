"""Sequential Base search with one shared repair opportunity per attempt.

Each finished attempt is appended to ``attempts.jsonl`` together with its
program. Resuming reads those rows back into the archive and continues; an
attempt cut off by the interruption is simply lost.
"""

from contextlib import nullcontext
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import time

from .config import Config
from .evaluation import Evaluator
from .models import Attempt, Candidate, FormatError, ModelReply, Program, SearchResult
from .prompts import PromptBuilder
from .selection import SELECTION_RULE, choose, reference_pool
from .storage import Store
from .task import Task


class TraceAAD:
    method = "traceaad_base"
    prompt_type = PromptBuilder
    generation_protocol = "static_algorithm_context_v1"
    reference_rule = "quality_pool_uniform_v1"
    mechanisms = ()

    def __init__(self, task: Task, model, *, config: Config | None = None,
                 run_dir=None, resume=False):
        self.task = task
        self.model = model
        self.config = config or Config()
        self.store = Store(run_dir)
        self.prompts = self.prompt_type(task, self.config)
        self.evaluator = Evaluator(task)
        self.archive: dict[int, Program] = {}
        self.attempts: list[Attempt] = []
        self.rng = random.Random(self.config.seed)
        if task.example_code is not None:
            self.archive[0] = Program(0, task.example_code, task.example_design, None, None, None, None)
        if resume:
            self._validate_resume(task, self.config, run_dir)
            for name in ("attempts.jsonl", "calls.jsonl"):
                self.store.repair_tail(name)
            for row in self.store.rows("attempts.jsonl"):
                attempt = Attempt(**row["attempt"])
                self.attempts.append(attempt)
                if row["program"] is not None:
                    program = Program.from_dict(row["program"])
                    self.archive[program.id] = program
                self.restore(attempt)
        else:
            self.store.write("run_config.json", {
                "method": self.method, "config": asdict(self.config), "task": task.description(),
                "model": getattr(model, "model", type(model).__name__),
                "selection_rule": SELECTION_RULE,
                "generation_protocol": self.generation_protocol,
                "reference_rule": self.reference_rule,
                "mechanisms": list(self.mechanisms),
            })

    def _validate_resume(self, task, config, directory):
        """Validate shared conditions before repairing or replaying journals."""
        path = Path(directory) / "run_config.json" if directory is not None else None
        if path is None or not path.is_file():
            raise ValueError("Resume requires its original run_config.json and original source snapshot")
        saved = json.loads(path.read_text())
        version = self.method.removeprefix("traceaad_").upper().replace("_", ".")
        differences = [name for name, value in {
            "method": self.method, "generation_protocol": self.generation_protocol,
        }.items() if saved.get(name) != value]
        if differences:
            raise ValueError(f"Incompatible {version} resume settings: " + ", ".join(differences)
                             + "; use the original source snapshot for older runs")
        if (saved.get("selection_rule") != SELECTION_RULE
                or saved.get("config", {}).get("selection_temperature") != config.selection_temperature):
            raise ValueError("Resume requires the same Boltzmann selection rule and temperature; "
                             "use the original source snapshot for older runs")
        expected = {"config": asdict(config), "task": json.loads(json.dumps(task.description())),
                    "reference_rule": self.reference_rule, "mechanisms": list(self.mechanisms)}
        differences = [name for name, value in expected.items() if saved.get(name) != value]
        if differences:
            raise ValueError(f"Incompatible {version} resume settings: " + ", ".join(differences))

    # Extension points for methods built on Base.
    def restore(self, attempt: Attempt):
        """Rebuild method state from one saved attempt, in order."""

    def select(self, attempt_id: int):
        parent, operator = choose(self.archive, self.config, self.rng)
        reference = self.reference(parent) if operator == "Crossover" else None
        if operator == "Crossover" and reference is None:
            operator = "Explore"
        return parent, operator, reference, {}

    def reference(self, parent: Program) -> Program | None:
        pool = reference_pool(self.archive, parent.id)
        return self.rng.choice(pool) if pool else None

    def evolution_prompt(self, attempt: Attempt, parent: Program, reference: Program | None) -> str:
        return self.prompts.evolution(self.archive, parent, attempt.operator, reference)

    def finish(self, attempt: Attempt):
        """Learn from one evaluated attempt before it is saved."""

    @property
    def evaluations_used(self) -> int:
        return sum(len(a.evaluations) for a in self.attempts)

    @property
    def model_calls(self) -> int:
        return sum(len(a.calls) for a in self.attempts)

    @property
    def consecutive_unevaluated(self) -> int:
        count = 0
        for attempt in reversed(self.attempts):
            if attempt.evaluations:
                break
            count += 1
        return count

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

    def _request(self, model, prompt: str, record: dict) -> str | None:
        """Send one request; ``record`` receives its status, usage and duration."""
        # The scheduler queue lies outside the request's own deadline.
        with getattr(model, "request_context", nullcontext)():
            started = time.monotonic()
            try:
                reply = model.complete(prompt, temperature=self.config.temperature,
                                       max_tokens=self.config.max_tokens,
                                       timeout=self.config.model_timeout)
                if isinstance(reply, str):
                    reply = ModelReply(reply)
                record.update(status="success", model=reply.model, finish_reason=reply.finish_reason,
                              usage=reply.usage, metadata=reply.metadata)
                content = reply.content
            except Exception as exc:
                record.update(status="error", error=f"{type(exc).__name__}: {exc}",
                              metadata=getattr(model, "last_metadata", {}))
                content = None
            record["seconds"] = time.monotonic() - started
        return content

    def _call(self, attempt: Attempt, prompt: str, repair_kind=None) -> str | None:
        record = {"stage": "repair" if repair_kind else "generate", "repair_kind": repair_kind}
        content = self._request(self.model, prompt, record)
        attempt.calls.append(record)
        self.store.append("calls.jsonl", {**record, "attempt_id": attempt.id,
                                          "prompt": prompt, "response": content})
        return content

    def _parse(self, attempt: Attempt, response: str) -> Candidate | None:
        try:
            return Candidate.parse(response)
        except FormatError as exc:
            attempt.calls[-1]["format_failure"] = str(exc)
            return None

    def _evaluate(self, attempt: Attempt, candidate: Candidate):
        with getattr(self.evaluator, "evaluation_context", nullcontext)():
            started = time.monotonic()
            evaluation = self.evaluator.evaluate(candidate.code)
            seconds = time.monotonic() - started
        if attempt.program_id is None:
            program_id = max(self.archive, default=0) + 1
            attempt.program_id = program_id
        program = Program(attempt.program_id, candidate.code, candidate.design, attempt.operator,
                          attempt.parent_id, attempt.reference_id, evaluation,
                          analysis=candidate.analysis)
        self.archive[program.id] = program
        attempt.evaluations.append({
            "id": self.evaluations_used + 1, "status": evaluation.status, "evaluation": asdict(evaluation),
            "seconds": seconds,
            "repair": attempt.repair_used, "program_id": program.id,
            "resources": getattr(self.evaluator, "last_dispatch", {}),
        })

    def step(self) -> Attempt | None:
        """Complete one generation attempt, including at most one repair."""
        if self.stop_reason is not None:
            return None
        attempt_id = len(self.attempts) + 1
        # One random stream per attempt, so a resumed run draws as an uninterrupted one.
        self.rng = random.Random(f"{self.config.seed}:{attempt_id}")
        initialized = sum(p.valid and p.operator == "Init" for p in self.archive.values())
        if initialized < self.config.initial_size:
            attempt = Attempt(attempt_id, 0, "Init")
            prompt = self.prompts.initialization(self.archive)
        else:
            parent, operator, reference, record = self.select(attempt_id)
            attempt = Attempt(attempt_id, parent.id, operator,
                              reference.id if reference else None, record=record)
            prompt = self.evolution_prompt(attempt, parent, reference)
        self.attempts.append(attempt)
        response = self._call(attempt, prompt)
        if response is not None:
            candidate = self._parse(attempt, response)
            if candidate is None:
                attempt.repair_used = True
                prompt = self.prompts.repair_format(response, attempt.calls[-1]["format_failure"])
                response = self._call(attempt, prompt, "format")
                candidate = self._parse(attempt, response) if response is not None else None
            if candidate is not None:
                self._evaluate(attempt, candidate)
                program = self.archive[attempt.program_id]
                if (not program.valid and not attempt.repair_used
                        and self.evaluations_used < self.config.evaluation_budget):
                    attempt.repair_used = True
                    prompt = self.prompts.repair_program(program)
                    response = self._call(attempt, prompt, program.evaluation.status)
                    repaired = self._parse(attempt, response) if response is not None else None
                    if repaired is not None:
                        analysis = f"{program.analysis}\nRepair: {repaired.analysis}"
                        self._evaluate(attempt, replace(repaired, analysis=analysis))
        self.finish(attempt)
        attempt.finished_at = time.time()
        program = self.archive.get(attempt.program_id)
        self.store.append("attempts.jsonl", {"attempt": asdict(attempt),
                                             "program": program.to_dict() if program else None})
        return attempt

    def run(self, *, on_attempt=None) -> SearchResult:
        """Run to a stopping condition; optionally report each finished attempt."""
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

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
