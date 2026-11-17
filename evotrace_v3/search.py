"""EvoTrace search: generate, evaluate, grow the evolutionary tree, and simplify into experience."""

from contextlib import nullcontext
from copy import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import time
from typing import Any

from .allocation import AllocationConfig, allocate_step
from .context import evolution_prompt, init_prompt, repair_format_prompt, repair_program_prompt
from .evaluation import Evaluator
from .models import (
    CANDIDATE_RESPONSE_FORMAT, Attempt, Candidate, Config, Edge, FormatError,
    ModelReply, Node, Program, ROOT_ID, SearchResult, TreePopulation,
)
from .reflection import EXPERIENCE_SCHEMA, problem_header, reflect_and_apply
from .storage import Store
from .task import Task
from .topology import TopologyConfig, plan_simplification


def with_output_format(model: Any, schema: dict, *, role: bool = False) -> Any:
    if role and hasattr(model, "for_role"):
        return model.for_role(schema)
    if hasattr(model, "request_options"):
        model = copy(model)
        model.request_options = {**model.request_options, "response_format": schema}
    return model


class EvoTrace:
    """基于树状种群、反思式淘汰与代表集预算分配的自动算法设计搜索。"""

    method = "evotrace_v3"
    selection_rule = "representative_rank_direct_sampling"
    generation_protocol = "tree_population_context"
    mechanisms = (
        "tree_population",
        "reflective_tree_simplification",
        "representative_cover_allocation",
        "tree_context",
    )

    def __init__(
        self,
        task: Task,
        model: Any,
        *,
        config: Config | None = None,
        topology_config: TopologyConfig | None = None,
        allocation_config: AllocationConfig | None = None,
        include_experiences: bool = True,
        run_dir: str | Path | None = None,
        resume: bool = False,
    ) -> None:
        config = config or Config()
        if config.initial_size < 1:
            raise ValueError("EvoTrace requires initial_size >= 1")
        self.task = task
        self.config = config
        self.topology_config = topology_config or TopologyConfig()
        self.allocation_config = allocation_config or AllocationConfig()
        self.include_experiences = include_experiences

        self.tree = TreePopulation()
        self.model = with_output_format(model, CANDIDATE_RESPONSE_FORMAT)
        self.reflection_model = with_output_format(model, EXPERIENCE_SCHEMA, role=True)
        self.store = Store(run_dir)
        self.evaluator = Evaluator(task)
        self.archive: dict[int, Program] = {}
        self.attempts: list[Attempt] = []
        if task.example_code is not None:
            self.archive[0] = Program(0, task.example_code, task.example_design, None, None, None, None)
        if resume:
            self._validate_resume()
            for name in ("attempts.jsonl", "calls.jsonl"):
                self.store.repair_tail(name)
            for row in self.store.rows("attempts.jsonl"):
                attempt = Attempt(**row["attempt"])
                self.attempts.append(attempt)
                if row["program"] is not None:
                    program = Program.from_dict(row["program"])
                    self.archive[program.id] = program
                self._attach_evaluated_attempt(attempt)
                if "tree_after" in row:
                    self.tree = TreePopulation.from_dict(row["tree_after"])
        else:
            self.store.write("run_config.json", {
                **self._run_settings(),
                "model": getattr(self.model, "model", type(self.model).__name__),
            })

    def _run_settings(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "config": asdict(self.config),
            "task": self.task.description(),
            "selection_rule": self.selection_rule,
            "generation_protocol": self.generation_protocol,
            "mechanisms": list(self.mechanisms),
            "topology": asdict(self.topology_config),
            "allocation": asdict(self.allocation_config),
            "include_experiences": self.include_experiences,
        }

    def _validate_resume(self) -> None:
        saved = self.store.read("run_config.json")
        if saved is None:
            raise ValueError("Resume requires its original run_config.json")
        expected = json.loads(json.dumps(self._run_settings()))
        differences = [key for key, value in expected.items() if saved.get(key) != value]
        if differences:
            raise ValueError(
                "Incompatible EvoTrace resume settings: " + ", ".join(differences)
                + "; resume with its original frozen source"
            )

    def _attach_evaluated_attempt(self, attempt: Attempt) -> None:
        prog = self.archive.get(attempt.program_id) if attempt.program_id else None
        if prog is None or prog.evaluation is None:
            return
        op = None if attempt.operator == "Init" else attempt.operator
        parent_id = attempt.parent_id if op in ("Refine", "Crossover") else ROOT_ID
        self.tree.add(
            Node(prog.id, prog.code, prog.design or "", prog.evaluation.fitness if prog.valid else None, prog.evaluation.failure),
            Edge(parent_id, prog.id, op, attempt.reference_id if op == "Crossover" else None),  # type: ignore[arg-type]
            evaluations=len(attempt.evaluations),
        )

    def _update_population(self, attempt: Attempt) -> bool:
        self._attach_evaluated_attempt(attempt)
        if attempt.program_id not in self.tree.nodes:
            return False
        if sum(p.valid and p.operator == "Init" for p in self.archive.values()) < self.config.initial_size:
            return False
        regions = plan_simplification(self.tree, self.topology_config)
        if not regions:
            return False
        self.tree, attempt.record["aux"] = reflect_and_apply(
            self.tree,
            regions,
            model=self.reflection_model,
            problem=problem_header(
                self.task.task_description, self.task.score_definition, self.task.function_template
            ),
            timeout=min(self.config.model_timeout, 180.0),
        )
        return True

    @property
    def evaluations_used(self) -> int:
        return sum(len(a.evaluations) for a in self.attempts)

    @property
    def model_calls(self) -> int:
        return sum(len(a.calls) for a in self.attempts)

    @property
    def best(self) -> Program | None:
        return min((p for p in self.archive.values() if p.valid),
                   key=lambda p: (p.evaluation.fitness, p.id), default=None)

    @property
    def stop_reason(self) -> str | None:
        if self.evaluations_used >= self.config.evaluation_budget:
            return "evaluation_budget_exhausted"
        unevaluated = next((i for i, a in enumerate(reversed(self.attempts)) if a.evaluations), len(self.attempts))
        if unevaluated >= self.config.max_unevaluated:
            return "consecutive_unevaluated_limit"
        return None

    def _call(self, attempt: Attempt, prompt: str, repair_kind=None) -> str | None:
        record = {"stage": "repair" if repair_kind else "generate", "repair_kind": repair_kind}
        with getattr(self.model, "request_context", nullcontext)():
            started = time.monotonic()
            try:
                reply = self.model.complete(
                    prompt, temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens, timeout=self.config.model_timeout,
                )
                if isinstance(reply, str):
                    reply = ModelReply(reply)
                record.update(status="success", model=reply.model, finish_reason=reply.finish_reason,
                              usage=reply.usage, metadata=reply.metadata)
                content = reply.content
            except Exception as exc:
                record.update(status="error", error=f"{type(exc).__name__}: {exc}",
                              metadata=getattr(self.model, "last_metadata", {}))
                content = None
            record["seconds"] = time.monotonic() - started
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

    def _evaluate(self, attempt: Attempt, candidate: Candidate) -> None:
        with getattr(self.evaluator, "evaluation_context", nullcontext)():
            started = time.monotonic()
            evaluation = self.evaluator.evaluate(candidate.code)
            seconds = time.monotonic() - started
        if attempt.program_id is None:
            attempt.program_id = max(self.archive, default=0) + 1
        program = Program(
            attempt.program_id, candidate.code, candidate.design, attempt.operator,
            attempt.parent_id, attempt.reference_id, evaluation, analysis=candidate.analysis,
        )
        self.archive[program.id] = program
        attempt.evaluations.append({
            "id": self.evaluations_used + 1, "status": evaluation.status, "evaluation": asdict(evaluation),
            "seconds": seconds, "repair": attempt.repair_used, "program_id": program.id,
            "resources": getattr(self.evaluator, "last_dispatch", {}),
        })

    def step(self) -> Attempt | None:
        if self.stop_reason is not None:
            return None
        attempt_id = len(self.attempts) + 1
        rng = random.Random(f"{self.config.seed}:{attempt_id}")
        if sum(p.valid and p.operator == "Init" for p in self.archive.values()) < self.config.initial_size:
            attempt = Attempt(attempt_id, ROOT_ID, "Init")
            prompt = init_prompt(self.task, self.archive, self.config.failure_count)
        else:
            parent_id, operator, ref_id, explore_ids = allocate_step(
                self.tree, rng, self.allocation_config, self.topology_config
            )
            attempt = Attempt(
                attempt_id, parent_id, operator, ref_id,
                record={"explore_ids": list(explore_ids)} if explore_ids else {},
            )
            prompt = evolution_prompt(
                self.task, self.tree, parent_id, operator, ref_id,
                explore_ids=explore_ids, include_experiences=self.include_experiences,
            )
        self.attempts.append(attempt)
        response = self._call(attempt, prompt)
        if response is not None:
            candidate = self._parse(attempt, response)
            if candidate is None:
                attempt.repair_used = True
                prompt = repair_format_prompt(self.task, response, attempt.calls[-1]["format_failure"])
                response = self._call(attempt, prompt, "format")
                candidate = self._parse(attempt, response) if response is not None else None
            if candidate is not None:
                self._evaluate(attempt, candidate)
                program = self.archive[attempt.program_id]
                if (not program.valid and not attempt.repair_used
                        and self.evaluations_used < self.config.evaluation_budget):
                    attempt.repair_used = True
                    prompt = repair_program_prompt(self.task, program)
                    response = self._call(attempt, prompt, program.evaluation.status)
                    repaired = self._parse(attempt, response) if response is not None else None
                    if repaired is not None:
                        analysis = f"{program.analysis}\nRepair: {repaired.analysis}"
                        self._evaluate(attempt, replace(repaired, analysis=analysis))
        simplified = self._update_population(attempt)
        attempt.finished_at = time.time()
        program = self.archive.get(attempt.program_id)
        row = {"attempt": asdict(attempt), "program": asdict(program) if program else None}
        if simplified:
            row["tree_after"] = asdict(self.tree)
        self.store.append("attempts.jsonl", row)
        return attempt

    def run(self, *, on_attempt=None) -> SearchResult:
        while (attempt := self.step()) is not None:
            if on_attempt is not None:
                on_attempt(attempt)
        result = SearchResult(
            "finished" if self.best is not None else "failed", self.stop_reason, self.best,
            self.evaluations_used, self.model_calls, len(self.attempts),
            sum(p.valid for p in self.archive.values()),
        )
        self.store.write("result.json", asdict(result))
        if result.best is not None and self.store.path is not None:
            (self.store.path / "best.py").write_text(result.best.code)
        return result
