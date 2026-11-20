"""EvoTrace V4 search: grow the tree population, allocate node budgets, and reflectively eliminate."""

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
    CANDIDATE_RESPONSE_FORMAT,
    Attempt,
    Candidate,
    Config,
    Edge,
    FormatError,
    ModelReply,
    Node,
    Program,
    ROOT_ID,
    SearchResult,
    TreePopulation,
)
from .reflection import EXPERIENCE_SCHEMA, problem_header, reflect_and_apply
from .task import Task
from .topology import TopologyConfig, apply_simplification, feasible_individuals, plan_simplification


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _append_json(path: Path, row: dict) -> None:
    with path.open("a") as output:
        output.write(json.dumps(row, ensure_ascii=False) + "\n")


def _complete_lines(path: Path) -> bytes:
    """文件中的完整行；被中断进程写了一半的末行先截掉，后续追加从新行开始。"""
    data = path.read_bytes() if path.exists() else b""
    end = data.rfind(b"\n") + 1
    if end < len(data):
        with path.open("r+b") as output:
            output.truncate(end)
    return data[:end]


def with_output_format(model: Any, schema: dict, *, role: bool = False) -> Any:
    if role and hasattr(model, "for_role"):
        return model.for_role(schema)
    if hasattr(model, "request_options"):
        model = copy(model)
        model.request_options = {**model.request_options, "response_format": schema}
    return model


class EvoTrace:
    """基于树状种群、反思式淘汰与在树节点预算分配的自动算法设计搜索。"""

    method = "evotrace_v4"
    selection_rule = "tree_population_rank_direct_sampling"
    generation_protocol = "tree_population_lineage_cards_analysis_code_design_v9"
    mechanisms = (
        "tree_population",
        "reflective_elimination",
        "tree_node_allocation",
        "traceable_lineage_context",
    )

    @classmethod
    def mechanisms_for(cls, reflection: bool = True) -> tuple[str, ...]:
        if reflection:
            return cls.mechanisms
        return tuple("tree_simplification" if m == "reflective_elimination" else m for m in cls.mechanisms)

    def __init__(
        self,
        task: Task,
        model: Any,
        *,
        config: Config | None = None,
        topology_config: TopologyConfig | None = None,
        allocation_config: AllocationConfig | None = None,
        include_experiences: bool = True,
        reflection: bool = True,
        run_dir: str | Path,
        resume: bool = False,
    ) -> None:
        self.task = task
        self.config = config or Config()
        self.topology_config = topology_config or TopologyConfig()
        self.allocation_config = allocation_config or AllocationConfig()
        self.reflection = reflection
        self.include_experiences = include_experiences and reflection
        self.mechanisms = self.mechanisms_for(reflection)

        self.tree = TreePopulation()
        self.model = with_output_format(model, CANDIDATE_RESPONSE_FORMAT)
        self.reflection_model = with_output_format(model, EXPERIENCE_SCHEMA, role=True) if self.reflection else None
        self.problem = problem_header(task.task_description, task.score_definition, task.function_template) if self.reflection else ""
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.evaluator = Evaluator(task)
        self.archive: dict[int, Program] = {}
        self.attempts: list[Attempt] = []
        if task.example_code is not None:
            self.archive[0] = Program(0, task.example_code, task.example_design, None, None, None, None)
        if resume:
            _complete_lines(self.run_dir / "calls.jsonl")
            for line in _complete_lines(self.run_dir / "attempts.jsonl").splitlines():
                row = json.loads(line)
                attempt = Attempt(**row["attempt"])
                self.attempts.append(attempt)
                if row["program"] is not None:
                    program = Program.from_dict(row["program"])
                    self.archive[program.id] = program
                self._attach_evaluated_attempt(attempt)
                if "tree_after" in row:
                    self.tree = TreePopulation.from_dict(row["tree_after"])
        else:
            _write_json(self.run_dir / "run_config.json", {
                "method": self.method,
                "config": asdict(self.config),
                "task": task.description(),
                "selection_rule": self.selection_rule,
                "generation_protocol": self.generation_protocol,
                "mechanisms": list(self.mechanisms),
                "topology": asdict(self.topology_config),
                "allocation": asdict(self.allocation_config),
                "reflection": self.reflection,
                "include_experiences": self.include_experiences,
                "model": getattr(self.model, "model", type(self.model).__name__),
            })
        self._resumed_at = len(self.attempts)

    @property
    def _initialized_count(self) -> int:
        return sum(p.valid and p.operator == "Init" for p in self.archive.values())

    def _attach_evaluated_attempt(self, attempt: Attempt) -> None:
        prog = self.archive.get(attempt.program_id) if attempt.program_id else None
        if prog is None or prog.evaluation is None:
            return
        op = None if attempt.operator == "Init" else attempt.operator
        self.tree.add(
            Node(
                prog.id,
                prog.code,
                prog.design or "",
                prog.evaluation.fitness if prog.valid else None,
                prog.evaluation.failure,
                analysis=prog.analysis or "",
            ),
            Edge(attempt.parent_id, prog.id, op, attempt.reference_id if op == "Crossover" else None),  # type: ignore[arg-type]
            evaluations=len(attempt.evaluations),
        )

    def _update_population(self, attempt: Attempt) -> bool:
        self._attach_evaluated_attempt(attempt)
        if attempt.program_id not in self.tree.nodes or self._initialized_count < self.config.initial_size:
            return False
        regions = plan_simplification(self.tree, self.topology_config)
        if not regions:
            return False
        if not self.reflection:
            self.tree = apply_simplification(self.tree, regions)
            return True
        self.tree, attempt.record["aux"] = reflect_and_apply(
            self.tree,
            regions,
            model=self.reflection_model,
            problem=self.problem,
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
        recent = self.attempts[self._resumed_at:]
        unevaluated = next((i for i, a in enumerate(reversed(recent)) if a.evaluations), len(recent))
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
        _append_json(self.run_dir / "calls.jsonl", {**record, "attempt_id": attempt.id,
                                                    "prompt": prompt, "response": content})
        return content

    def _parse(self, attempt: Attempt, response: str | None) -> Candidate | None:
        if response is None:
            return None
        try:
            return Candidate.parse(response)
        except FormatError as exc:
            attempt.calls[-1]["format_failure"] = str(exc)
            return None

    def _generate(self, attempt: Attempt, prompt: str, repair_kind: str | None = None) -> Candidate | None:
        """一次生成；回复不合格式且本次尝试尚未修复时，做一次格式修复。"""
        response = self._call(attempt, prompt, repair_kind)
        candidate = self._parse(attempt, response)
        if candidate is None and response is not None and not attempt.repair_used:
            attempt.repair_used = True
            feedback = attempt.calls[-1]["format_failure"]
            response = self._call(attempt, repair_format_prompt(self.task, response, feedback), "format")
            candidate = self._parse(attempt, response)
        return candidate

    def _budget_left(self) -> bool:
        return self.evaluations_used < self.config.evaluation_budget

    def _evaluate(self, attempt: Attempt, candidate: Candidate) -> Program:
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
        return program

    def step(self) -> Attempt | None:
        if self.stop_reason is not None:
            return None
        attempt_id = len(self.attempts) + 1
        rng = random.Random(f"{self.config.seed}:{attempt_id}")
        if self._initialized_count < self.config.initial_size or not feasible_individuals(self.tree):
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
        candidate = self._generate(attempt, prompt)
        if candidate is not None and self._budget_left():
            program = self._evaluate(attempt, candidate)
            if not program.valid and not attempt.repair_used and self._budget_left():
                attempt.repair_used = True
                repaired = self._generate(attempt, repair_program_prompt(self.task, program), program.evaluation.status)
                if repaired is not None and self._budget_left():
                    self._evaluate(attempt, replace(repaired, analysis=f"{program.analysis}\nRepair: {repaired.analysis}"))
        simplified = self._update_population(attempt)
        attempt.finished_at = time.time()
        program = self.archive.get(attempt.program_id)
        row = {"attempt": asdict(attempt), "program": asdict(program) if program else None}
        if simplified:
            row["tree_after"] = asdict(self.tree)
        _append_json(self.run_dir / "attempts.jsonl", row)
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
        _write_json(self.run_dir / "result.json", asdict(result))
        return result
