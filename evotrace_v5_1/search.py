"""EvoTrace V5.1 主循环：有界谱系森林、就地升级、“第二次机会”反思式淘汰与 Sigmoid 强竞争。"""

import json
import os
import random
import time
from contextlib import nullcontext
from copy import copy
from dataclasses import asdict
from pathlib import Path

from .allocation import choose_step
from .context import evolution_prompt, init_prompt, repair_format_prompt, repair_program_prompt
from .evaluation import Evaluator
from .models import (
    CANDIDATE_RESPONSE_FORMAT,
    ROOT_ID,
    Attempt,
    Candidate,
    Config,
    Evaluation,
    FormatError,
    ModelReply,
    Program,
    SearchResult,
    TreePopulation,
)
from .reflection import EXPERIENCE_SCHEMA, reflect_reduction
from .topology import apply_reduction, duplicate_of, next_reduction


def _write(path: Path, value) -> None:
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8") as out:
        json.dump(value, out, ensure_ascii=False, allow_nan=False)
        out.flush()
        os.fsync(out.fileno())
    temp.replace(path)


def _append(path: Path, value) -> None:
    data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode() + b"\n"
    with path.open("ab") as out:
        out.write(data)
        out.flush()
        os.fsync(out.fileno())


def _rows(path: Path):
    if path.is_file():
        with path.open("r+b") as stream:
            for line in iter(stream.readline, b""):
                if not line.endswith(b"\n"):
                    stream.truncate(stream.tell() - len(line))
                    break
                yield json.loads(line)


def _model_for(model, schema, *, auxiliary: bool = False):
    if auxiliary and hasattr(model, "for_role"):
        return model.for_role(schema)
    if hasattr(model, "request_options"):
        model = copy(model)
        model.request_options = {**model.request_options, "response_format": schema}
    return model


class EvoTrace:
    method = "evotrace_v5_1"
    selection_rule = "median_mad_sigmoid_direct_discount_v1"
    generation_protocol = "second_chance_lineage_v1"
    mechanisms = (
        "bounded_lineage_forest",
        "second_chance_reflective_elimination",
        "median_mad_sigmoid_allocation",
        "isolated_operator_context",
    )

    @classmethod
    def mechanisms_for(cls, reflection: bool = True) -> tuple[str, ...]:
        return tuple(
            "bounded_tree_simplification"
            if not reflection and m == "second_chance_reflective_elimination"
            else m
            for m in cls.mechanisms
        )

    def __init__(
        self,
        task,
        model,
        *,
        config=None,
        run_dir,
        resume: bool = False,
        include_experiences: bool = True,
        reflection: bool = True,
    ):
        self.task, self.config = task, config or Config()
        self.reflection = reflection
        self.include_experiences = include_experiences and reflection
        self.mechanisms = self.mechanisms_for(reflection)
        self.model = _model_for(model, CANDIDATE_RESPONSE_FORMAT)
        self.reflection_model = (
            _model_for(model, EXPERIENCE_SCHEMA, auxiliary=True) if reflection else None
        )
        self.evaluator = Evaluator(task)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.journal = self.run_dir / "attempts.jsonl"
        self.pending = self.run_dir / "pending.json"
        self.archive: dict[int, Program] = (
            {0: Program(0, task.example_code, task.example_design, None, None, None, None)}
            if task.example_code is not None
            else {}
        )
        self.tree = TreePopulation()
        self.attempts: list[Attempt] = []
        self._used = self._unevaluated = 0
        self._init_run(
            {
                "method": self.method,
                "config": asdict(self.config),
                "task": task.description(),
                "selection_rule": self.selection_rule,
                "generation_protocol": self.generation_protocol,
                "mechanisms": self.mechanisms,
                "reflection": reflection,
                "include_experiences": self.include_experiences,
                "model": getattr(model, "model", type(model).__name__),
            },
            resume=resume,
        )

    def _init_run(self, descriptor: dict, *, resume: bool) -> None:
        config_path = self.run_dir / "run_config.json"
        if not resume:
            if any(p.exists() for p in (config_path, self.journal, self.pending)):
                raise FileExistsError(f"Run already exists: {self.run_dir}")
            _write(config_path, descriptor)
            return

        original = json.loads(config_path.read_text())
        current = json.loads(json.dumps(descriptor))
        for key in (
            "method",
            "task",
            "reflection",
            "include_experiences",
            "generation_protocol",
            "model",
        ):
            if original[key] != current[key]:
                raise ValueError(f"Resume requires the original {key}")
        for key, value in original["config"].items():
            if key != "evaluation_budget" and value != current["config"][key]:
                raise ValueError(f"Resume requires the original {key}")

        self._restore()
        reserved = (
            len(json.loads(self.pending.read_text())["attempt"]["evaluations"])
            if self.pending.exists()
            else 0
        )
        if self.config.evaluation_budget < max(
            self._used + reserved, original["config"]["evaluation_budget"]
        ):
            raise ValueError("Resume can only extend the evaluation budget")
        if self.config.evaluation_budget != original["config"]["evaluation_budget"]:
            original["config"]["evaluation_budget"] = self.config.evaluation_budget
            _write(config_path, original)

    def _load_programs(self, row: dict, parent_id: int | None = None) -> None:
        for raw in [*row.get("failed_programs", []), row["program"]]:
            if raw:
                p = Program.from_dict(raw)
                self.archive[p.id] = p
                if parent_id is not None and p.valid and p.duplicate_of is None:
                    self.tree.parents[p.id] = parent_id
                    self.tree.direct_evals.setdefault(p.id, 0)

    def _restore(self) -> None:
        for _ in _rows(self.run_dir / "calls.jsonl"):
            pass
        for row in _rows(self.journal):
            attempt = Attempt(**row["attempt"])
            self._load_programs(row, parent_id=attempt.parent_id)
            if attempt.parent_id != ROOT_ID and attempt.parent_id in self.tree.parents:
                self.tree.direct_evals[attempt.parent_id] = self.tree.direct_evals.get(
                    attempt.parent_id, 0
                ) + len(attempt.evaluations)
            if "tree_after" in row:
                self.tree = TreePopulation.from_dict(row["tree_after"])
            self.attempts.append(attempt)
            self._used += len(attempt.evaluations)
            self._unevaluated = 0 if attempt.evaluations else self._unevaluated + 1
        if (
            self.pending.exists()
            and json.loads(self.pending.read_text())["attempt"]["id"] <= len(self.attempts)
        ):
            self.pending.unlink()

    def _resume_pending(self) -> tuple[Attempt, Program]:
        row = json.loads(self.pending.read_text())
        attempt = Attempt(**row["attempt"])
        if attempt.id != len(self.attempts) + 1:
            raise ValueError("Pending attempt is out of order")
        if self._used + len(attempt.evaluations) > self.config.evaluation_budget:
            raise ValueError("Budget is smaller than the reserved evaluations")
        self._load_programs(row)
        return attempt, self.archive[attempt.program_id]

    @property
    def evaluations_used(self) -> int:
        return self._used

    @property
    def model_calls(self) -> int:
        return sum(len(a.calls) for a in self.attempts)

    @property
    def best(self) -> Program | None:
        return min(
            (p for p in self.archive.values() if p.valid),
            key=lambda p: (p.evaluation.fitness, p.id),
            default=None,
        )

    @property
    def stop_reason(self) -> str | None:
        if self._used >= self.config.evaluation_budget:
            return "evaluation_budget_exhausted"
        if self._unevaluated >= self.config.max_unevaluated:
            return "consecutive_unevaluated_limit"
        return None

    def _call(self, attempt: Attempt, prompt: str, repair_kind: str | None = None):
        log = {"stage": "repair" if repair_kind else "generate", "repair_kind": repair_kind}
        with getattr(self.model, "request_context", nullcontext)():
            start = time.monotonic()
            try:
                reply = self.model.complete(
                    prompt,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.model_timeout,
                )
                if isinstance(reply, str):
                    reply = ModelReply(reply)
                log.update(
                    status="success",
                    model=reply.model,
                    finish_reason=reply.finish_reason,
                    usage=reply.usage,
                    metadata=reply.metadata,
                )
                content = reply.content
            except Exception as exc:  # noqa: BLE001 - persist backend failures
                log.update(
                    status="error",
                    error=f"{type(exc).__name__}: {exc}",
                    metadata=getattr(self.model, "last_metadata", {}),
                )
                content = None
            log["seconds"] = time.monotonic() - start
        attempt.calls.append(log)
        _append(
            self.run_dir / "calls.jsonl",
            {**log, "attempt_id": attempt.id, "prompt": prompt, "response": content},
        )
        return content

    def _generate(self, attempt: Attempt, prompt: str, repair_kind: str | None = None):
        response = self._call(attempt, prompt, repair_kind)
        if response is None:
            return None
        try:
            return Candidate.parse(response)
        except FormatError as exc:
            attempt.calls[-1]["format_failure"] = str(exc)
            if attempt.repair_used:
                return None
            attempt.repair_used = True
            return self._generate(
                attempt, repair_format_prompt(self.task, response, str(exc)), "format"
            )

    def _new_attempt(self) -> tuple[Attempt, str]:
        i = len(self.attempts) + 1
        rng = random.Random(f"{self.config.seed}:{i}")
        initial = sum(
            p.valid and p.operator == "Init" and p.duplicate_of is None
            for p in self.archive.values()
        )
        if initial < self.config.initial_size or not self.tree.parents:
            return Attempt(i, ROOT_ID, "Init"), init_prompt(
                self.task, self.archive, self.config.failure_count
            )

        parent, operator, ref, leaders, trace = choose_step(
            self.tree,
            self.archive,
            rng,
            selection_lambda=self.config.selection_lambda,
            explore_references=self.config.explore_references,
        )
        show_memory = self.include_experiences
        if operator == "Refine" and show_memory and self.config.reflection_mask_prob > 0.0:
            show_memory = rng.random() >= self.config.reflection_mask_prob

        repaired_failed = frozenset(
            r["program_id"]
            for prev in self.attempts
            if prev.repair_used
            and prev.program_id is not None
            and (final := self.archive.get(prev.program_id)) is not None
            and final.valid
            for r in prev.evaluations
            if r["program_id"] != prev.program_id
        )
        prompt, context_info = evolution_prompt(
            self.task,
            self.tree,
            self.archive,
            parent,
            operator,
            ref,
            explore_ids=leaders,
            show_memory=show_memory,
            cross_lineage=bool(trace.get("cross_lineage", True)),
            exclude_failures=repaired_failed,
        )
        attempt = Attempt(
            i,
            parent,
            operator,
            ref,
            record={"allocation": trace, "explore_ids": list(leaders), "context": context_info},
        )
        return attempt, prompt

    def _row(self, attempt: Attempt) -> dict:
        ids = dict.fromkeys(r["program_id"] for r in attempt.evaluations)
        return {
            "attempt": asdict(attempt),
            "failed_programs": [asdict(self.archive[i]) for i in ids if i != attempt.program_id],
            "program": asdict(self.archive[attempt.program_id]) if attempt.program_id else None,
        }

    def _evaluate_candidate(self, attempt: Attempt, candidate: Candidate) -> Program:
        eid = self._used + len(attempt.evaluations) + 1
        if eid > self.config.evaluation_budget:
            raise RuntimeError("Evaluation budget exhausted")
        pending = Evaluation("error", failure="Interrupted evaluation; outcome unknown")
        program = Program(
            eid,
            candidate.code,
            candidate.design,
            attempt.operator,
            attempt.parent_id,
            attempt.reference_id,
            pending,
            candidate.analysis,
        )
        attempt.program_id = eid
        self.archive[eid] = program
        record = {
            "id": eid,
            "kind": "generation",
            "program_id": eid,
            "repair": attempt.repair_used,
            "status": "error",
            "interrupted": True,
            "evaluation": asdict(pending),
        }
        attempt.evaluations.append(record)
        with getattr(self.evaluator, "evaluation_context", nullcontext)():
            _write(self.pending, self._row(attempt))
            start = time.monotonic()
            evaluation = self.evaluator.evaluate(program.code)
            record.update(
                status=evaluation.status,
                seconds=time.monotonic() - start,
                resources=getattr(self.evaluator, "last_dispatch", {}),
                evaluation=asdict(evaluation),
            )
            record.pop("interrupted")
            program.evaluation = evaluation
            _write(self.pending, self._row(attempt))
        return program

    def _simplify(self) -> tuple[list[dict], list[dict]]:
        aux: list[dict] = []
        reductions: list[dict] = []
        while (
            event := next_reduction(self.tree, self.archive, self.config.population_capacity)
        ) is not None:
            note, report = reflect_reduction(
                self.task,
                self.tree,
                self.archive,
                event,
                self.reflection_model,
                min(180.0, self.config.model_timeout),
            )
            apply_reduction(self.tree, event, note)
            reductions.append({**asdict(event), "removed": list(event.removed)})
            if report is not None:
                aux.append(report)
        return aux, reductions

    def step(self) -> Attempt | None:
        if self.stop_reason is not None and not self.pending.exists():
            return None
        if self.pending.exists():
            attempt, program = self._resume_pending()
        else:
            attempt, prompt = self._new_attempt()
            candidate = self._generate(attempt, prompt)
            program = self._evaluate_candidate(attempt, candidate) if candidate else None

        if (
            program
            and not program.valid
            and not attempt.repair_used
            and not attempt.evaluations[-1].get("interrupted")
            and self._used + len(attempt.evaluations) < self.config.evaluation_budget
        ):
            attempt.repair_used = True
            candidate = self._generate(
                attempt, repair_program_prompt(self.task, program), program.evaluation.status
            )
            if candidate:
                program = self._evaluate_candidate(attempt, candidate)

        if attempt.parent_id != ROOT_ID and attempt.parent_id in self.tree.parents:
            self.tree.direct_evals[attempt.parent_id] = self.tree.direct_evals.get(
                attempt.parent_id, 0
            ) + len(attempt.evaluations)

        inserted = False
        if program and program.valid:
            program.duplicate_of = duplicate_of(
                program, (self.archive[i] for i in sorted(self.tree.parents))
            )
            if program.duplicate_of is None:
                self.tree.parents[program.id] = attempt.parent_id
                self.tree.direct_evals.setdefault(program.id, 0)
                inserted = True

        reductions: list[dict] = []
        if inserted:
            aux, reductions = self._simplify()
            if reductions or aux:
                attempt.record.update(aux=aux, reductions=reductions)

        attempt.finished_at = time.time()
        row = self._row(attempt)
        if reductions:
            row["tree_after"] = self.tree.snapshot()
        _append(self.journal, row)
        self.attempts.append(attempt)
        self._used += len(attempt.evaluations)
        self._unevaluated = 0 if attempt.evaluations else self._unevaluated + 1
        self.pending.unlink(missing_ok=True)
        return attempt

    def run(self, *, on_attempt=None) -> SearchResult:
        while (attempt := self.step()) is not None:
            if on_attempt is not None:
                on_attempt(attempt)
        result = SearchResult(
            "finished" if self.best else "failed",
            self.stop_reason,
            self.best,
            self._used,
            self.model_calls,
            len(self.attempts),
            sum(p.valid for p in self.archive.values()),
        )
        _write(self.run_dir / "result.json", asdict(result))
        return result
