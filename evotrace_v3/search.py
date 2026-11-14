"""EvoTrace search: generate, evaluate, grow the tree, and consolidate experience."""

from contextlib import nullcontext
from copy import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import time
from typing import Any

from .config import Config
from .task import Task
from .evaluation import Evaluator
from .storage import Store
from .models import Attempt, Candidate, FormatError, ModelReply, Program, SearchResult

from .allocation import AllocationConfig, allocate_step
from .context import ContextConfig, EvoTracePromptBuilder
from .models import CANDIDATE_RESPONSE_FORMAT, Edge, Node, TreePopulation
from .reflection import reflect_and_apply
from .topology import TopologyConfig, plan_consolidation

def deserialize_tree(data: dict[str, Any]) -> TreePopulation:
    return TreePopulation(
        **{
            **data,
            "nodes": {int(i): Node(**node) for i, node in data["nodes"].items()},
            "edges": {int(i): Edge(**edge) for i, edge in data["edges"].items()},
            "used_crossover_pairs": [(parent, reference) for parent, reference in data["used_crossover_pairs"]],
        }
    )


class EvoTrace:
    """基于树状种群与反思简化的完整自动算法设计框架。"""

    method = "evotrace_v3"
    selection_rule = "rank_subtree_sampling"
    generation_protocol = "tree_population_multiscale"
    reference_rule = "quality_pool_max_ast_distance"
    mechanisms = ("tree_population", "dynamic_reflective_simplification", "individual_tree_allocation", "multiscale_context")

    def __init__(
        self,
        task: Task,
        model: Any,
        *,
        config: Config | None = None,
        topology_config: TopologyConfig | None = None,
        allocation_config: AllocationConfig | None = None,
        context_config: ContextConfig | None = None,
        reflection_model: Any = "same",
        reflection_temperature: float = 0.3,
        reflection_max_tokens: int = 1536,
        reference_code: bool = False,
        run_dir: str | Path | None = None,
        resume: bool = False,
    ) -> None:
        config = config or Config()
        if config.initial_size < 1:
            raise ValueError("EvoTrace requires initial_size >= 1")
        self.topology_config = topology_config or TopologyConfig()
        self.allocation_config = allocation_config or AllocationConfig()
        self.context_config = context_config or ContextConfig()
        if reference_code:
            self.context_config = replace(self.context_config, include_reference_code=True)

        self.reflection_temperature = reflection_temperature
        self.reflection_max_tokens = reflection_max_tokens

        gen_model = model
        if hasattr(model, "request_options"):
            gen_model = copy(model)
            gen_model.request_options = {
                **model.request_options,
                "response_format": CANDIDATE_RESPONSE_FORMAT,
            }

        self.reflection_model = model if reflection_model == "same" else reflection_model

        self.tree = TreePopulation()
        self.consolidations_count = 0
        self.reflection_calls_count = 0

        self.task = task
        self.model = gen_model
        self.config = config
        self.store = Store(run_dir)
        self.prompts = EvoTracePromptBuilder(task, config)
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
                self._restore_tree(attempt, row["tree_after"] if "consolidation" in attempt.record else None)
        else:
            self.store.write("run_config.json", {
                **self._run_settings(),
                "model": getattr(gen_model, "model", type(gen_model).__name__),
            })

    def _run_settings(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "config": asdict(self.config),
            "task": self.task.description(),
            "selection_rule": self.selection_rule,
            "generation_protocol": self.generation_protocol,
            "reference_rule": self.reference_rule,
            "mechanisms": list(self.mechanisms),
            "topology": asdict(self.topology_config),
            "allocation": asdict(self.allocation_config),
            "context": asdict(self.context_config),
            "reflection_enabled": self.reflection_model is not None,
            "reflection_temperature": self.reflection_temperature,
            "reflection_max_tokens": self.reflection_max_tokens,
        }

    def _validate_resume(self) -> None:
        saved = self.store.read("run_config.json")
        if saved is None:
            raise ValueError("Resume requires its original run_config.json")
        expected = json.loads(json.dumps(self._run_settings()))
        differences = [key for key, value in expected.items() if saved.get(key) != value]
        if differences:
            raise ValueError("Incompatible EvoTrace resume settings: " + ", ".join(differences))

    def _attach_evaluated_attempt(self, attempt: Attempt) -> None:
        if attempt.program_id is None:
            return
        prog = self.archive.get(attempt.program_id)
        if prog is None or prog.evaluation is None:
            return
        from .reflection import clean_prompt_text

        op = None if attempt.operator == "Init" else attempt.operator
        parent_id = self.tree.root_id if op is None else attempt.parent_id
        node = Node(
            id=prog.id,
            code=prog.code,
            design=clean_prompt_text(prog.design or ""),
            fitness=prog.evaluation.fitness if prog.valid else None,
            failure=prog.evaluation.failure,
        )
        edge = Edge(
            parent_id=parent_id,
            child_id=prog.id,
            operator=op,  # type: ignore[arg-type]
            reference_id=attempt.reference_id if op == "Crossover" else None,
        )
        self.tree.add(node, edge, evaluations=len(attempt.evaluations))

    def _restore_tree(self, attempt: Attempt, snapshot: dict | None) -> None:
        self._attach_evaluated_attempt(attempt)
        if snapshot is not None:
            self.tree = deserialize_tree(snapshot)
            self.consolidations_count += 1
            self.reflection_calls_count += sum(
                bool(record["prompt_chars"])
                for record in attempt.record["consolidation"]["reflection_calls"]
            )

    def _update_population(self, attempt: Attempt) -> None:
        self._attach_evaluated_attempt(attempt)
        if attempt.program_id not in self.tree.nodes:
            return
        initialized = sum(p.valid and p.operator == "Init" for p in self.archive.values())
        if initialized < self.config.initial_size:
            return
        plan = plan_consolidation(self.tree, self.topology_config)
        if not plan.path_contractions and not plan.branch_consolidations:
            return
        # Keep reflection behavior independent of context display so Experience-off ablation is clean;
        # callers wanting zero reflection calls pass reflection_model=None explicitly.
        new_tree, records = reflect_and_apply(
            self.tree,
            plan,
            model=self.reflection_model,
            task_description=self.task.task_description,
            score_definition=self.task.score_definition,
            function_template=self.task.function_template,
            temperature=self.reflection_temperature,
            max_tokens=self.reflection_max_tokens,
            timeout=min(self.config.model_timeout, 180.0),
        )
        self.tree = new_tree
        self.consolidations_count += 1
        llm_reflections = sum(1 for r in records if r.prompt_chars > 0)
        self.reflection_calls_count += llm_reflections

        # Record auxiliary reflection token usage for Reporter.publish!
        aux_entries = [
            {
                "role": "reflection",
                "kind": r.kind,
                "target_id": r.target_id,
                "status": "fallback" if r.used_fallback else "success",
                "requests": 1,
                "errors": int(r.used_fallback),
                "usage": dict(r.usage),
            }
            for r in records
            if r.prompt_chars > 0
        ]
        if aux_entries:
            attempt.record.setdefault("aux", []).extend(aux_entries)

        consolidation_summary = {
            "cycle": self.consolidations_count,
            "kept_ids": list(plan.kept_ids),
            "triggers": list(plan.triggers),
            "path_contractions": [asdict(c) for c in plan.path_contractions],
            "branch_consolidations": [asdict(b) for b in plan.branch_consolidations],
            "reflection_calls": [asdict(r) for r in records],
        }
        attempt.record["consolidation"] = consolidation_summary
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

    def _call(self, attempt: Attempt, prompt: str, repair_kind=None) -> str | None:
        """Send and record one generation or repair request."""
        record = {"stage": "repair" if repair_kind else "generate", "repair_kind": repair_kind}
        # The scheduler queue lies outside the request's own deadline.
        with getattr(self.model, "request_context", nullcontext)():
            started = time.monotonic()
            try:
                reply = self.model.complete(prompt, temperature=self.config.temperature,
                                       max_tokens=self.config.max_tokens,
                                            timeout=self.config.model_timeout)
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
        rng = random.Random(f"{self.config.seed}:{attempt_id}")
        initialized = sum(p.valid and p.operator == "Init" for p in self.archive.values())
        if initialized < self.config.initial_size:
            attempt = Attempt(attempt_id, 0, "Init")
            prompt = self.prompts.initialization(self.archive)
        else:
            decision = allocate_step(self.tree, rng, self.allocation_config)
            attempt = Attempt(attempt_id, decision.parent_id, decision.operator, decision.reference_id)
            attempt.record["allocation"] = {
                "rule": self.selection_rule,
                **asdict(decision),
                "parent_development_evaluations": self.tree.nodes[decision.parent_id].development_evaluations,
            }
            prompt, view = self.prompts.evolution_from_tree(
                self.tree, decision.parent_id, decision.operator,
                decision.reference_id, self.context_config,
                count_tokens=getattr(self.model, "count_prompt_tokens", None),
            )
            attempt.record["context"] = view
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
        self._update_population(attempt)
        attempt.finished_at = time.time()
        program = self.archive.get(attempt.program_id)
        row = {"attempt": asdict(attempt), "program": asdict(program) if program else None}
        if "consolidation" in attempt.record:
            row["tree_after"] = asdict(self.tree)
        self.store.append("attempts.jsonl", row)
        if "tree_after" in row:
            self.store.write("tree_latest.json", row["tree_after"])
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
        self.store.write("result.json", asdict(result))
        if result.best is not None and self.store.path is not None:
            (self.store.path / "best.py").write_text(result.best.code)
        return result
