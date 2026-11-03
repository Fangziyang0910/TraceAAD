"""Matched-age development tasks with limited discrepancy continuation search."""

from dataclasses import asdict, replace
import hashlib
import json
import math
from uuid import uuid4

from .runtime.llm import generate, ModelCallError
from .runtime.scheduling import SchedulerError
from .delivery import DeliveryError, validate_source
from .program import Program
from .loop import SearchLoop
from .config import Config
from .models import DevelopmentState, DevelopmentTask, has_frontier, next_state
from .prompts import PROMPT_REVISION, PromptBuilderV18, PromptTooLarge, normalized_source, parse_code


class TraceAADV18(SearchLoop):
    METHOD = "traceaad_v18"
    MECHANISM_REVISION = "v18-repeat-guard-1"

    def __init__(self, *, config=None, initial_programs=(), **kwargs):
        super().__init__(config=config or Config(), **kwargs)
        saved = (self.facts.state or {}).get("v18", {})
        policy = json.loads(json.dumps(asdict(self.config)))
        if self.facts.state and (not saved or saved.get("config") != policy):
            raise ValueError("V18 resume requires the stored V18 configuration")
        if self.facts.state and saved.get("prompt_revision") != PROMPT_REVISION:
            raise ValueError("V18 prompt revision differs; resume with the original frozen source or start a new run")
        if self.facts.state and saved.get("mechanism_revision") != self.MECHANISM_REVISION:
            raise ValueError("V18 mechanism revision differs; resume with the original frozen source or start a new run")
        self.tasks = {int(k): DevelopmentTask(**v) for k, v in saved.get("tasks", {}).items()}
        self.states = {int(k): DevelopmentState(**v) for k, v in saved.get("states", {}).items()}
        self.pending = saved.get("pending")
        self.last_diverge = saved.get("last_diverge", 0)
        self.no_change_counts = saved.get("no_change_counts", {})
        self.stop_reason = saved.get("stop_reason")
        self.prompts = PromptBuilderV18(
            self.llm, kwargs.get("task"), self.training, self.programs, self.states, self.config
        )
        if initial_programs:
            if self.facts.state or len(initial_programs) > self.config.roots:
                raise ValueError("provide at most roots pre-evaluated programs, only for a new run")
            for program in initial_programs:
                validate_source(program.code, self.template)
                if not program.valid or program.fitness is None or not math.isfinite(program.fitness):
                    raise ValueError("provided roots must be valid, finite, pre-evaluated programs")
            for original in initial_programs:
                program = replace(
                    original, id=len(self.programs) + 1, generation=0, action="Provided",
                    parent_id=None, reference_id=None, repair_of=None, depth=0,
                    request_id=None, call_ids=[],
                    request={"request_type": "PROVIDED", "acquisition": original.metadata()},
                )
                self._normalize_result(program)
                self.programs[program.id] = program
                self._new_task(program)
                self._save(program=program, new=True, candidate=False)

    def _save(self, **record):
        snapshot = self.progress.snapshot()
        snapshot["budget_axis"] = "模型请求"
        snapshot["v18"] = {
            "config": asdict(self.config),
            "prompt_revision": PROMPT_REVISION,
            "mechanism_revision": self.MECHANISM_REVISION,
            "tasks": {k: asdict(v) for k, v in self.tasks.items()},
            "states": {k: asdict(v) for k, v in self.states.items()},
            "pending": self.pending,
            "last_diverge": self.last_diverge,
            "no_change_counts": dict(self.no_change_counts),
            "stop_reason": self.stop_reason,
        }
        self.facts.commit(snapshot, **record)

    def _best(self):
        return min(self.archive.values(), key=lambda p: (p.fitness, p.id), default=None)

    def _new_state(self, task_id, program_id, **values):
        state = DevelopmentState(len(self.states) + 1, task_id, program_id, **values)
        self.states[state.id] = state
        return state

    def _new_task(self, program):
        previous = next((p for p in self.archive.values() if p.id < program.id
                         and self._source_hash(p) == self._source_hash(program)), None)
        if previous is not None:
            program.request.update(root_admitted=False, duplicate_of=previous.id, task_id=None,
                                   origin_task_id=previous.request["origin_task_id"])
            return
        task_id = len(self.tasks) + 1
        state = self._new_state(task_id, program.id)
        self.tasks[task_id] = DevelopmentTask(
            task_id, program.id, state.id, program.id, [-program.fitness],
            depth_limit=self.config.initial_depth_limit, stack=[(state.id, 0)],
        )
        program.request.update(task_id=task_id, origin_task_id=task_id, root_admitted=True)

    @staticmethod
    def _source_hash(program):
        return program.request.get("source_hash") or hashlib.sha256(normalized_source(program.code).encode()).hexdigest()

    def _generation_key(self, action, parent=None):
        return action + (":" + self._source_hash(parent) if parent is not None else "")

    def _closed(self, action, parent=None):
        return self.no_change_counts.get(self._generation_key(action, parent), 0) >= self._generation_limit(action)

    def _generation_limit(self, action):
        # Before any usable root exists, retain the original initialization allowance.
        return self.config.init_attempt_limit if action == "Init" and not self.tasks else self.config.max_no_change

    def _blocked_programs(self):
        return {p.id for p in self.archive.values() if self._closed("Develop", p)}

    def _observe_generation(self, program, repeated):
        key = program.request["generation_key"]
        no_change = not program.valid or repeated
        count = self.no_change_counts.get(key, 0) + 1 if no_change else 0
        limit = self._generation_limit(program.action)
        self.no_change_counts[key] = count
        program.request.update(no_new_candidate=no_change, no_change_count=count,
                               no_change_limit=limit, generation_closed=count >= limit)

    def _statuses(self):
        best = self._best()
        if best is None:
            return []
        champion = self.tasks[best.request["origin_task_id"]]
        blocked = self._blocked_programs()
        champion_open = has_frontier(champion, self.states, self.config, blocked)
        changes = []
        for task in self.tasks.values():
            checkpoint = None
            if task.age >= self.config.first_checkpoint:
                checkpoint = self.config.first_checkpoint
                while checkpoint * self.config.checkpoint_multiplier <= task.age:
                    checkpoint *= self.config.checkpoint_multiplier
            own = task.best_curve[checkpoint] if checkpoint is not None else None
            reference = champion.best_curve[checkpoint] if checkpoint is not None and champion.age >= checkpoint else None
            if not has_frontier(task, self.states, self.config, blocked):
                status = "EXHAUSTED"
            elif task is champion:
                status = "CHAMPION"
            elif checkpoint is None:
                status = "YOUNG"
            elif reference is None:
                # A closed reference cannot later supply the missing age snapshot.
                status = "WAIT_REFERENCE" if champion_open else "ACTIVE"
            else:
                status = "ACTIVE" if own >= reference else "PAUSED"
            comparison = {
                "task_id": task.id, "status": status, "champion": champion.id,
                "checkpoint": checkpoint, "quality": own, "reference_quality": reference,
            }
            if comparison != task.comparison:
                changes.append(comparison)
            task.status, task.comparison = status, comparison
        return changes

    def _allocate(self):
        self._statuses()
        if not self.config.same_age:
            return min((t for t in self.tasks.values() if t.status != "EXHAUSTED"),
                       key=lambda t: (self.programs[t.best_program_id].fitness, t.id), default=None)
        return min(
            (t for t in self.tasks.values() if t.status in {"YOUNG", "CHAMPION", "ACTIVE"}),
            key=lambda t: (t.age, t.id), default=None,
        )

    def _divergence_due(self):
        best = self._best()
        return (
            self.config.diverge and best is not None
            and self.budget_used - max(best.generation, self.last_diverge) >= self.config.stagnation_window
            and self.config.budget - self.budget_used >= self.config.first_checkpoint + 1
        )

    def _expand(self, state, program):
        state.result_id = program.id
        parent = self.programs[state.program_id]
        retry = dict(retry_from=state.id, attempt_count=state.attempt_count + 1)
        duplicate = program.valid and normalized_source(program.code) == normalized_source(parent.code)
        program.request["duplicate_parent"] = duplicate
        repeated = program.valid and any(
            p.id < program.id and p.valid and p.request.get("generation_key") == program.request["generation_key"]
            and self._source_hash(p) == self._source_hash(program) for p in self.programs.values()
        )
        program.request["repeated_proposal"] = repeated
        self._observe_generation(program, duplicate or repeated)
        if program.request["no_new_candidate"]:
            if not program.request["generation_closed"]:
                # A failed proposal is not another continuation decision.
                state.preferred = self._new_state(state.task_id, parent.id, address=state.address, **retry).id
            return
        preferred_follow = program.fitness < parent.fitness
        for edge, follow in (("0", preferred_follow), ("1", not preferred_follow)):
            child = self._new_state(
                state.task_id, program.id if follow else parent.id,
                address=state.address + edge, **({} if follow else retry),
            )
            if edge == "0":
                state.preferred = child.id
            else:
                state.alternative = child.id

    def _normalize_result(self, program):
        raw = program.fitness
        if program.valid and (raw is None or not math.isfinite(raw)):
            program.status = "invalid_output"
            program.failure = {"kind": "invalid_output", "error": "evaluation returned no finite score"}
            program.fitness = None
        if program.valid:
            program.request["raw_score"] = raw
            program.fitness = raw if self.config.objective == "minimize" else -raw
            program.request["quality"] = -program.fitness
        else:
            program.request.update(raw_score=None, quality=None)
        program.request["source_hash"] = hashlib.sha256(normalized_source(program.code).encode()).hexdigest()

    def _complete(self):
        program = self.programs[self.pending["program_id"]]
        evaluations = program.evaluate(self.training, self.template) if program.status == "pending" else []
        self._normalize_result(program)
        task_id, state_id = program.request.get("task_id"), program.request.get("state_id")
        if state_id is not None:
            task = self.tasks[task_id]
            self._expand(self.states[state_id], program)
            task.age += 1
            if program.valid and -program.fitness > task.best_curve[-1]:
                task.best_program_id = program.id
            task.best_curve.append(-self.programs[task.best_program_id].fitness)
            program.request["task_age_after_completion"] = task.age
        elif program.valid:
            self._new_task(program)
            self._observe_generation(program, not program.request["root_admitted"])
            program.request["task_age_after_completion"] = 0 if program.request["root_admitted"] else None
        else:
            self._observe_generation(program, False)
        program.request["race_updates"] = self._statuses()
        best = self._best()
        program.request.update(
            result_status=program.status,
            global_best_after_completion=best.id if best else None,
            global_best_quality=-best.fitness if best else None,
        )
        self.pending = None
        self._save(program=program, candidate=True, evaluations=evaluations)

    def _record_response(self, request, details, error=None):
        calls = details["calls"]
        self.progress.record_calls(calls, request, self.config.output_tokens)
        code, failure = "", None
        if error:
            failure = {"kind": "api_error", "error": str(error)}
        else:
            try:
                code = parse_code(details.get("content"), details.get("finish_reason"))
            except DeliveryError as exc:
                failure = {"kind": "delivery_failed", "error": str(exc)}
        parent_id = request.get("parent_id")
        record = {k: v for k, v in request.items() if k != "prompt"}
        record.update(
            global_request_id=self.budget_used,
            usage=calls[0].get("usage") or None,
            model_latency=calls[0]["seconds"],
            model_and_sampling_config={"model": calls[0].get("model"), **calls[0].get("sampling", {})},
        )
        program = Program(
            id=len(self.programs) + 1, code=code, idea="", analysis=None,
            action=request["action"], generation=self.budget_used,
            parent_id=parent_id, reference_id=request.get("reference_id"),
            depth=self.programs[parent_id].depth + 1 if parent_id else 0,
            status=failure["kind"] if failure else "pending", failure=failure,
            request_id=calls[0]["request_id"], call_ids=[c["request_id"] for c in calls], request=record,
        )
        self.programs[program.id] = program
        self.pending = {"kind": "evaluating", "program_id": program.id}
        # Persist the response before execution so a scheduler interruption never repeats the model request.
        self._save(program=program, new=True, candidate=False, calls=calls)

    def _request(self, request):
        parent_id = request.get("parent_id") or request.get("reference_id")
        request["generation_key"] = self._generation_key(request["action"], self.programs.get(parent_id))
        self.progress.generations += 1
        if request["action"] == "Init":
            self.progress.init_attempts += 1
        if request["action"] == "Diverge" or request["request_type"] == "RESTART":
            self.last_diverge = self.budget_used
        self.pending = {"kind": "calling", "request": request}
        self._save()
        try:
            details = generate(self.llm, request["prompt"], max_tokens=self.config.output_tokens, retry_limit=0)
            error = None
        except ModelCallError as exc:
            details, error = {"calls": exc.calls}, str(exc)
        self._record_response(request, details, error)
        self._complete()

    def _recover_call(self):
        request = self.pending["request"]
        error = "interrupted request: provider completion and usage unknown; reserved request remains charged"
        call = {
            "transport_id": uuid4().hex, "prompt": request["prompt"], "response": "",
            "usage": {}, "finish_reason": None, "model": getattr(self.llm, "model", None),
            "seconds": None, "error": error, "sampling": {}, "interrupted": True,
        }
        self._record_response(request, {"calls": [call]}, error)
        self._complete()

    def step(self):
        """Complete a pending attempt or execute at most one new model request."""
        if self.pending is not None:
            return self._recover_call() if self.pending["kind"] == "calling" else self._complete()
        if self.phase not in {"roots", "search", "freeze"}:
            return
        if self.budget_used >= self.config.budget or (
            self.config.max_seconds is not None and self.progress.seconds >= self.config.max_seconds
        ):
            return self._freeze()
        if self.phase == "roots":
            if (len(self.tasks) < self.config.roots and self.progress.init_attempts < self.config.init_attempt_limit
                    and not self._closed("Init")):
                request = self.prompts.build("Init")
                request.update(request_type="INITIALIZE", task_id=None, origin_task_id=None, state_id=None)
                return self._request(request)
            self.progress.phase = "search" if self.tasks else "no_valid_root"
            self._save()
        if self.phase == "search":
            task, request = self._allocate(), None
            if self.config.diverge and (self._divergence_due() or task is None):
                reference = self._best()
                if not self._closed("Diverge", reference):
                    request = self.prompts.build("Diverge", reference)
                    request.update(request_type="DIVERGE", reference_id=reference.id,
                                   task_id=None, origin_task_id=None, state_id=None)
                elif not self._closed("Init"):
                    request = self.prompts.build("Init")
                    request.update(request_type="RESTART", task_id=None, origin_task_id=None, state_id=None)
                if request is not None:
                    request["entry_reason"] = "no_eligible_continuation" if task is None else "stagnation"
            if request is None:
                if task is None:
                    self.stop_reason = "no_eligible_generation"
                    return self._freeze()
                state = next_state(task, self.states, self.config, self._blocked_programs())
                assert state is not None
                parent = self.programs[state.program_id]
                request = self.prompts.build("Develop", parent, state)
                request.update(
                    request_type="DEVELOP", parent_id=parent.id, task_id=task.id, origin_task_id=task.id,
                    state_id=state.id, origin_state_id=state.id, decision_path=state.address,
                    depth_limit=task.depth_limit, state_depth=len(state.address),
                    discrepancy_count=state.address.count("1"), allocation=dict(task.comparison),
                )
            return self._request(request)
        if self.phase == "freeze":
            self._freeze()

    def _summary(self, status, error=None):
        summary = super()._summary(status, error)
        summary.update(
            budget_axis="模型请求", budget_basis="all model requests, including setup, failures and duplicates",
            v18={
                "mechanism_revision": self.MECHANISM_REVISION,
                "same_age": self.config.same_age, "lds": self.config.lds,
                "early_first": self.config.early_first,
                "tasks": {str(t.id): {k: v for k, v in asdict(t).items() if k != "stack"} for t in self.tasks.values()},
                "states": len(self.states),
                "setup_requests": sum(p.action in {"Init", "Diverge"} for p in self.programs.values()),
                "development_requests": sum(p.action == "Develop" for p in self.programs.values()),
                "alternative_requests": sum(p.request.get("discrepancy_count", 0) > 0 for p in self.programs.values()),
                "duplicate_requests": sum(p.request.get("duplicate_parent", False) for p in self.programs.values()),
                "rejected_duplicate_roots": sum(p.request.get("root_admitted") is False for p in self.programs.values()),
                "no_change_counts": dict(self.no_change_counts),
                "stop_reason": self.stop_reason,
                "pending": self.pending is not None,
                "token_totals": "usage when supplied; otherwise shared estimated accounting (per-call usage retained)",
            },
        )
        self.facts.save_summary(summary)
        return summary

    def run(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        try:
            while self.phase in {"roots", "search", "freeze"} or self.pending is not None:
                self.step()
        except SchedulerError as exc:
            return self._summary("service_unavailable", str(exc))
        except PromptTooLarge as exc:
            return self._summary("prompt_too_large", str(exc))
        except KeyboardInterrupt:
            self._save()
            self._summary("interrupted")
            raise
        return self._summary(self.phase)
