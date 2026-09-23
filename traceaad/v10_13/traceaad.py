"""TraceAAD V10.13: simple evolutionary search with operator-specific prompts."""

from __future__ import annotations

import json
import math
import random
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from core import SecureEvaluator
from . import parsing
from .prompts import PromptBuilder
from .selection import (
    OPERATORS,
    OPERATOR_PROBABILITIES,
    QUALITY_ESS_TARGET, PIVOT_UNIFORM_PROBABILITY, DONOR_UNIFORM_PROBABILITY,
    select_donor,
    sample_parent,
)
from .storage import RunStorage, atomic_json, digest, truncate_torn_tail
from .tree import Node, SearchTree

REPAIRABLE_FAILURES = {
    "exec_error", "runtime_error", "timeout", "invalid_result", "nonfinite_fitness",
}


def _timestamp():
    return datetime.now().isoformat(timespec="seconds")


def _restore_rng(rng, state):
    rng.setstate((state[0], tuple(state[1]), state[2]))


@dataclass
class Candidate:
    """One scheduled generation; repair reuses its original search choice."""

    candidate_id: int
    prompt: str
    prompt_tokens: int
    prompt_hash: str
    requested_operator: str
    operator: str
    parent_id: int | None
    donor_id: int | None
    parent_fitness: float | None
    donor_fitness: float | None
    reference_ids: list
    selection: dict
    parent_selected: bool
    best_before: float | None
    history_ids: list[int] = field(default_factory=list)
    llm_attempts: int = 0
    repair_of: int | None = None


class TraceAADV1013:
    METHOD = "v1013"
    PROMPT_POLICY = "evolutionary_operator_context_v10_13_final"

    def __init__(self, *, evaluation, llm, run_dir, budget=1000, n_roots=8,
                 history_depth=3, output_tokens=8192, max_input_tokens=24320,
                 seed=0):
        if budget < n_roots or n_roots < 1 or history_depth < 0:
            raise ValueError("invalid budget, root, or history settings")
        self.evaluation, self.llm = evaluation, llm
        self.run_dir = Path(run_dir)
        self.budget, self.n_roots, self.history_depth = budget, n_roots, history_depth
        if output_tokens < 1 or max_input_tokens < 1 or output_tokens + max_input_tokens > 32768 - 256:
            raise ValueError("input + output must fit the 32768-token context with a 256-token margin")
        self.output_tokens, self.max_input_tokens = output_tokens, max_input_tokens
        self.evaluator = SecureEvaluator(evaluation)
        self._template_program = evaluation.template_program
        self._parse_interface, target_stub = parsing.template_target(self._template_program)
        self.task_contract = (
            "# Task\n\n" + evaluation.task_description.strip() +
            "\n\nImplement the target function described below. Its body and any supporting "
            "code are your design space.\n\nTarget function:\n```python\n" +
            target_stub + "\n```"
        )
        self.tree = SearchTree()
        self.rng = random.Random(seed)
        self.started_at = _timestamp()
        self.completed_candidates = 0
        self.evaluations_used = 0
        self._invalid_streak = 0
        self.storage = RunStorage(self.run_dir)
        self.mechanism = {
            "method": self.METHOD,
            "budget": budget,
            "n_roots": n_roots,
            "prompt_policy": self.PROMPT_POLICY,
            "parser_protocol": "target_function_anchored_idea_code_v10_13",
            "history_depth": history_depth,
            "output_tokens": output_tokens,
            "max_input_tokens": max_input_tokens,
            "operator_probabilities": OPERATOR_PROBABILITIES,
            "quality_ess_target": QUALITY_ESS_TARGET,
            "pivot_uniform_probability": PIVOT_UNIFORM_PROBABILITY,
            "donor_uniform_probability": DONOR_UNIFORM_PROBABILITY,
            "task_contract_hash": digest(self.task_contract),
            "template_hash": digest(self._template_program),
            "seed": seed,
            "context_mapping": {"Refine": "formation", "Tune": "formation",
                                "Pivot": "single_reference", "Fuse": "single_donor"},
            "llm": {name: getattr(llm, name, None) for name in
                    ("model", "base_url", "temperature", "top_p", "enable_thinking")},
        }
        self.prompts = PromptBuilder(
            llm, self.task_contract,
            max_tokens=max_input_tokens,
            history_depth=history_depth,
            lookup=self.tree.nodes.get,
            all_nodes=self.tree.all_nodes,
        )

    def _candidate(self, prompt, **fields):
        prompt_tokens = self.prompts.count(prompt)
        if prompt_tokens > self.max_input_tokens:
            raise ValueError("candidate prompt exceeds the configured input token budget")
        return Candidate(
            candidate_id=self.completed_candidates + 1,
            best_before=self.tree.best().fitness if self.tree.nodes else None,
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            prompt_hash=digest(prompt),
            **fields,
        )

    def _schedule_repair(self, previous):
        prompt = parsing.build_repair_prompt(
            self.task_contract,
            self.storage.failed_response(previous["candidate_id"]),
            previous,
        )
        return self._candidate(
            prompt,
            repair_of=previous["candidate_id"],
            requested_operator=previous.get("requested_operator", "Init"),
            operator=previous.get("operator", "Init"),
            parent_id=previous.get("parent_id"),
            donor_id=previous.get("donor_id"),
            parent_fitness=previous.get("parent_fitness"),
            donor_fitness=previous.get("donor_fitness"),
            reference_ids=previous.get("reference_ids", []),
            history_ids=previous.get("history_ids", []),
            selection={},
            parent_selected=False,
        )

    def _schedule_candidate(self):
        previous = self.storage.last_event
        if previous and previous["candidate_id"] != self.completed_candidates:
            previous = None
        if previous and previous.get("status") == "eval_failed" and previous.get("reason") not in REPAIRABLE_FAILURES:
            raise RuntimeError(f"evaluation infrastructure failed: {previous.get('reason')}")
        if previous and (previous.get("status") == "invalid_output" or
                         previous.get("reason") in REPAIRABLE_FAILURES) and not previous.get("repair_of"):
            return self._schedule_repair(previous)

        if len(self.tree.roots) < self.n_roots:
            context = self.prompts.build_initial_context()
            return self._candidate(
                context.prompt,
                requested_operator="Init",
                operator="Init",
                parent_id=None,
                donor_id=None,
                parent_fitness=None,
                donor_fitness=None,
                reference_ids=context.reference_ids,
                selection={"fallback_reason": context.fallback_reason} if context.fallback_reason else {},
                parent_selected=False,
            )

        requested_operator = self.rng.choices(
            OPERATORS, weights=OPERATOR_PROBABILITIES.values(),
        )[0]
        parent, selection = sample_parent(
            self.tree.all_nodes(), requested_operator, self.rng,
        )
        selection["parent_count_after"] = parent.attempts + 1
        parent.attempts += 1

        operator = requested_operator
        donor = None
        if requested_operator in ("Pivot", "Fuse"):
            donor, donor_selection = select_donor(self.tree.all_nodes(), parent, self.rng)
            selection.update(donor_selection)
            if donor is None and requested_operator == "Fuse":
                operator = "Refine"
                selection["fallback_reason"] = "donor_unavailable"

        context = self.prompts.build_development(parent, operator, donor)
        if context.fallback_reason:
            selection["fallback_reason"] = context.fallback_reason
        reference = context.reference_program
        return self._candidate(
            context.prompt,
            requested_operator=requested_operator,
            operator=context.operator,
            parent_id=parent.id,
            donor_id=reference.id if reference else None,
            parent_fitness=parent.fitness,
            donor_fitness=reference.fitness if reference else None,
            reference_ids=context.reference_ids,
            history_ids=context.history_ids,
            selection=selection,
            parent_selected=True,
        )

    def _generate(self, candidate):
        candidate.llm_attempts += 1
        started = time.time()
        record = {
            "ts": _timestamp(),
            "call_id": f"{candidate.candidate_id}:{candidate.llm_attempts}",
            "candidate_id": candidate.candidate_id,
            "operator": candidate.operator,
            "prompt_tokens": candidate.prompt_tokens,
            "prompt_hash": candidate.prompt_hash,
            "prompt": candidate.prompt,
        }
        try:
            details = self.llm.draw_sample_with_details(
                candidate.prompt, max_tokens=self.output_tokens,
            )
        except Exception:
            record.update(seconds=time.time() - started, error=traceback.format_exc())
            self.storage.record_call(record)
            raise
        record.update(
            response=details["content"],
            seconds=time.time() - started,
            finish_reason=details.get("finish_reason") or "unknown",
            usage=details.get("usage"),
            model=details.get("model"),
            response_id=details.get("response_id"),
            stage="repair" if candidate.repair_of else "generation",
        )
        self.storage.record_call(record)
        return record

    def _read_fitness(self, result):
        reason, error_type, error, trace = (
            result.failure_kind, result.error_type, result.error, result.traceback,
        )
        if result.result is None:
            return None, reason, error_type, error, trace
        try:
            value = float(result.result)
        except (TypeError, ValueError, OverflowError) as exc:
            return None, "invalid_result", type(exc).__name__, str(exc), trace
        if math.isfinite(value):
            return value, reason, error_type, error, trace
        return None, "nonfinite_fitness", "NonfiniteFitness", "nonfinite fitness", trace

    def _evaluate_and_add(self, parsed, candidate):
        if parsed is None:
            return None, None
        evaluation_id = self.evaluations_used + 1
        started = time.time()
        try:
            fitness, reason, error_type, error, trace = self._read_fitness(
                self.evaluator.evaluate_program_with_details(parsed.program_code),
            )
        except Exception as exc:
            fitness, reason, error_type, error, trace = (
                None, "evaluation_error", type(exc).__name__, str(exc), traceback.format_exc(),
            )
        outcome = {
            "candidate_id": candidate.candidate_id,
            "evaluation_id": evaluation_id,
            "fitness": fitness,
            "reason": reason,
            "error_type": error_type,
            "error": error,
            "traceback": trace,
            "eval_seconds": time.time() - started,
            "repair_of": candidate.repair_of,
        }
        self.evaluations_used = evaluation_id
        if fitness is None:
            return outcome, None

        node = self.tree.add(
            code=parsed.program_code,
            idea=parsed.idea,
            fitness=fitness,
            evaluation_id=evaluation_id,
            parent_id=candidate.parent_id,
            operator=candidate.operator,
            donor_id=candidate.donor_id,
        )
        self.storage.record_node(node)
        return outcome, node

    def _settle(self, candidate, completion, parse_error, outcome, node):
        if parse_error is not None:
            self._invalid_streak += 1
            status, reason = "invalid_output", parse_error
        else:
            self._invalid_streak = 0
            status, reason = (
                ("ok", None) if outcome["fitness"] is not None
                else ("eval_failed", outcome["reason"])
            )
        parent_improved = bool(
            node is not None and candidate.parent_id is not None and
            node.fitness > candidate.parent_fitness
        )
        frontier_improved = bool(
            node is not None and candidate.best_before is not None and
            node.fitness > candidate.best_before
        )
        record = {
            "ts": _timestamp(),
            "candidate_id": candidate.candidate_id,
            "requested_operator": candidate.requested_operator,
            "operator": candidate.operator,
            "repair_of": candidate.repair_of,
            "parent_id": candidate.parent_id,
            "donor_id": candidate.donor_id,
            "parent_fitness": candidate.parent_fitness,
            "donor_fitness": candidate.donor_fitness,
            "reference_ids": candidate.reference_ids,
            "history_ids": candidate.history_ids,
            "selection": candidate.selection,
            "parent_selected": candidate.parent_selected,
            "prompt_tokens": candidate.prompt_tokens,
            "prompt_hash": candidate.prompt_hash,
            "status": status,
            "reason": reason,
            "budget_used": self.evaluations_used,
            "evaluation_id": outcome["evaluation_id"] if outcome else None,
            "eval_seconds": outcome.get("eval_seconds") if outcome else None,
            "llm_seconds": completion["seconds"],
            "node_id": node.id if node else None,
            "fitness": node.fitness if node else None,
            "parent_improved": parent_improved if node else None,
            "frontier_improved": frontier_improved if node else None,
        }
        if outcome is not None:
            record.update(
                error_type=outcome.get("error_type"),
                error=outcome.get("error"),
                traceback=outcome.get("traceback"),
                best_fitness=self.tree.best().fitness if self.tree.nodes else None,
            )
        self.storage.record_event(record)
        self.completed_candidates = candidate.candidate_id
        self._save_checkpoint()
        if self._invalid_streak >= 50:
            raise RuntimeError("50 consecutive generations produced no valid output")

    def _run_candidate(self):
        candidate = self._schedule_candidate()
        completion = self._generate(candidate)
        parsed, parse_error = parsing.parse_candidate(
            completion["response"], completion["finish_reason"],
            self._parse_interface, self._template_program,
        )
        outcome, node = self._evaluate_and_add(parsed, candidate)
        self._settle(candidate, completion, parse_error, outcome, node)

    def _save_checkpoint(self):
        atomic_json(self.storage.state_path, {
            "mechanism": self.mechanism,
            "started_at": self.started_at,
            "rng_state": list(self.rng.getstate()),
            "nodes": [asdict(node) for node in self.tree.all_nodes()],
            "parent_selection_counts": {
                str(node.id): node.attempts for node in self.tree.all_nodes()
                if node.attempts
            },
            "budget_used": self.evaluations_used,
            "completed_candidates": self.completed_candidates,
            "invalid_streak": self._invalid_streak,
            "last_event": self.storage.last_event,
        })

    def _resume_checkpoint(self):
        state = json.loads(self.storage.state_path.read_text())
        if state.get("mechanism") != self.mechanism:
            raise ValueError("checkpoint configuration differs from V10.13")
        self.started_at = state["started_at"]
        _restore_rng(self.rng, state["rng_state"])
        for entry in state["nodes"]:
            self.tree.add_raw(Node(**entry))
        self.evaluations_used = state["budget_used"]
        self.completed_candidates = state["completed_candidates"]
        self._invalid_streak = state["invalid_streak"]
        self.storage.last_event = state.get("last_event")

    def _write_summary(self, status, error=None):
        best = self.tree.best() if self.tree.nodes else None
        payload = {
            "status": status,
            "method": self.METHOD,
            "started_at": self.started_at,
            "finished_at": _timestamp(),
            "budget": self.budget,
            "budget_used": self.evaluations_used,
            "num_nodes": len(self.tree.nodes),
            "num_roots": len(self.tree.roots),
            "parent_attempts": self.tree.parent_selections,
            "best": None if best is None else {
                "node_id": best.id,
                "fitness": best.fitness,
                "idea": best.idea,
                "code": best.code,
                "evaluation_id": best.evaluation_id,
                "operator": best.operator,
                "parent_id": best.parent_id,
                "donor_id": best.donor_id,
            },
        }
        if error:
            payload["error"] = error
        atomic_json(self.storage.summary_path, payload)

    def run(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for journal in (self.storage.nodes_path, self.storage.events_path,
                        self.storage.llm_calls_path):
            truncate_torn_tail(journal)
        if self.storage.state_path.exists():
            self._resume_checkpoint()
        else:
            self._save_checkpoint()
        try:
            while self.evaluations_used < self.budget:
                self._run_candidate()
                print(
                    f"{self.METHOD}: budget={self.evaluations_used}/{self.budget} "
                    f"nodes={len(self.tree.nodes)} parent_attempts={self.tree.parent_selections}",
                    flush=True,
                )
            if len(self.tree.roots) < self.n_roots:
                raise RuntimeError("budget exhausted before initialization completed")
            self._write_summary("finished")
        except KeyboardInterrupt:
            self._write_summary("interrupted")
            raise
        except Exception:
            self._write_summary("error", traceback.format_exc())
            raise
