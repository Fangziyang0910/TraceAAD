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
    QUALITY_ESS_TARGET, PARENT_UNIFORM_PROBABILITY, REFERENCE_COUNT,
    reference_shortlist,
    sample_parent,
)
from .storage import RunStorage, atomic_json, digest, truncate_torn_tail
from .tree import Node, SearchTree

REPAIRABLE_FAILURES = {
    "exec_error", "runtime_error", "timeout", "invalid_result", "nonfinite_fitness",
}


class UncertainEvaluationError(RuntimeError):
    """A durable reservation exists but its evaluation receipt is missing."""


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
    trial_ids: list[int] = field(default_factory=list)
    loaded_reference_id: int | None = None
    context_reads: int = 0
    llm_attempts: int = 0
    repair_of: int | None = None


class TraceAADV1013:
    METHOD = "v1013"
    REVISION = "v10.13-r2"
    PROMPT_POLICY = "idea_context_optional_read_full_edit_v10_13_r2"

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
        self.pending = None
        self.storage = RunStorage(self.run_dir)
        self.mechanism = {
            "method": self.METHOD,
            "budget": budget,
            "n_roots": n_roots,
            "prompt_policy": self.PROMPT_POLICY,
            "revision": self.REVISION,
            "parser_protocol": "code_first_json_full_edit_optional_idea_v2",
            "persistence_protocol": "staged_candidate_receipt_v2",
            "history_depth": history_depth,
            "output_tokens": output_tokens,
            "max_input_tokens": max_input_tokens,
            "operator_probabilities": OPERATOR_PROBABILITIES,
            "quality_ess_target": QUALITY_ESS_TARGET,
            "parent_uniform_probability": PARENT_UNIFORM_PROBABILITY,
            "reference_count": REFERENCE_COUNT,
            "reference_policy": "quality_uniform_underexposed_then_model_read",
            "task_contract_hash": digest(self.task_contract),
            "template_hash": digest(self._template_program),
            "seed": seed,
            "context_mapping": {"Refine": "idea_formation_optional_trials",
                                "Tune": "idea_formation_optional_trials",
                                "Pivot": "idea_shortlist_optional_code",
                                "Fuse": "idea_shortlist_optional_code"},
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
        response = self.storage.failed_response(previous['candidate_id'])
        parent = self.tree.nodes.get(previous.get('parent_id'))
        parsed, _ = parsing.parse_candidate(
            response, 'unknown', self._parse_interface, self._template_program,
            base_code=parent.code if parent else None,
            allowed_donor_ids=previous.get('reference_ids', []),
        )
        prompt = parsing.build_repair_prompt(
            self.task_contract,
            response,
            previous,
            base_code=parent.code if parent else None,
            failed_program=parsed.program_code if parsed else None,
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
            trial_ids=previous.get('trial_ids', []),
            loaded_reference_id=previous.get('loaded_reference_id'),
            context_reads=0,  # This repair makes no new context read; repair_of disables reads.
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

        references = []
        if requested_operator in ("Pivot", "Fuse"):
            references, stats = reference_shortlist(self.tree.all_nodes(), parent, self.rng)
            selection.update(stats)

        context = self.prompts.build_development(parent, requested_operator, references=references)
        if context.fallback_reason:
            selection["fallback_reason"] = context.fallback_reason
        for identity in context.reference_ids:
            self.tree.nodes[identity].reference_uses += 1
        return self._candidate(
            context.prompt,
            requested_operator=requested_operator,
            operator=context.operator,
            parent_id=parent.id,
            donor_id=None,
            parent_fitness=parent.fitness,
            donor_fitness=None,
            reference_ids=context.reference_ids,
            history_ids=context.history_ids,
            selection=selection,
            parent_selected=True,
        )

    def _generate(self, candidate):
        candidate.llm_attempts += 1
        self.pending['candidate'] = asdict(candidate)
        self.pending['call_id'] = f'{candidate.candidate_id}:{candidate.llm_attempts}'
        self._save_checkpoint()  # Reserve the call ID before dispatch.
        started = time.time()
        record = {
            "ts": _timestamp(),
            "call_id": f"{candidate.candidate_id}:{candidate.llm_attempts}",
            "candidate_id": candidate.candidate_id,
            "operator": candidate.operator,
            "context_round": candidate.context_reads,
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
        self.evaluations_used = evaluation_id
        self.pending.update(stage='evaluating', evaluation_id=evaluation_id)
        self._save_checkpoint()  # A missing receipt must never trigger automatic reevaluation.
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
        self.storage.record_evaluation(outcome)
        self.pending.update(stage='evaluated', outcome=outcome)
        self._save_checkpoint()
        return outcome, self._node_for_outcome(parsed, candidate, outcome)

    def _node_for_outcome(self, parsed, candidate, outcome):
        if outcome is None or outcome['fitness'] is None:
            return None

        node = self.tree.add(
            code=parsed.program_code,
            idea=parsed.idea,
            fitness=outcome['fitness'],
            evaluation_id=outcome['evaluation_id'],
            parent_id=candidate.parent_id,
            operator=candidate.operator,
            donor_id=candidate.donor_id,
            idea_fields=parsed.idea_fields,
        )
        return node

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
        calls = [self.storage.index(self.storage.llm_calls_path, 'call_id').get(
                 f'{candidate.candidate_id}:{i}') for i in range(1, candidate.llm_attempts + 1)]
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
            "trial_ids": candidate.trial_ids,
            "loaded_reference_id": candidate.loaded_reference_id,
            "context_reads": candidate.context_reads,
            "output_mode": self.pending.get('parsed', {}).get('mode') if self.pending.get('parsed') else None,
            "llm_calls": candidate.llm_attempts,
            "selection": candidate.selection,
            "parent_selected": candidate.parent_selected,
            "prompt_tokens": candidate.prompt_tokens,
            "prompt_hash": candidate.prompt_hash,
            "status": status,
            "reason": reason,
            "budget_used": self.evaluations_used,
            "evaluation_id": outcome["evaluation_id"] if outcome else None,
            "eval_seconds": outcome.get("eval_seconds") if outcome else None,
            "llm_seconds": sum(call['seconds'] for call in calls if call),
            "recorded_llm_calls": sum(call is not None for call in calls),
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
        self.completed_candidates = candidate.candidate_id
        self.storage.last_event = record
        self.pending.update(stage='committing', event=record,
                            node=asdict(node) if node else None)
        self._save_checkpoint()  # State is authoritative; journals are idempotent projections.
        self._flush_commit()
        if self._invalid_streak >= 50:
            raise RuntimeError("50 consecutive generations produced no valid output")

    def _flush_commit(self):
        if self.pending.get('node') is not None:
            self.storage.record_node(Node(**self.pending['node']))
        self.storage.record_event(self.pending['event'])
        self.pending = None
        self._save_checkpoint()

    def _run_candidate(self):
        if self.pending is None:
            candidate = self._schedule_candidate()
            self.pending = {'stage': 'scheduled', 'candidate': asdict(candidate)}
            self._save_checkpoint()
        if self.pending['stage'] == 'committing':
            self._flush_commit()
            return
        candidate = Candidate(**self.pending['candidate'])
        stage = self.pending['stage']
        if stage == 'evaluating':
            receipt = self.storage.index(self.storage.evaluations_path, 'evaluation_id').get(
                self.pending['evaluation_id'])
            if receipt is None:
                raise UncertainEvaluationError(
                    f"evaluation {self.pending['evaluation_id']} was reserved, but has no durable result; "
                    "automatic reevaluation is blocked; retain the checkpoint and journals")
            if receipt['candidate_id'] != candidate.candidate_id:
                raise ValueError('evaluation receipt belongs to a different candidate')
            self.pending.update(stage='evaluated', outcome=receipt)
            self._save_checkpoint()
            stage = 'evaluated'
        if stage in ('parsed', 'evaluated'):
            parsed = parsing.ParsedCandidate(**self.pending['parsed']) if self.pending['parsed'] else None
            outcome = self.pending.get('outcome')
            if stage == 'parsed':
                outcome, node = self._evaluate_and_add(parsed, candidate)
            else:
                node = self._node_for_outcome(parsed, candidate, outcome)
            self._settle(candidate, self.pending['completion'], self.pending.get('parse_error'), outcome, node)
            return
        if stage == 'scheduled':
            # Reuse a completed call that landed in the journal before the checkpoint.
            call = self.storage.index(self.storage.llm_calls_path, 'call_id').get(
                self.pending.get('call_id'))
            completion = call if call and 'response' in call else self._generate(candidate)
            self.pending.update(stage='generated', candidate=asdict(candidate), completion=completion)
            self._save_checkpoint()
        completion = self.pending['completion']
        payload = parsing.response_object(completion['response'])
        if payload is not None and payload.get('mode') == 'context':
            if candidate.parent_id is not None and not candidate.context_reads and not candidate.repair_of:
                requested_id = payload.get('reference_id')
                reference = (self.tree.nodes[requested_id] if type(requested_id) is int and
                             requested_id in candidate.reference_ids else None)
                include_trials = payload.get('trials') is True and candidate.operator in ('Refine', 'Tune')
                context = self.prompts.build_development(
                    self.tree.nodes[candidate.parent_id], candidate.operator,
                    references=[self.tree.nodes[i] for i in candidate.reference_ids],
                    read_reference=reference, include_trials=include_trials,
                    allow_context=False,
                )
                candidate.prompt, candidate.prompt_hash = context.prompt, digest(context.prompt)
                candidate.prompt_tokens = self.prompts.count(context.prompt)
                candidate.history_ids, candidate.trial_ids = context.history_ids, context.trial_ids
                candidate.context_reads = 1
                candidate.loaded_reference_id = context.reference_program.id if context.reference_program else None
                if context.fallback_reason:
                    candidate.selection['read_fallback_reason'] = context.fallback_reason
                candidate.selection['context_request'] = {
                    'reference_id': requested_id if type(requested_id) is int else None,
                    'trials': payload.get('trials') is True,
                }
                candidate.selection['read_reference_ids'] = context.reference_ids
                candidate.selection['local_trials_requested_and_allowed'] = include_trials
                self.pending = {'stage': 'scheduled', 'candidate': asdict(candidate)}
                self._save_checkpoint()
                return self._run_candidate()
            parsed, parse_error = None, 'context_error: context reads are limited to one development round'
        else:
            parsed, parse_error = self._parse_completion(completion, candidate)
        if parsed is not None:
            candidate.donor_id = parsed.donor_id
            candidate.donor_fitness = (self.tree.nodes[parsed.donor_id].fitness
                                       if parsed.donor_id is not None else None)
        self.pending.update(stage='parsed', candidate=asdict(candidate),
                            parsed=asdict(parsed) if parsed else None, parse_error=parse_error)
        self._save_checkpoint()
        outcome, node = self._evaluate_and_add(parsed, candidate)
        self._settle(candidate, completion, parse_error, outcome, node)

    def _parse_completion(self, completion, candidate):
        parent = self.tree.nodes.get(candidate.parent_id)
        parsed, parse_error = parsing.parse_candidate(
            completion["response"], completion["finish_reason"],
            self._parse_interface, self._template_program,
            base_code=parent.code if parent else None,
            allowed_donor_ids=candidate.reference_ids,
        )
        return parsed, parse_error

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
            "pending": self.pending,
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
        self.pending = state.get('pending')
        # Fail closed on legacy/unexplained journal tails; never silently reuse IDs.
        events = self.storage.index(self.storage.events_path, 'candidate_id')
        nodes = self.storage.index(self.storage.nodes_path, 'id')
        receipts = self.storage.index(self.storage.evaluations_path, 'evaluation_id')
        committing = self.pending and self.pending['stage'] == 'committing'
        missing_events = {self.completed_candidates} if committing else set()
        missing_nodes = ({self.pending['node']['id']} if committing and self.pending.get('node') else set())
        expected_events = set(range(1, self.completed_candidates + 1))
        if set(events) - expected_events or expected_events - set(events) - missing_events:
            raise ValueError('event journal is inconsistent with the durable checkpoint')
        if set(nodes) - set(self.tree.nodes) or set(self.tree.nodes) - set(nodes) - missing_nodes:
            raise ValueError('node journal is inconsistent with the durable checkpoint')
        expected_receipts = set(range(1, self.evaluations_used + 1))
        uncertain = ({self.pending['evaluation_id']} if self.pending and
                     self.pending['stage'] == 'evaluating' else set())
        if set(receipts) - expected_receipts or expected_receipts - set(receipts) - uncertain:
            raise ValueError('evaluation journal is inconsistent with the durable checkpoint')
        for event in events.values():
            identity = event.get('evaluation_id')
            if identity is not None and (identity not in receipts or
                    receipts[identity]['candidate_id'] != event['candidate_id']):
                raise ValueError('event and evaluation receipt disagree')
        for identity, record in nodes.items():
            current = asdict(self.tree.nodes[identity])
            if any(record[key] != current[key] for key in record
                   if key not in ('attempts', 'reference_uses')):
                raise ValueError('node journal and checkpoint disagree')

    def _write_summary(self, status, error=None):
        best = self.tree.best() if self.tree.nodes else None
        payload = {
            "status": status,
            "method": self.METHOD,
            "revision": self.REVISION,
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
                "idea_fields": best.idea_fields,
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
        with self.storage.writer_lock():
            return self._run_locked()

    def _run_locked(self):
        # Reject old revision checkpoints before touching their journals.
        if self.storage.state_path.exists():
            state = json.loads(self.storage.state_path.read_text())
            if state.get('mechanism') != self.mechanism:
                raise ValueError('checkpoint belongs to a different revision; use its frozen runtime')
        elif any(path.exists() and path.stat().st_size for path in (
                self.storage.nodes_path, self.storage.events_path, self.storage.llm_calls_path,
                self.storage.evaluations_path)):
            raise ValueError('journals exist without a checkpoint; refusing to start over')
        for journal in (self.storage.nodes_path, self.storage.events_path,
                        self.storage.llm_calls_path, self.storage.evaluations_path):
            truncate_torn_tail(journal)
        if self.storage.state_path.exists():
            self._resume_checkpoint()
        else:
            self._save_checkpoint()
        try:
            while self.pending is not None or self.evaluations_used < self.budget:
                self._run_candidate()
                event = self.storage.last_event
                if event and event.get('status') == 'eval_failed' and event.get('reason') not in REPAIRABLE_FAILURES:
                    raise RuntimeError(f"evaluation infrastructure failed: {event.get('reason')}")
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
        except UncertainEvaluationError:
            self._write_summary('uncertain_evaluation', traceback.format_exc())
            raise
        except Exception:
            self._write_summary("error", traceback.format_exc())
            raise
