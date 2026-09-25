"""TraceAAD V11.0: percentile-plus-bonus scheduling with operator-conditional context."""

import json
import math
import random
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from core import SecureEvaluator
from . import parsing
from .prompts import ReferenceContextBuilder, TrajectoryBuilder
from .selection import (CodeBook, EXPLORATION_C, N_REFERENCES, OPERATORS,
                        OPERATOR_PROBABILITIES, code_key, quality_percentiles,
                        reciprocal_rank_sample)
from .storage import RunStorage, atomic_json, truncate_torn_tail
from .tree import Node, SearchTree

REPAIRABLE_FAILURES = {"exec_error", "runtime_error", "timeout", "invalid_result", "nonfinite_fitness"}


def _timestamp():
    return datetime.now().isoformat(timespec="seconds")


def _restore_rng(rng, state):
    rng.setstate((state[0], tuple(state[1]), state[2]))


@dataclass
class Candidate:
    """One scheduled design attempt (in-memory; checkpoints store settled
    state only). Field names match the journal fields they are reported in."""

    candidate_id: int
    prompt: str
    prompt_tokens: int
    requested_operator: str
    operator: str
    parent_id: int | None
    parent_fitness: float | None
    reference_ids: list
    selection: dict
    best_before: float | None
    llm_attempts: int = 0
    repair_of: int | None = None


class TraceAADV110:
    METHOD = "v110"
    PROMPT_POLICY = "operator_conditional_v1"

    def __init__(self, *, evaluation, llm, run_dir, budget=1000, n_roots=8,
                 traj_gens=8, output_tokens=8192, max_input_tokens=24576,
                 n_references=N_REFERENCES, seed=0):
        if budget < n_roots or n_roots < 1 or traj_gens < 0 or n_references < 1:
            raise ValueError("invalid budget, root, history, or reference settings")
        self.evaluation, self.llm = evaluation, llm
        self.run_dir = Path(run_dir)
        self.budget, self.n_roots, self.traj_gens = budget, n_roots, traj_gens
        self.output_tokens, self.max_input_tokens = output_tokens, max_input_tokens
        self.n_references = n_references
        self.secure = SecureEvaluator(evaluation)
        self._template_program = evaluation.template_program
        self._parse_interface, target_stub = parsing.template_target(self._template_program)
        self.task_contract = (
            "# Task\n\n" + evaluation.task_description.strip() +
            "\n\nImplement the target function described below. Its body and any supporting "
            "code are your design space.\n\nTarget function:\n```python\n" +
            target_stub + "\n```"
        )
        self.tree = SearchTree()
        self.codebook = CodeBook()
        self.rng = random.Random(seed)
        self.parent_selection_counts = {}
        self.step_counter = self.budget_used = self.completed_attempts = 0
        self.started_at = _timestamp()
        self._invalid_streak = 0
        self.storage = RunStorage(self.run_dir)
        self.mechanism = {
            "method": self.METHOD, "budget": budget, "n_roots": n_roots,
            "prompt_policy": self.PROMPT_POLICY,
            "parser_protocol": "target_function_anchored_module_v2",
            "traj_gens": traj_gens,
            "output_tokens": output_tokens, "max_input_tokens": max_input_tokens,
            "operator_probabilities": OPERATOR_PROBABILITIES,
            "exploration_c": EXPLORATION_C,
            "n_references": n_references,
            "reference_weighting": "reciprocal_rank",
            "llm": {name: getattr(llm, name, None) for name in
                    ("model", "base_url", "temperature", "top_p", "enable_thinking")},
        }
        self.builder = TrajectoryBuilder(
            llm, self.task_contract,
            max_tokens=max_input_tokens,
            max_events=traj_gens, lookup=self.tree.nodes.get,
            all_nodes=self.tree.all_nodes,
        )
        self.reference_builder = ReferenceContextBuilder(
            llm, self.task_contract,
            max_tokens=max_input_tokens,
            max_events=traj_gens, lookup=self.tree.nodes.get,
            all_nodes=self.tree.all_nodes,
        )

    # --- Scheduling: decide the next attempt ---------------------------------

    def _candidate(self, prompt, **fields):
        return Candidate(
            candidate_id=self.completed_attempts + 1,
            best_before=self.tree.best().fitness if self.tree.nodes else None,
            prompt=prompt, prompt_tokens=self.builder.count(prompt), **fields,
        )

    def _score_codes(self):
        keys = self.codebook.keys()
        means = [self.codebook.mean_fitness(key) for key in keys]
        percentiles = quality_percentiles(means)
        scores, bonuses = [], []
        for key, percentile in zip(keys, percentiles):
            attempts = self.codebook.attempts(key)
            bonus = EXPLORATION_C * math.sqrt(math.log1p(self.step_counter) / (1 + attempts))
            bonuses.append(bonus)
            scores.append(percentile + bonus)
        return keys, means, percentiles, bonuses, scores

    def select_parent(self):
        """Pick the top-scored code, uniform among ties, then a uniform formation node."""
        keys, means, percentiles, bonuses, scores = self._score_codes()
        top = max(scores)
        ties = [index for index, score in enumerate(scores) if score == top]
        index = self.rng.choice(ties)
        key = keys[index]
        node_ids = self.codebook.node_ids(key)
        parent = self.tree.nodes[self.rng.choice(node_ids)]
        selection = {
            "mean_fitness": means[index],
            "percentile": percentiles[index],
            "attempts": self.codebook.attempts(key),
            "bonus": bonuses[index],
            "score": scores[index],
            "tie_group_size": len(ties),
            "formation_candidates": len(node_ids),
        }
        return parent, selection

    def reference_pool(self, parent):
        """Archive nodes with a recorded idea, deduped by code (latest record),
        excluding the current algorithm's code."""
        current = code_key(parent.code)
        pool = {}
        for node in self.tree.all_nodes():
            key = code_key(node.code)
            if key == current or not node.idea or not node.idea.strip():
                continue
            pool[key] = node
        return list(pool.values())

    def _schedule_repair(self, previous):
        prompt = parsing.repair_prompt(
            self.task_contract, self.storage.failed_response(previous["candidate_id"]), previous)
        return self._candidate(
            prompt, repair_of=previous["candidate_id"],
            operator=previous.get("operator", "Init"),
            requested_operator=previous.get("requested_operator", "Init"),
            parent_id=previous.get("parent_id"),
            parent_fitness=previous.get("parent_fitness"),
            reference_ids=[], selection={},
        )

    def _schedule(self):
        previous = self.storage.last_event
        if previous and previous["candidate_id"] != self.completed_attempts:
            previous = None
        if previous and previous.get("status") == "eval_failed" and previous.get("reason") not in REPAIRABLE_FAILURES:
            raise RuntimeError(f"evaluation infrastructure failed: {previous.get('reason')}")
        if previous and (previous.get("status") == "invalid_output" or
                         previous.get("reason") in REPAIRABLE_FAILURES) and not previous.get("repair_of"):
            return self._schedule_repair(previous)

        requested = operator = "Init"
        parent = None
        selection = {}
        reference_ids = []
        if len(self.tree.roots) >= self.n_roots:
            requested = operator = self.rng.choices(OPERATORS, weights=OPERATOR_PROBABILITIES.values())[0]
            parent, selection = self.select_parent()
            if requested in ("Pivot", "Fuse"):
                sampled = reciprocal_rank_sample(self.reference_pool(parent),
                                                 self.n_references, self.rng)
                text, retained = self.reference_builder.build(
                    parent, requested, sampled, require_minimal=requested != "Fuse")
                reference_ids = [node.id for node in retained]
                if requested == "Fuse" and not retained:
                    # Decide the final operator first; the fallback prompt carries
                    # the capacity check (Refine rules), never the Fuse minimum.
                    operator = "Refine"
                    selection["fallback_reason"] = ("reference_pool_empty" if not sampled
                                                    else "references_trimmed_to_zero")
                    text = self.builder.build(parent, "Refine")
                    reference_ids = []
            else:
                text = self.builder.build(parent, operator)
        else:
            text = self.builder.build_initial()
        return self._candidate(
            text, requested_operator=requested, operator=operator,
            parent_id=parent.id if parent else None,
            parent_fitness=parent.fitness if parent else None,
            reference_ids=reference_ids, selection=selection,
        )

    # --- Execution: generate, parse, evaluate --------------------------------

    def _generate(self, candidate):
        """Call the LLM once; return the completion record."""
        candidate.llm_attempts += 1
        started = time.time()
        record = {"ts": _timestamp(),
                  "call_id": f"{candidate.candidate_id}:{candidate.llm_attempts}",
                  "candidate_id": candidate.candidate_id}
        try:
            details = self.llm.draw_sample_with_details(
                candidate.prompt, max_tokens=self.output_tokens
            )
        except Exception:
            record.update(seconds=time.time() - started, error=traceback.format_exc())
            self.storage.record_call(record)
            raise
        record.update(response=details["content"], seconds=time.time() - started,
                      finish_reason=details.get("finish_reason") or "unknown", usage=details.get("usage"),
                      model=details.get("model"), response_id=details.get("response_id"),
                      stage="repair" if candidate.repair_of else "generation")
        self.storage.record_call(record)
        return record

    def _score_result(self, result):
        reason, error_type, error, trace = (
            result.failure_kind, result.error_type, result.error, result.traceback)
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
        """Evaluate a parsed candidate and add its node; (None, None) if unparsable."""
        if parsed is None:
            return None, None
        evaluation_id = self.budget_used + 1
        started = time.time()
        try:
            fitness, reason, error_type, error, trace = self._score_result(
                self.secure.evaluate_program_with_details(parsed.program_code))
        except Exception as exc:
            fitness, reason, error_type, error, trace = (
                None, "evaluation_error", type(exc).__name__, str(exc), traceback.format_exc())
        outcome = {"candidate_id": candidate.candidate_id, "evaluation_id": evaluation_id,
                   "fitness": fitness, "reason": reason, "error_type": error_type, "error": error,
                   "traceback": trace, "eval_seconds": time.time() - started,
                   "repair_of": candidate.repair_of}
        self.budget_used = evaluation_id
        node = None
        if fitness is not None:
            node = self.tree.add(code=parsed.program_code, idea=parsed.idea, fitness=fitness,
                                 evaluation_id=evaluation_id, parent_id=candidate.parent_id,
                                 operator=candidate.operator, donor_id=None)
            self.storage.record_node(node)
            self.codebook.register(node)
        return outcome, node

    # --- Settlement: journal event, checkpoint -------------------------------

    def _settle(self, candidate, completion, parse_error, outcome, node):
        """Record the finished candidate and checkpoint the settled state."""
        if parse_error is not None:
            self._invalid_streak += 1
            status, reason = "invalid_output", parse_error
        else:
            self._invalid_streak = 0
            status, reason = ("ok", None) if outcome["fitness"] is not None else ("eval_failed", outcome["reason"])
        if candidate.parent_id is not None:
            # Commit one generation attempt against the parent code: n += 1, T += 1.
            self.step_counter += 1
            parent = self.tree.nodes[candidate.parent_id]
            self.codebook.note_attempt(code_key(parent.code))
            self.parent_selection_counts[parent.id] = self.parent_selection_counts.get(parent.id, 0) + 1
        record = {
            "ts": _timestamp(),
            "candidate_id": candidate.candidate_id,
            "requested_operator": candidate.requested_operator,
            "operator": candidate.operator, "repair_of": candidate.repair_of,
            "parent_id": candidate.parent_id, "donor_id": None,
            "parent_fitness": candidate.parent_fitness, "donor_fitness": None,
            "reference_ids": candidate.reference_ids, "selection": candidate.selection,
            "prompt_tokens": candidate.prompt_tokens,
            "status": status, "reason": reason,
            "budget_used": self.budget_used,
            "evaluation_id": outcome["evaluation_id"] if outcome else None,
            "eval_seconds": outcome.get("eval_seconds") if outcome else None,
            "llm_seconds": completion["seconds"],
            "node_id": node.id if node else None,
            "fitness": node.fitness if node else None,
        }
        if outcome is not None:
            record.update(error_type=outcome.get("error_type"),
                          error=outcome.get("error"),
                          traceback=outcome.get("traceback"),
                          best_fitness=self.tree.best().fitness if self.tree.nodes else None)
        if node and candidate.parent_id is not None:
            record.update(parent_improved=node.fitness > candidate.parent_fitness,
                          frontier_improved=node.fitness > candidate.best_before)
        self.storage.record_event(record)
        self.completed_attempts = candidate.candidate_id
        self._save_checkpoint()
        if self._invalid_streak >= 50:
            raise RuntimeError("50 consecutive generations produced no valid output")

    def _run_candidate(self):
        candidate = self._schedule()
        completion = self._generate(candidate)
        parsed, parse_error = parsing.parse_candidate(
            completion["response"], completion["finish_reason"],
            self._parse_interface, self._template_program,
        )
        outcome, node = self._evaluate_and_add(parsed, candidate)
        self._settle(candidate, completion, parse_error, outcome, node)

    # --- Checkpoint and recovery ----------------------------------------------

    def _save_checkpoint(self):
        """Persist the fully settled state; recovery resumes from here, so an
        interrupted in-flight candidate is simply redone."""
        atomic_json(self.storage.state_path, {
            "mechanism": self.mechanism,
            "started_at": self.started_at,
            "rng_state": list(self.rng.getstate()),
            "nodes": [asdict(node) for node in self.tree.all_nodes()],
            "code_attempts": self.codebook.attempts_table(),
            "parent_selection_counts": self.parent_selection_counts,
            "step_counter": self.step_counter,
            "budget_used": self.budget_used,
            "completed_attempts": self.completed_attempts,
            "invalid_streak": self._invalid_streak,
            "last_event": self.storage.last_event,
        })

    def _resume_checkpoint(self):
        state = json.loads(self.storage.state_path.read_text())
        if state.get("mechanism") != self.mechanism:
            raise ValueError("checkpoint configuration differs from V11.0")
        self.started_at = state["started_at"]
        _restore_rng(self.rng, state["rng_state"])
        for entry in state["nodes"]:
            node = Node(**entry)
            self.tree.add_raw(node)
            self.codebook.register(node)
        self.codebook.restore_attempts(state["code_attempts"])
        self.parent_selection_counts = {int(key): value
                                        for key, value in state["parent_selection_counts"].items()}
        self.step_counter = state["step_counter"]
        self.budget_used = state["budget_used"]
        self.completed_attempts = state["completed_attempts"]
        self._invalid_streak = state["invalid_streak"]
        self.storage.last_event = state["last_event"]

    def _write_summary(self, status, error=None):
        best = self.tree.best() if self.tree.nodes else None
        payload = {"status": status, "method": self.METHOD, "started_at": self.started_at,
                   "finished_at": _timestamp(), "budget": self.budget,
                   "budget_used": self.budget_used, "num_nodes": len(self.tree.nodes),
                   "num_codes": len(self.codebook.entries), "num_roots": len(self.tree.roots),
                   "best": None if best is None else {"node_id": best.id, "fitness": best.fitness,
                   "idea": best.idea, "code": best.code, "evaluation_id": best.evaluation_id,
                   "operator": best.operator, "parent_id": best.parent_id, "donor_id": best.donor_id}}
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
            while self.budget_used < self.budget:
                self._run_candidate()
                print(f"{self.METHOD}: budget={self.budget_used}/{self.budget} "
                      f"nodes={len(self.tree.nodes)} codes={len(self.codebook.entries)}", flush=True)
            if len(self.tree.roots) < self.n_roots:
                raise RuntimeError("budget exhausted before initialization completed")
            self._write_summary("finished")
        except KeyboardInterrupt:
            self._write_summary("interrupted")
            raise
        except Exception:
            self._write_summary("error", traceback.format_exc())
            raise
