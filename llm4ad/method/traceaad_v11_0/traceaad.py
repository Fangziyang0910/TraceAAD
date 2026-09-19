"""TraceAAD V11.0: percentile-plus-bonus scheduling with operator-conditional context."""

import ast
import json
import math
import os
import random
import time
import traceback
from bisect import bisect_left, bisect_right
from dataclasses import asdict
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from llm4ad.base import SecureEvaluator
from . import errors, trajectory
from .core import (Node, SearchTree, UnknownEvaluation, atomic_json, digest,
                   read_journal, truncate_torn_tail)
from .trajectory import ReferenceContextBuilder, TrajectoryBuilder

OPERATORS = ("Refine", "Tune", "Pivot", "Fuse")
OPERATOR_PROBABILITIES = {operator: 0.25 for operator in OPERATORS}
EXPLORATION_C = 0.1
N_REFERENCES = 8
REPAIRABLE_FAILURES = {"exec_error", "runtime_error", "timeout", "invalid_result", "nonfinite_fitness"}


def _timestamp():
    return datetime.now().isoformat(timespec="seconds")


def _restore_rng(rng, state):
    rng.setstate((state[0], tuple(state[1]), state[2]))


@lru_cache(maxsize=8192)
def code_key(code):
    return ast.dump(ast.parse(code), include_attributes=False)


def quality_percentiles(mean_scores):
    """Midrank percentile in [0, 1] across distinct codes; 0.5 for a single code."""
    total = len(mean_scores)
    if total == 1:
        return [0.5]
    ascending = sorted(mean_scores)
    percentiles = []
    for score in mean_scores:
        low, high = bisect_left(ascending, score), bisect_right(ascending, score)
        lower, equal = low, high - low
        percentiles.append((lower + (equal - 1) / 2) / (total - 1))
    return percentiles


def reference_ranks(fitnesses):
    """Midrank over fitness, better fitness first; ties share the average rank."""
    ascending = sorted(fitnesses)
    total = len(fitnesses)
    ranks = []
    for fitness in fitnesses:
        low, high = bisect_left(ascending, fitness), bisect_right(ascending, fitness)
        ranks.append((total - high) + (high - low + 1) / 2)
    return ranks


def reciprocal_rank_sample(nodes, k, rng):
    """Draw up to k nodes without replacement, weighting each node by 1/rank.

    Ranks and weights stay at their original values during the draw; the
    remaining pool is implicitly renormalized at every pick.
    """
    ordered = sorted(nodes, key=lambda node: (-node.fitness, node.id))
    weights = [1.0 / rank for rank in reference_ranks([node.fitness for node in ordered])]
    pool = list(zip(ordered, weights))
    chosen = []
    while len(chosen) < k and pool:
        pick = rng.random() * sum(weight for _, weight in pool)
        index, cumulative = 0, 0.0
        while index < len(pool) - 1:
            cumulative += pool[index][1]
            if pick < cumulative:
                break
            index += 1
        chosen.append(pool.pop(index)[0])
    return chosen


class CodeBook:
    """Scheduling statistics aggregated per distinct AST code key."""

    def __init__(self):
        self.entries = {}

    def register(self, node):
        """Record one valid evaluation; new codes enter with zero attempts."""
        entry = self.entries.setdefault(code_key(node.code),
                                        {"attempts": 0, "scores": [], "node_ids": []})
        entry["scores"].append(node.fitness)
        entry["node_ids"].append(node.id)

    def note_attempt(self, key):
        self.entries[key]["attempts"] += 1

    def keys(self):
        return list(self.entries)

    def attempts(self, key):
        return self.entries[key]["attempts"]

    def mean_fitness(self, key):
        scores = self.entries[key]["scores"]
        return sum(scores) / len(scores)

    def node_ids(self, key):
        return self.entries[key]["node_ids"]

    def attempts_table(self):
        return {digest(key): entry["attempts"] for key, entry in self.entries.items()}


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
        self._parse_interface, target_stub = errors.template_target(self._template_program)
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
        self._parse_error = None
        self._last_response = None
        self.pending = None
        self.events_path = self.run_dir / "events.jsonl"
        self.llm_calls_path = self.run_dir / "llm_calls.jsonl"
        self.nodes_path = self.run_dir / "nodes.jsonl"
        self.pending_path = self.run_dir / "pending_candidate.json"
        self.state_path = self.run_dir / "tree_state.json"
        self.summary_path = self.run_dir / "logs" / "run_summary.json"
        self._outcomes = {}
        self._logged_calls = set()
        self._logged_events = set()
        self._logged_nodes = set()
        self._nodes_by_evaluation = {}
        self._last_event = None
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
            "task_contract_hash": digest(self.task_contract),
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
        self._rebuild_from_logs()

    def parse_response(self, response, finish_reason="unknown"):
        parsed, error = errors.parse_candidate(response, finish_reason,
                                                self._parse_interface,
                                                self._template_program,
                                                token_counter=self.llm.count_tokens)
        self._parse_error = error
        return parsed

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
            "code_digest": digest(key)[:16],
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

    def _current_event(self):
        previous = self._last_event
        if previous and previous["candidate_id"] == self.completed_attempts:
            return previous
        return None

    def _failed_response(self, candidate_id):
        if self._last_response and self._last_response[0] == candidate_id:
            return self._last_response[1]
        return next(record["response"] for record in reversed(read_journal(self.llm_calls_path))
                    if record.get("candidate_id") == candidate_id and "response" in record)

    def _pending(self, prompt, **fields):
        pending = {
            "candidate_id": self.completed_attempts + 1, "phase": "selected",
            "best_before": self.tree.best().fitness if self.tree.nodes else None,
            "prompt": prompt, "prompt_tokens": self.builder.count(prompt),
            "prompt_hash": digest(prompt), "rng_state": list(self.rng.getstate()),
            "llm_attempts": 0, "reference_ids": [],
        }
        pending.update(fields)
        return pending

    def _schedule_repair(self, previous):
        prompt = errors.repair_prompt(
            self.task_contract, self._failed_response(previous["candidate_id"]), previous)
        return self._pending(
            prompt, repair_of=previous["candidate_id"],
            operator=previous.get("operator", "Init"),
            requested_operator=previous.get("requested_operator", "Init"),
            parent_id=previous.get("parent_id"), donor_id=None,
            parent_fitness=previous.get("parent_fitness"), donor_fitness=None,
            reference_ids=[],
        )

    def _schedule(self):
        previous = self._current_event()
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
        return self._pending(
            text, requested_operator=requested, operator=operator,
            parent_id=parent.id if parent else None, donor_id=None,
            parent_fitness=parent.fitness if parent else None, donor_fitness=None,
            reference_ids=reference_ids, selection=selection,
        )

    def _persist_pending(self):
        atomic_json(self.pending_path, self.pending)

    def _log_call(self, record):
        if record["call_id"] in self._logged_calls:
            return
        self._append_record(self.llm_calls_path, record)
        self._logged_calls.add(record["call_id"])
        if "response" in record:
            self._last_response = (record["candidate_id"], record["response"])

    def _generate_pending(self):
        self.pending["llm_attempts"] += 1
        self._persist_pending()
        started = time.time()
        record = {"ts": _timestamp(),
                  "call_id": f"{self.pending['candidate_id']}:{self.pending['llm_attempts']}",
                  "candidate_id": self.pending["candidate_id"], "operator": self.pending["operator"],
                  "prompt_tokens": self.pending["prompt_tokens"],
                  "prompt_hash": self.pending["prompt_hash"], "max_tokens": self.output_tokens}
        try:
            details = self.llm.draw_sample_with_details(
                self.pending["prompt"], max_tokens=self.output_tokens
            )
        except Exception:
            record.update(seconds=time.time() - started, error=traceback.format_exc())
            self._log_call(record)
            raise
        record.update(response=details["content"], seconds=time.time() - started,
                      finish_reason=details.get("finish_reason") or "unknown", usage=details.get("usage"),
                      model=details.get("model"), response_id=details.get("response_id"),
                      stage="repair" if self.pending.get("repair_of") else "generation")
        self.pending.update(phase="responded", completion=record)
        self._persist_pending()
        self._log_call(record)

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

    def _evaluate_pending(self, parsed):
        """Return the settled outcome, calling the evaluator at most once per candidate."""
        if self.pending["phase"] == "evaluated":
            return self.pending["outcome"]
        if self.pending["phase"] == "evaluating":
            outcome = self._outcomes.get(self.pending["candidate_id"])
            if outcome is None:
                raise UnknownEvaluation("evaluation reservation has no receipt")
            return outcome
        self.pending.update(phase="evaluating", evaluation_id=self.budget_used + 1)
        self._persist_pending()
        started = time.time()
        try:
            fitness, reason, error_type, error, trace = self._score_result(
                self.secure.evaluate_program_with_details(parsed[2]))
        except Exception as exc:
            fitness, reason, error_type, error, trace = (
                None, "evaluation_error", type(exc).__name__, str(exc), traceback.format_exc())
        outcome = {"candidate_id": self.pending["candidate_id"], "evaluation_id": self.pending["evaluation_id"],
                   "fitness": fitness, "reason": reason, "error_type": error_type, "error": error,
                   "traceback": trace, "eval_seconds": time.time() - started,
                   "repair_of": self.pending.get("repair_of")}
        self.pending.update(phase="evaluated", outcome=outcome)
        self._persist_pending()
        return outcome

    def _settle_node(self, outcome, parsed):
        """Create the node for a valid evaluation exactly once per candidate."""
        existing = self._nodes_by_evaluation.get(outcome["evaluation_id"])
        if existing is not None:
            return self.tree.nodes[existing]
        node = self.tree.add(code=parsed[2], idea=parsed[0], fitness=outcome["fitness"],
                             evaluation_id=outcome["evaluation_id"],
                             parent_id=self.pending["parent_id"],
                             operator=self.pending["operator"], donor_id=None)
        # Every program is recorded exactly once, appended at creation.
        if node.id not in self._logged_nodes:
            self._append_record(self.nodes_path, asdict(node))
            self._logged_nodes.add(node.id)
        self._nodes_by_evaluation[node.evaluation_id] = node.id
        self.codebook.register(node)
        return node

    def _settle_attempt(self):
        """Commit one generation attempt against the parent code: n += 1, T += 1."""
        self.step_counter += 1
        parent = self.tree.nodes[self.pending["parent_id"]]
        self.codebook.note_attempt(code_key(parent.code))
        self.parent_selection_counts[parent.id] = self.parent_selection_counts.get(parent.id, 0) + 1

    def _advance(self):
        if self.pending is None:
            self.pending = self._schedule()
            self._persist_pending()
        if self.pending["phase"] == "selected":
            self._generate_pending()
        self._log_call(self.pending["completion"])
        completion = self.pending["completion"]
        parsed = self.parse_response(completion["response"], completion["finish_reason"])
        node = None
        outcome = None
        if parsed is None:
            self._invalid_streak += 1
            status, reason = "invalid_output", self._parse_error
        else:
            self._invalid_streak = 0
            outcome = self._evaluate_pending(parsed)
            self.budget_used = outcome["evaluation_id"]
            status, reason = ("ok", None) if outcome["fitness"] is not None else ("eval_failed", outcome["reason"])
            if outcome["fitness"] is not None:
                node = self._settle_node(outcome, parsed)
        if self.pending["candidate_id"] not in self._logged_events:
            if self.pending["parent_id"] is not None:
                self._settle_attempt()
            record = {key: value for key, value in self.pending.items()
                      if key not in ("prompt", "rng_state", "completion", "phase", "outcome")}
            record.update(ts=_timestamp(), status=status, reason=reason,
                          budget_used=self.budget_used,
                          evaluation_id=outcome["evaluation_id"] if outcome else None,
                          eval_seconds=outcome.get("eval_seconds") if outcome else None,
                          llm_seconds=completion["seconds"], node_id=node.id if node else None,
                          fitness=node.fitness if node else None)
            if outcome is not None:
                record.update(error_type=outcome.get("error_type"),
                              error=outcome.get("error"),
                              traceback=outcome.get("traceback"),
                              best_fitness=self.tree.best().fitness if self.tree.nodes else None)
            if node and self.pending["parent_id"] is not None:
                record.update(parent_improved=node.fitness > self.pending["parent_fitness"],
                              frontier_improved=node.fitness > self.pending["best_before"])
            self._append_record(self.events_path, record)
            self._logged_events.add(self.pending["candidate_id"])
        self.completed_attempts = self.pending["candidate_id"]
        self._save_state()
        self.pending_path.unlink(missing_ok=True)
        self.pending = None
        if self._invalid_streak >= 50:
            raise RuntimeError("50 consecutive generations produced no valid output")

    def _append_record(self, path, record):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path == self.events_path:
            self._last_event = record

    def _save_state(self):
        atomic_json(self.state_path, {"mechanism": self.mechanism,
                                      "started_at": self.started_at,
                                      "rng_state": list(self.rng.getstate()),
                                      "parent_selection_counts": self.parent_selection_counts,
                                      "code_attempts": self.codebook.attempts_table(),
                                      "step_counter": self.step_counter,
                                      "budget_used": self.budget_used,
                                      "completed_attempts": self.completed_attempts,
                                      "invalid_streak": self._invalid_streak})

    def _rebuild_from_logs(self):
        """Restore every replayable quantity from the append journals (the truth)."""
        for journal in (self.nodes_path, self.events_path, self.llm_calls_path):
            truncate_torn_tail(journal)
        for entry in read_journal(self.nodes_path):
            node = Node(**entry)
            self.tree.add_raw(node)
            self.codebook.register(node)
            self._logged_nodes.add(node.id)
            if node.evaluation_id is not None:
                self._nodes_by_evaluation[node.evaluation_id] = node.id
        events = read_journal(self.events_path)
        self._outcomes = {record["candidate_id"]: record for record in events}
        self._logged_events = set(self._outcomes)
        self._logged_calls = {record["call_id"] for record in read_journal(self.llm_calls_path)}
        self._last_event = events[-1] if events else None
        self.completed_attempts = max(self._outcomes, default=0)
        self.budget_used = max((record.get("evaluation_id") for record in events
                                if record.get("evaluation_id") is not None), default=0)
        attempts = 0
        for record in events:
            parent_id = record.get("parent_id")
            if parent_id is None:
                continue
            attempts += 1
            parent = self.tree.nodes[parent_id]
            self.codebook.note_attempt(code_key(parent.code))
            self.parent_selection_counts[parent.id] = self.parent_selection_counts.get(parent.id, 0) + 1
        self.step_counter = attempts
        streak = 0
        for record in reversed(events):
            if record.get("status") != "invalid_output":
                break
            streak += 1
        self._invalid_streak = streak

    def _load_state(self):
        state = json.loads(self.state_path.read_text())
        if state.get("mechanism") != self.mechanism:
            raise ValueError("checkpoint configuration differs from V11.0")
        self.started_at = state["started_at"]
        _restore_rng(self.rng, state["rng_state"])
        if self.pending_path.exists():
            pending = json.loads(self.pending_path.read_text())
            # The pending record carries the freshest schedule-time RNG state.
            _restore_rng(self.rng, pending["rng_state"])
            if pending["candidate_id"] <= self.completed_attempts:
                # Settlement is already committed to the journals; finish the
                # cleanup. Save first: losing the pending file before the
                # checkpoint would strand the freshest RNG state.
                self._save_state()
                self.pending_path.unlink()
            elif pending["candidate_id"] != self.completed_attempts + 1:
                raise ValueError("pending candidate is not the next attempt")
            else:
                self.pending = pending

    def _write_summary(self, status, error=None):
        best = self.tree.best() if self.tree.nodes else None
        payload = {"status": status, "method": self.METHOD, "started_at": self.started_at,
                   "finished_at": _timestamp(), "budget": self.budget,
                   "budget_used": self.budget_used, "num_nodes": len(self.tree.nodes),
                   "num_codes": len(self.codebook.entries), "num_roots": len(self.tree.roots),
                   "num_steps": self.step_counter,
                   "best": None if best is None else {"node_id": best.id, "fitness": best.fitness,
                   "idea": best.idea, "code": best.code, "evaluation_id": best.evaluation_id,
                   "operator": best.operator, "parent_id": best.parent_id, "donor_id": best.donor_id}}
        if error:
            payload["error"] = error
        atomic_json(self.summary_path, payload)

    def run(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.state_path.exists():
            self._load_state()
        else:
            self._save_state()
        try:
            while self.budget_used < self.budget:
                self._advance()
                print(f"{self.METHOD}: budget={self.budget_used}/{self.budget} "
                      f"nodes={len(self.tree.nodes)} codes={len(self.codebook.entries)}", flush=True)
            if len(self.tree.roots) < self.n_roots:
                raise RuntimeError("budget exhausted before initialization completed")
            self._write_summary("finished")
        except UnknownEvaluation:
            self._write_summary("blocked", traceback.format_exc())
            raise
        except KeyboardInterrupt:
            self._write_summary("interrupted")
            raise
        except Exception:
            self._write_summary("error", traceback.format_exc())
            raise
