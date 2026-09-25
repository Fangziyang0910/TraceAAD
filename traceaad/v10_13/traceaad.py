"""TraceAAD V10.13: compact evolutionary search over generated programs."""

from __future__ import annotations

import math
import random
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from core import SecureEvaluator

from .parsing import parse_candidate, template_target
from .prompts import PromptBuilder
from .selection import OPERATORS, OPERATOR_PROBABILITIES, sample_parent, sample_reference
from .storage import RunStorage, read_journal
from .tree import Node, SearchTree


def _timestamp():
    return datetime.now().isoformat(timespec="seconds")


def _restore_rng(rng, state):
    rng.setstate((state[0], tuple(state[1]), state[2]))


@dataclass
class Candidate:
    id: int
    prompt: str
    operator: str
    parent: Node | None
    reference: Node | None


class TraceAADV1013:
    """Generate, evaluate, select, and persist V10.13 candidates."""

    METHOD = "v1013"

    def __init__(self, *, evaluation, llm, run_dir, budget=1000, n_roots=8,
                 history_depth=3, output_tokens=8192, max_input_tokens=24320,
                 seed=0):
        if budget < n_roots or n_roots < 1 or history_depth < 0:
            raise ValueError("invalid budget, root, or history settings")
        if output_tokens < 1 or max_input_tokens < 1:
            raise ValueError("token limits must be positive")

        self.llm = llm
        self.run_dir = Path(run_dir)
        self.budget = budget
        self.n_roots = n_roots
        self.output_tokens = output_tokens
        self.evaluator = SecureEvaluator(evaluation)
        self.template_program = evaluation.template_program
        self.interface, target_stub = template_target(self.template_program)
        self.task_prompt = (
            "# Task\n\n" + evaluation.task_description.strip() +
            "\n\nImplement the target function below. Its body and supporting code are "
            "the design space.\n\n```python\n" + target_stub + "\n```"
        )

        self.tree = SearchTree()
        self.rng = random.Random(seed)
        self.started_at = _timestamp()
        self.candidate_count = 0
        self.evaluations_used = 0
        self.invalid_streak = 0
        self.storage = RunStorage(self.run_dir)
        self.prompts = PromptBuilder(
            llm,
            self.task_prompt,
            max_tokens=max_input_tokens,
            history_depth=history_depth,
            lookup=self.tree.nodes.get,
            all_nodes=self.tree.all_nodes,
        )

    def _schedule(self):
        candidate_id = self.candidate_count + 1
        if len(self.tree.roots) < self.n_roots:
            prompt = self.prompts.build_initial()
            return Candidate(candidate_id, prompt.prompt, "Init", None, None)

        operator = self.rng.choices(
            OPERATORS, weights=OPERATOR_PROBABILITIES.values(),
        )[0]
        parent = sample_parent(self.tree.all_nodes(), self.rng)
        reference = sample_reference(self.tree.all_nodes(), parent, self.rng) if operator == "Fuse" else None
        if operator == "Fuse" and reference is None:
            operator = "Refine"
        prompt = self.prompts.build_development(parent, operator, reference)
        if operator == "Fuse" and prompt.reference is None:
            operator = "Refine"
            prompt = self.prompts.build_development(parent, operator)
        reference = prompt.reference
        return Candidate(
            candidate_id,
            prompt.prompt,
            operator,
            parent,
            reference,
        )

    def _generate(self, candidate):
        started = time.time()
        record = {
            "candidate_id": candidate.id,
            "prompt": candidate.prompt,
        }
        try:
            details = self.llm.draw_sample_with_details(
                candidate.prompt, max_tokens=self.output_tokens,
            )
        except Exception:
            record["error"] = traceback.format_exc()
            self.storage.record_call(record)
            raise
        record.update(
            response=details["content"],
            finish_reason=details.get("finish_reason") or "unknown",
            usage=details.get("usage"),
            model=details.get("model"),
            seconds=time.time() - started,
        )
        self.storage.record_call(record)
        return record

    @staticmethod
    def _fitness(result):
        if result.result is None:
            return None, result.failure_kind, result.error
        try:
            value = float(result.result)
        except (TypeError, ValueError, OverflowError) as exc:
            return None, "invalid_result", str(exc)
        if not math.isfinite(value):
            return None, "nonfinite_fitness", "nonfinite fitness"
        return value, result.failure_kind, result.error

    def _evaluate(self, parsed, candidate):
        self.evaluations_used += 1
        started = time.time()
        try:
            fitness, reason, error = self._fitness(
                self.evaluator.evaluate_program_with_details(parsed.program_code),
            )
        except Exception as exc:
            fitness, reason, error = None, "evaluation_error", str(exc)

        node = None
        if fitness is not None:
            node = self.tree.add(
                code=parsed.program_code,
                idea=parsed.idea,
                fitness=fitness,
                parent_id=candidate.parent.id if candidate.parent else None,
                operator=candidate.operator,
                reference_id=candidate.reference.id if candidate.reference else None,
            )
            self.storage.record_node(node)
        return node, reason, error, time.time() - started

    def _run_candidate(self):
        candidate = self._schedule()
        completion = self._generate(candidate)
        parsed, parse_error = parse_candidate(
            completion["response"],
            completion["finish_reason"],
            self.interface,
            self.template_program,
            base_code=candidate.parent.code if candidate.parent else None,
        )

        node = None
        eval_seconds = None
        if parsed is None:
            self.invalid_streak += 1
            status, reason, error = "invalid_output", parse_error, None
        else:
            self.invalid_streak = 0
            node, reason, error, eval_seconds = self._evaluate(parsed, candidate)
            status = "ok" if node else "eval_failed"

        self.candidate_count = candidate.id
        event = {
            "ts": _timestamp(),
            "candidate_id": candidate.id,
            "evaluation_id": self.evaluations_used if parsed else None,
            "budget_used": self.evaluations_used,
            "operator": candidate.operator,
            "parent_id": candidate.parent.id if candidate.parent else None,
            "reference_id": candidate.reference.id if candidate.reference else None,
            "status": status,
            "reason": reason,
            "error": error,
            "node_id": node.id if node else None,
            "fitness": node.fitness if node else None,
            "eval_seconds": eval_seconds,
        }
        self.storage.record_event(event)
        self._save_state()
        if self.invalid_streak >= 50:
            raise RuntimeError("50 consecutive generations produced no valid output")

    def _save_state(self):
        self.storage.save_state({
            "started_at": self.started_at,
            "rng_state": list(self.rng.getstate()),
            "candidate_count": self.candidate_count,
            "budget_used": self.evaluations_used,
            "invalid_streak": self.invalid_streak,
        })

    def _resume(self):
        import json

        state = json.loads(self.storage.state_path.read_text(encoding="utf-8"))
        self.started_at = state.get("started_at", self.started_at)
        if state.get("rng_state"):
            _restore_rng(self.rng, state["rng_state"])
        for entry in read_journal(self.storage.nodes_path):
            self.tree.add_raw(Node(
                id=entry["id"],
                code=entry["code"],
                idea=entry.get("idea", ""),
                fitness=entry["fitness"],
                parent_id=entry.get("parent_id"),
                operator=entry.get("operator", "Init"),
                reference_id=entry.get("reference_id"),
            ))
        self.candidate_count = state.get("candidate_count", 0)
        self.evaluations_used = state.get("budget_used", 0)
        self.invalid_streak = state.get("invalid_streak", 0)

    def _write_summary(self, status, error=None):
        best = self.tree.best() if self.tree.nodes else None
        summary = {
            "status": status,
            "method": self.METHOD,
            "started_at": self.started_at,
            "finished_at": _timestamp(),
            "budget": self.budget,
            "budget_used": self.evaluations_used,
            "num_nodes": len(self.tree.nodes),
            "num_roots": len(self.tree.roots),
            "best": asdict(best) if best else None,
        }
        if error:
            summary["error"] = error
        self.storage.save_summary(summary)

    def run(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.storage.state_path.exists():
            self._resume()
        else:
            self._save_state()
        try:
            while self.evaluations_used < self.budget:
                self._run_candidate()
                print(
                    f"{self.METHOD}: budget={self.evaluations_used}/{self.budget} "
                    f"nodes={len(self.tree.nodes)}",
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
