"""TraceAAD V10.4: V10.3 search with design-then-code generation."""

from __future__ import annotations

import json
import time
import traceback
from datetime import datetime
from typing import Any

from llm4ad.method.traceaad_v10_3.traceaad import (
    MAX_CONSECUTIVE_INVALID,
    TraceAADV103,
    _Attempt,
    _strip_thinking,
)

from .prompts import build_code_prompt, build_idea_prompt


class _TwoStageAttempt(_Attempt):
    def __init__(self, operator: str, prompt: str, response: str, llm_seconds: float):
        super().__init__(operator, prompt, response, llm_seconds)
        self.idea_calls = 0
        self.code_calls = 0


class TraceAADV104(TraceAADV103):
    """Keep V10.3 search intact and separate algorithm design from realization."""

    METHOD = "v104"

    def __init__(self, *, idea_output_tokens: int = 1024, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if idea_output_tokens <= 0:
            raise ValueError("idea_output_tokens must be positive")
        if idea_output_tokens >= self.max_context_tokens:
            raise ValueError("idea_output_tokens must be smaller than max_context_tokens")
        self.idea_output_tokens = idea_output_tokens
        self.idea_max_prompt_chars = int(
            (self.max_context_tokens - idea_output_tokens) * 3.5
        )
        self.llm_calls_path = self.run_dir / "llm_calls.jsonl"
        self._pending_idea: dict[str, Any] | None = None

    def _save_state(self) -> None:
        state = {
            "version": 2,
            "started_at": self.started_at,
            "nodes": self.tree.to_state(),
            "rng_state": list(self.rng.getstate()),
            "parent_selection_counts": self.parent_selection_counts,
            "step_counter": self.step_counter,
            "batch_counter": self.step_counter,
            "budget_used": self.budget_used,
            "pending_idea": self._pending_idea,
        }
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.state_path)

    def _load_state(self) -> None:
        super()._load_state()
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        self._pending_idea = state.get("pending_idea")

    def _log_llm_call(
        self,
        *,
        stage: str,
        operator: str,
        parent_id: int | None,
        donor_id: int | None,
        prompt: str,
        response: str | None,
        seconds: float,
        error: str | None = None,
    ) -> None:
        payload = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "step": self.step_counter,
            "stage": stage,
            "operator": operator,
            "parent_id": parent_id,
            "donor_id": donor_id,
            "prompt": prompt,
            "response": response,
            "seconds": seconds,
        }
        if error is not None:
            payload["error"] = error
        self._append_jsonl(self.llm_calls_path, payload)

    def _draw(
        self,
        *,
        stage: str,
        operator: str,
        parent_id: int | None,
        donor_id: int | None,
        prompt: str,
        max_tokens: int,
    ) -> tuple[str, float]:
        started = time.time()
        try:
            response = self.llm.draw_sample(prompt, max_tokens=max_tokens)
        except Exception:
            seconds = time.time() - started
            self._log_llm_call(
                stage=stage,
                operator=operator,
                parent_id=parent_id,
                donor_id=donor_id,
                prompt=prompt,
                response=None,
                seconds=seconds,
                error=traceback.format_exc(),
            )
            raise
        seconds = time.time() - started
        self._log_llm_call(
            stage=stage,
            operator=operator,
            parent_id=parent_id,
            donor_id=donor_id,
            prompt=prompt,
            response=response,
            seconds=seconds,
        )
        return response, seconds

    @staticmethod
    def _idea_from_response(response: str) -> str | None:
        idea = _strip_thinking(response).strip()
        return idea or None

    def _idea_prompt(self, current, ancestors, operator, donor) -> str:
        prompt = build_idea_prompt(
            task_contract=self.task_contract,
            current=current,
            ancestors=ancestors,
            operator=operator,
            donor=donor,
            max_events=self.traj_gens,
        )
        if len(prompt) > self.idea_max_prompt_chars:
            raise ValueError(
                "idea generation state exceeds context budget: "
                f"{len(prompt)} > {self.idea_max_prompt_chars} chars"
            )
        return prompt

    def _code_prompt(self, current, donor, idea: str) -> str:
        prompt = build_code_prompt(
            task_contract=self.task_contract,
            current=current,
            donor=donor,
            idea=idea,
        )
        if len(prompt) > self.max_prompt_chars:
            raise ValueError(
                "code generation state exceeds context budget: "
                f"{len(prompt)} > {self.max_prompt_chars} chars"
            )
        return prompt

    def _register_invalid(self) -> None:
        self._invalid_streak += 1
        if self._invalid_streak > MAX_CONSECUTIVE_INVALID:
            raise RuntimeError(
                f"{MAX_CONSECUTIVE_INVALID} consecutive generations produced no "
                "valid output; the serving backend is misbehaving"
            )

    def _generate(self, operator, current, ancestors, donor) -> _TwoStageAttempt:
        parent_id = None if current is None else current.id
        donor_id = None if donor is None else donor.id
        pending = self._pending_idea

        if pending is None:
            idea_prompt = self._idea_prompt(current, ancestors, operator, donor)
            idea_response, idea_seconds = self._draw(
                stage="idea",
                operator=operator,
                parent_id=parent_id,
                donor_id=donor_id,
                prompt=idea_prompt,
                max_tokens=self.idea_output_tokens,
            )
            attempt = _TwoStageAttempt(
                operator, idea_prompt, idea_response, idea_seconds
            )
            attempt.idea_calls = 1
            idea = self._idea_from_response(idea_response)
            attempt.idea = idea
            if idea is None:
                attempt.reason = "empty idea response"
                self._register_invalid()
                return attempt
            self._pending_idea = {
                "operator": operator,
                "parent_id": parent_id,
                "donor_id": donor_id,
                "idea": idea,
                "idea_prompt": idea_prompt,
                "idea_response": idea_response,
                "idea_seconds": idea_seconds,
            }
            self._save_state()
        else:
            expected = (operator, parent_id, donor_id)
            actual = (
                pending["operator"],
                pending["parent_id"],
                pending["donor_id"],
            )
            if actual != expected:
                raise RuntimeError(
                    f"pending idea state {actual!r} does not match generation {expected!r}"
                )
            idea = pending["idea"]
            attempt = _TwoStageAttempt(
                operator,
                pending["idea_prompt"],
                pending["idea_response"],
                float(pending.get("idea_seconds", 0.0)),
            )
            attempt.idea = idea

        code_prompt = self._code_prompt(current, donor, idea)
        code_response, code_seconds = self._draw(
            stage="code",
            operator=operator,
            parent_id=parent_id,
            donor_id=donor_id,
            prompt=code_prompt,
            max_tokens=self.output_tokens,
        )
        attempt.code_calls = 1
        attempt.llm_seconds += code_seconds
        attempt.response = code_response
        parsed = super().parse_response(
            f"Latest Design Idea: staged design\n{code_response}"
        )
        self._pending_idea = None
        if parsed is None:
            attempt.reason = "code unparseable or function signature mismatch"
            self._register_invalid()
            return attempt
        _, attempt.code, attempt.program = parsed
        self._invalid_streak = 0
        return attempt

    def _expand_parent(self) -> None:
        if self._pending_idea is None:
            super()._expand_parent()
            return

        pending = self._pending_idea
        parent_id = pending["parent_id"]
        if parent_id is None:
            raise RuntimeError("initialization pending idea reached expansion phase")
        parent = self.tree.nodes[parent_id]
        donor_id = pending["donor_id"]
        donor = None if donor_id is None else self.tree.nodes[donor_id]
        attempt = self._generate(pending["operator"], parent, [], donor)
        if attempt.program is not None:
            self._evaluate(attempt)

        self.step_counter += 1
        if attempt.program is not None and attempt.fitness is not None:
            node = self.tree.add(
                code=attempt.code,
                idea=attempt.idea,
                fitness=attempt.fitness,
                evaluation_id=attempt.evaluation_id,
                parent_id=parent.id,
                operator=attempt.operator,
                donor_id=donor_id,
            )
            attempt.node_id = node.id
        self._log_attempt(attempt, parent_id=parent.id, donor_id=donor_id)
        self._save_state()

    def _log_attempt(self, attempt, *, parent_id, donor_id) -> None:
        payload = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "step": self.step_counter,
            "operator": attempt.operator,
            "parent_id": parent_id,
            "donor_id": donor_id,
            "status": attempt.status,
            "fitness": attempt.fitness,
            "node_id": attempt.node_id,
            "reason": attempt.reason,
        }
        if isinstance(attempt, _TwoStageAttempt):
            payload.update(
                idea_calls=attempt.idea_calls,
                code_calls=attempt.code_calls,
            )
        self._append_jsonl(self.events_path, payload)
