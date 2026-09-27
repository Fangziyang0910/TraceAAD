"""Frozen policy choices. Budgets count attempted candidates, not valid roots."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Config:
    budget: int = 1000
    max_evaluations: int = 1000
    init_proposals: int = 8
    init_mode: str = "hybrid"
    regions: int = 8
    trial_fraction: float = .2
    recheck_fraction: float = .1
    trial_length: int = 3
    delta: float = 1e-6
    challenger_gap: float = .1
    min_behavior_distance: float = .01
    output_tokens: int = 8192
    max_input_tokens: int = 24320
    evidence_tokens: int = 4000
    max_events: int = 6
    history_depth: int = 6
    output_mode: str = "full"
    evaluation_seeds: tuple[int, ...] = (730241,)
    comparison_tolerance: float = 1e-6
    final_candidates: int = 5
    max_total_tokens: int | None = None
    max_seconds: float | None = None
    evidence_policy: str = "conditional"
    comparison_feedback: bool = True
    seed: int = 0

    def __post_init__(self):
        for field in ("budget", "max_evaluations", "init_proposals", "regions",
                      "trial_length", "output_tokens", "max_input_tokens",
                      "evidence_tokens", "max_events", "history_depth", "final_candidates"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if self.regions > 8 or self.final_candidates > 5 or self.trial_length > 3:
            raise ValueError("V10.14 supports at most 8 regions, 5 finalists and 3 trial steps")
        if not (0 <= self.trial_fraction <= .2 and 0 <= self.recheck_fraction <= .1):
            raise ValueError("trial/recheck fractions exceed V10.14 caps")
        for field in ("delta", "challenger_gap", "min_behavior_distance", "comparison_tolerance"):
            if not math.isfinite(getattr(self, field)) or getattr(self, field) < 0:
                raise ValueError(f"{field} must be finite and nonnegative")
        if self.delta == 0:
            raise ValueError("delta must be positive")
        if self.init_mode not in {"independent", "hybrid", "sequential"}:
            raise ValueError("unknown initialization mode")
        if self.output_mode not in {"full", "edit"}:
            raise ValueError("output_mode must be full or edit")
        if self.evidence_policy not in {"none", "trajectory", "bag", "conditional"}:
            raise ValueError("unknown evidence policy")
        if not self.evaluation_seeds or len(set(self.evaluation_seeds)) != len(self.evaluation_seeds):
            raise ValueError("evaluation seeds must be nonempty and unique")
        if any(not isinstance(s, int) or not 0 <= s < 2**32 for s in self.evaluation_seeds):
            raise ValueError("evaluation seeds must be uint32 integers")
        if self.max_total_tokens is not None and self.max_total_tokens < 1:
            raise ValueError("max_total_tokens must be positive")
        if self.max_seconds is not None and (not math.isfinite(self.max_seconds) or self.max_seconds <= 0):
            raise ValueError("max_seconds must be finite and positive")

    @property
    def root_attempts(self):
        # Small runs reserve room for bootstrap/development; no retry-to-N roots.
        return min(self.init_proposals, max(1, self.budget // 2))
