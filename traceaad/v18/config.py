"""V18 defaults from the age-matched racing / LDS design specification."""

from dataclasses import dataclass
import math

from traceaad.config import Config as BaseConfig


@dataclass
class Config(BaseConfig):
    history_depth: int = 4
    attempts_shown: int = 3
    first_checkpoint: int = 5
    checkpoint_multiplier: int = 2
    initial_depth_limit: int = 2
    depth_increment: int = 1
    stagnation_window: int = 40
    context_tokens: int = 32768
    same_age: bool = True
    lds: bool = True
    early_first: bool = True
    diverge: bool = True
    objective: str = "minimize"
    max_seconds: float | None = None

    def __post_init__(self):
        super().__post_init__()
        for name in (
            "budget", "roots", "init_attempt_limit", "history_depth", "output_tokens",
            "first_checkpoint", "checkpoint_multiplier", "initial_depth_limit",
            "depth_increment", "stagnation_window", "context_tokens",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.attempts_shown) is not int or self.attempts_shown < 0:
            raise ValueError("attempts_shown must be a nonnegative integer")
        if self.checkpoint_multiplier < 2 or self.initial_depth_limit < 2:
            raise ValueError("checkpoint_multiplier and initial_depth_limit must be at least 2")
        if self.context_tokens <= self.output_tokens:
            raise ValueError("context_tokens must exceed output_tokens")
        if self.objective not in {"minimize", "maximize"}:
            raise ValueError("objective must be minimize or maximize")
        for name in ("same_age", "lds", "early_first", "diverge"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if self.max_seconds is not None and (
            isinstance(self.max_seconds, bool)
            or not math.isfinite(self.max_seconds)
            or self.max_seconds <= 0
        ):
            raise ValueError("max_seconds must be finite and positive")
