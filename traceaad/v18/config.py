"""Search parameters and execution validation for TraceAAD v18."""

from dataclasses import dataclass, field
import math

from .limits import INSTANCE_SECONDS
from .runtime.evaluate import EVALUATION_SEED
from .instance_evaluation import validate_execution

EXPERIMENT = "traceaad_v18"
DEFAULT_OPERATOR_WEIGHTS = {"Refine": 0.45, "Explore": 0.30, "Crossover": 0.25}


@dataclass
class Config:
    budget: int = 1000
    roots: int = 5
    init_attempt_limit: int = 16
    history_depth: int = 4
    output_tokens: int = 16384
    evaluation_seeds: tuple[int, ...] = (EVALUATION_SEED,)
    seed: int = 0
    attempts_shown: int = 3
    progress_shown: int = 8
    operators: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_OPERATOR_WEIGHTS))
    eval_timeout_seconds: float = float(INSTANCE_SECONDS)
    eval_workers: int = 1
    scheduler_socket: str | None = None
    max_no_change: int = 3
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
        if (
            min(
                self.budget,
                self.roots,
                self.init_attempt_limit,
                self.history_depth,
                self.output_tokens,
            )
            < 1
        ):
            raise ValueError("search limits must be positive")
        if not self.evaluation_seeds or any(
            type(s) is not int or not 0 <= s < 2**32 for s in self.evaluation_seeds
        ):
            raise ValueError("evaluation seeds must be nonempty uint32 integers")
        if (
            not self.operators
            or min(self.operators.values()) < 0
            or sum(self.operators.values()) <= 0
        ):
            raise ValueError("operator weights must be nonnegative with a positive total")
        unknown = set(self.operators) - set(DEFAULT_OPERATOR_WEIGHTS)
        if unknown:
            raise ValueError(f"unknown operators: {sorted(unknown)}")
        validate_execution(self.eval_timeout_seconds, self.eval_workers)
        for name in (
            "budget", "roots", "init_attempt_limit", "history_depth", "output_tokens",
            "first_checkpoint", "checkpoint_multiplier", "initial_depth_limit",
            "depth_increment", "stagnation_window", "context_tokens", "max_no_change",
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
