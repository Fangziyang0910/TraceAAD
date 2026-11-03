"""Search parameters and execution validation for TraceAAD v15."""

from dataclasses import dataclass, field

from .limits import INSTANCE_SECONDS
from .runtime.evaluate import EVALUATION_SEED
from .instance_evaluation import validate_execution

EXPERIMENT = "traceaad_v15"
DEFAULT_OPERATOR_WEIGHTS = {"Refine": 0.45, "Explore": 0.30, "Crossover": 0.25}
ALLOCATIONS = ("effort", "fitness")
EXPOSURES = ("info", "code")


@dataclass
class Config:
    budget: int = 1000
    roots: int = 5
    init_attempt_limit: int = 16
    history_depth: int = 8
    output_tokens: int = 8192
    evaluation_seeds: tuple[int, ...] = (EVALUATION_SEED,)
    seed: int = 0
    attempts_shown: int = 8
    progress_shown: int = 8
    operators: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_OPERATOR_WEIGHTS))
    eval_timeout_seconds: float = float(INSTANCE_SECONDS)
    eval_workers: int = 1
    scheduler_socket: str | None = None
    revisit: bool = True
    diverge: bool = True
    stagnation_window: int = 40
    mature_steps: int = 4
    map_size: int = 8
    map_distance: float = 0.3
    diverge_shown: int = 4
    allocation: str = "effort"
    exposure: str = "info"

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
        if min(self.stagnation_window, self.map_size) < 1 or min(self.mature_steps, self.diverge_shown) < 0:
            raise ValueError("V14 limits must be positive")
        if not 0 < self.map_distance < 1:
            raise ValueError("map distance must lie in (0, 1)")
        if self.allocation not in ALLOCATIONS or self.exposure not in EXPOSURES:
            raise ValueError(f"allocation must be one of {ALLOCATIONS}, exposure one of {EXPOSURES}")
