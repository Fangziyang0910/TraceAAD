"""V12.0 parameters: search, per-instance execution and re-proposal linking in Explore."""

from dataclasses import dataclass, field

from benchmarks.tasks import INSTANCE_SECONDS
from traceaad.common.config import SearchConfig
from traceaad.common.instance_evaluation import validate_execution

EXPERIMENT = "traceaad_v12_0"
OPERATORS = {"Refine": 0.45, "Explore": 0.30, "Crossover": 0.25}


@dataclass
class Config(SearchConfig):
    attempts_shown: int = 8
    progress_shown: int = 8
    development_steps: int = 3
    development_probability: float = 0.125
    operators: dict[str, float] = field(default_factory=lambda: dict(OPERATORS))
    eval_timeout_seconds: float = float(INSTANCE_SECONDS)
    eval_workers: int = 1
    scheduler_socket: str | None = None
    linking: bool = True
    # Nearest earlier Designs shown to the match step; the model judges which share the computation.
    match_candidates: int = 20
    match_output_tokens: int = 1024
    # Share of linked proposals whose implementation step shows the earlier program (diagnostic batches < 1).
    display_probability: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if not self.operators or min(self.operators.values()) < 0 or sum(self.operators.values()) <= 0:
            raise ValueError("operator weights must be nonnegative with a positive total")
        if not 0 <= self.development_probability <= 1 or self.development_steps < 0:
            raise ValueError("invalid development probability or steps")
        unknown = set(self.operators) - set(OPERATORS)
        if unknown:
            raise ValueError(f"unknown operators: {sorted(unknown)}")
        validate_execution(self.eval_timeout_seconds, self.eval_workers)
        if self.match_candidates < 1 or self.match_output_tokens < 1:
            raise ValueError("match limits must be positive")
        if not 0 <= self.display_probability <= 1:
            raise ValueError("display probability must be within [0, 1]")


@dataclass
class BaseConfig(Config):
    """The same search without linking: the base arm, and the runs E0 replays."""
    linking: bool = False


@dataclass
class DisplayConfig(Config):
    """Diagnostic arm: a linked proposal sees the earlier program at random, half of the time."""
    display_probability: float = 0.5
