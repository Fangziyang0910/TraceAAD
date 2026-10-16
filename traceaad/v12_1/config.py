"""V12.1 search and execution parameters."""

from dataclasses import dataclass, field

REVISION = "research-protocol-v6-20261012-raw-objective"


from benchmarks.tasks import INSTANCE_SECONDS
from .instance_evaluation import validate_execution

EXPERIMENT = "traceaad_v12_1"
INDIVIDUAL_FORMAT = "program-individual-v1"
OPERATORS = {"Refine": 0.45, "Explore": 0.30, "Crossover": 0.25}


@dataclass
class Config:
    budget: int = 1000
    roots: int = 5
    init_attempt_limit: int = 16
    history_depth: int = 8
    output_tokens: int = 16384
    final_candidates: int = 5
    evaluation_seeds: tuple[int, ...] = (730241,)
    seed: int = 0

    attempts_shown: int = 8
    progress_shown: int = 8
    operators: dict[str, float] = field(default_factory=lambda: dict(OPERATORS))
    eval_timeout_seconds: float = float(INSTANCE_SECONDS)
    eval_workers: int = 1
    scheduler_socket: str | None = None
    linking: bool = True
    # Nearest earlier Designs shown to the match step; the model judges which share the computation.
    match_candidates: int = 20
    # Share of linked proposals whose implementation step shows the earlier program (diagnostic batches < 1).
    display_probability: float = 1.0

    # Completion steps after an exploration's first version, while its latest version does not carry out the Design.
    completion_steps: int = 2
    # Share of explorations that may be completed (the randomized arm uses 0.5).
    completion_probability: float = 1.0
    # "read": complete from what verification read; "resample": implement the proposal again in the start
    # algorithm, as at first (the arm that spends the same candidates on an independent implementation).
    completion: str = "read"

    def __post_init__(self):
        if (
            min(
                self.budget,
                self.roots,
                self.init_attempt_limit,
                self.history_depth,
                self.output_tokens,
                self.final_candidates,
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
            raise ValueError(
                "operator weights must be nonnegative with a positive total"
            )
        unknown = set(self.operators) - set(OPERATORS)
        if unknown:
            raise ValueError(f"unknown operators: {sorted(unknown)}")
        validate_execution(self.eval_timeout_seconds, self.eval_workers)
        if self.match_candidates < 1:
            raise ValueError("match limits must be positive")
        if not 0 <= self.display_probability <= 1:
            raise ValueError("display probability must be within [0, 1]")

        if self.completion_steps < 0:
            raise ValueError("invalid completion steps")
        if not 0 <= self.completion_probability <= 1:
            raise ValueError("completion probability must be within [0, 1]")
        if self.completion not in ("read", "resample"):
            raise ValueError("completion must be 'read' or 'resample'")


@dataclass
class DrawConfig(Config):
    completion_probability: float = 0.5


@dataclass
class ResampleConfig(Config):
    completion: str = "resample"
