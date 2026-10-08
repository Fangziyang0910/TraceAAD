"""V10.21 search parameters: V10.17's three steps, plus per-call instance execution defaults."""

from dataclasses import dataclass, field

from benchmarks.tasks import INSTANCE_SECONDS
from traceaad.v10_17.config import Config as V1017Config
from traceaad.common.instance_evaluation import validate_execution

EXPERIMENT = "traceaad_v10_21"
# No step prescribes adding search: the model decides how to improve the algorithm.
OPERATORS = {"Refine": 0.45, "Explore": 0.30, "Crossover": 0.25}


@dataclass
class Config(V1017Config):
    operators: dict[str, float] = field(default_factory=lambda: dict(OPERATORS))
    eval_timeout_seconds: float = float(INSTANCE_SECONDS)
    eval_workers: int = 1
    scheduler_socket: str | None = None

    def __post_init__(self):
        super().__post_init__()
        unknown = set(self.operators) - set(OPERATORS)
        if unknown:
            raise ValueError(f"unknown operators: {sorted(unknown)}")
        validate_execution(self.eval_timeout_seconds, self.eval_workers)
