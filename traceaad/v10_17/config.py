"""V10.17 mechanism parameters."""

from dataclasses import dataclass, field
from traceaad.common.config import SearchConfig, REVISION

EXPERIMENT = "traceaad_v10_17"
OPERATORS = {"Refine": 0.45, "Explore": 0.3, "Crossover": 0.25}


@dataclass
class Config(SearchConfig):
    attempts_shown: int = 8
    progress_shown: int = 8
    development_steps: int = 3
    development_probability: float = 0.125
    operators: dict[str, float] = field(default_factory=lambda: dict(OPERATORS))

    def __post_init__(self):
        super().__post_init__()
        if not self.operators or min(self.operators.values()) < 0 or sum(self.operators.values()) <= 0:
            raise ValueError("operator weights must be nonnegative with a positive total")
        if not 0 <= self.development_probability <= 1 or self.development_steps < 0:
            raise ValueError("invalid development probability or steps")
