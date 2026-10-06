"""V10.18 mechanism parameters."""

from dataclasses import dataclass
from traceaad.common.config import SearchConfig, REVISION

EXPERIMENT = "traceaad_v10_18"
OPERATORS = {"Refine": 0.35, "Explore": 0.4, "Crossover": 0.25}


@dataclass
class Config(SearchConfig):
    attempts_shown: int = 8
    progress_shown: int = 8
    development_min: int = 2
    development_stall: int = 2
    development_max: int = 8
    refine_share: float = OPERATORS["Refine"]
    explore_share: float = OPERATORS["Explore"]
    crossover_share: float = OPERATORS["Crossover"]

    @property
    def operators(self):
        return {"Refine": self.refine_share, "Explore": self.explore_share, "Crossover": self.crossover_share}

    def __post_init__(self):
        super().__post_init__()
        if not self.operators or min(self.operators.values()) < 0 or sum(self.operators.values()) <= 0:
            raise ValueError("operator weights must be nonnegative with a positive total")
        if not 0 <= self.development_min <= self.development_max or self.development_stall < 1:
            raise ValueError("invalid development limits")
