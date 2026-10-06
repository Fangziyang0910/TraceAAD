"""V10.15 mechanism parameters."""

from dataclasses import dataclass, field
from traceaad.common.config import SearchConfig, REVISION

EXPERIMENT = "traceaad_v10_15"
OPERATORS = {"Refine": 0.45, "Explore": 0.3, "Crossover": 0.25}


@dataclass
class Config(SearchConfig):
    explore_cards: int = 0
    operators: dict[str, float] = field(default_factory=lambda: dict(OPERATORS))

    def __post_init__(self):
        super().__post_init__()
        if not self.operators or min(self.operators.values()) < 0 or sum(self.operators.values()) <= 0:
            raise ValueError("operator weights must be nonnegative with a positive total")
        if self.explore_cards < 0:
            raise ValueError("explore_cards must be nonnegative")
