"""V10.20 mechanism parameters: V10.17's, with a share for Deepen."""

from dataclasses import dataclass, field

from traceaad.v10_17.config import Config as V1017Config

EXPERIMENT = "traceaad_v10_20"
# Deepen takes 0.15; Refine, Explore and Crossover each give 0.05 of V10.17's 0.45 / 0.30 / 0.25.
OPERATORS = {"Refine": 0.40, "Explore": 0.25, "Crossover": 0.20, "Deepen": 0.15}
ACTIONS = ("Refine", "Explore", "Crossover", "Deepen")


@dataclass
class Config(V1017Config):
    operators: dict[str, float] = field(default_factory=lambda: dict(OPERATORS))

    def __post_init__(self):
        super().__post_init__()
        unknown = set(self.operators) - set(ACTIONS)
        if unknown:
            raise ValueError(f"unknown operators: {sorted(unknown)}")
