"""Fixed defaults for the independently implemented TraceAAD Base."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Config:
    evaluation_budget: int = 1000
    initial_size: int = 5
    temperature: float = 1.0
    max_tokens: int = 16384
    model_timeout: float = 600.0
    seed: int = 0
    ancestor_count: int = 8
    offspring_count: int = 3
    failure_count: int = 2
    max_unevaluated: int = 10
    refine_weight: float = 0.45
    explore_weight: float = 0.30
    crossover_weight: float = 0.25

    def __post_init__(self):
        for name in ("evaluation_budget", "initial_size", "max_tokens", "max_unevaluated"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("ancestor_count", "offspring_count", "failure_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        for name in ("temperature", "refine_weight", "explore_weight", "crossover_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(self.model_timeout) or self.model_timeout <= 0:
            raise ValueError("model_timeout must be finite and positive")
        if self.refine_weight + self.explore_weight <= 0:
            raise ValueError("Refine or Explore must have positive weight")

