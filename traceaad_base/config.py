"""Search settings for TraceAAD Base."""

from dataclasses import dataclass


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
