"""EvoTrace search budget and model settings."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    evaluation_budget: int = 1000
    initial_size: int = 5
    temperature: float = 1.0
    max_tokens: int = 16384
    model_timeout: float = 600.0
    seed: int = 0
    failure_count: int = 2
    max_unevaluated: int = 10
