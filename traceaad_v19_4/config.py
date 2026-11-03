"""Search settings for trajectory context, shared experience and node response."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Config:
    evaluation_budget: int = 1000
    initial_size: int = 5
    seed: int = 0
    temperature: float = 1.0
    max_tokens: int = 16384
    model_timeout: float = 600.0
    parent_floor: float = 0.05
    selection_mode: str = "trajectory"
    candidate_count: int = 4
    response_window: int = 256
    response_strength: float = 2.0
    generation_history: bool = True
    history_steps: int = 3
    alternative_count: int = 2
    experience_k: int = 3
    long_term_interval: int = 10
    long_term_window: int = 30
    tokenizer: str = "deepseek-ai/DeepSeek-V4.1-Flash"
    reflection_mode: str = "trajectory"
    max_unevaluated: int = 10

    def __post_init__(self):
        for name in ("evaluation_budget", "initial_size", "max_tokens",
                     "history_steps", "candidate_count", "long_term_interval", "long_term_window", "max_unevaluated",
                     "response_window"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("alternative_count", "experience_k"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if not math.isfinite(self.model_timeout) or self.model_timeout <= 0:
            raise ValueError("model_timeout must be finite and positive")
        if not math.isfinite(self.parent_floor) or not 0 <= self.parent_floor < 1:
            raise ValueError("parent_floor must be in [0, 1)")
        if self.selection_mode not in {"trajectory", "rank", "direct", "neighborhood"}:
            raise ValueError("selection_mode must be trajectory, rank, direct or neighborhood")
        if (isinstance(self.response_strength, bool) or not math.isfinite(self.response_strength)
                or self.response_strength <= 0):
            raise ValueError("response_strength must be finite and positive")
        if type(self.generation_history) is not bool:
            raise ValueError("generation_history must be boolean")
        if self.reflection_mode not in {"trajectory", "local", "siblings", "none"}:
            raise ValueError("reflection_mode must be trajectory, local, siblings or none")
        if not isinstance(self.tokenizer, str) or not self.tokenizer:
            raise ValueError("tokenizer must name a local tokenizer or a cached repository")
