"""Search settings for trajectory reflection with shared design experience."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Config:
    evaluation_budget: int = 1000
    initial_size: int = 5
    seed: int = 0
    temperature: float = 1.0
    max_tokens: int = 16384
    reflection_max_tokens: int = 2048
    model_timeout: float = 600.0
    parent_floor: float = 0.05
    history_steps: int = 3
    alternative_count: int = 2
    experience_k: int = 3
    max_lessons: int = 3
    material_tokens: int = 8192
    context_window: int = 1048576
    protocol_tokens: int = 64
    tokenizer: str = "deepseek-ai/DeepSeek-V4.1-Flash"
    reflection_mode: str = "trajectory"
    max_unevaluated: int = 10

    def __post_init__(self):
        for name in ("evaluation_budget", "initial_size", "max_tokens", "reflection_max_tokens",
                     "history_steps", "max_lessons", "material_tokens", "context_window", "max_unevaluated"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("alternative_count", "experience_k", "protocol_tokens"):
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
        if self.reflection_mode not in {"trajectory", "local", "siblings", "none"}:
            raise ValueError("reflection_mode must be trajectory, local, siblings or none")
        if not isinstance(self.tokenizer, str) or not self.tokenizer:
            raise ValueError("tokenizer must name a local tokenizer or a cached repository")
        if max(self.max_tokens, self.reflection_max_tokens) + self.protocol_tokens >= self.context_window:
            raise ValueError("The context window must leave room for model input")
