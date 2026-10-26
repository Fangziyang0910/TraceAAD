"""V13.1 fixed policy; shared evaluation constraints are inherited unchanged."""
from dataclasses import dataclass
import math
from traceaad.config import Config as BaseConfig


@dataclass
class Config(BaseConfig):
    output_tokens: int = 8192
    prompt_tokens: int = 24000
    max_model_calls: int | None = None
    warmup: int = 16
    free_per_ten: int = 2
    diagnostic_per_ten: int = 1
    diagnostic_threshold: float = 1e-4
    lineage_decay: float = .5
    improvement_delta: float = 1e-9
    match_batch: int = 4
    max_conditions: int = 3
    library_path: str | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.max_model_calls is None:
            self.max_model_calls = 8 * self.budget
        if min(self.prompt_tokens, self.max_model_calls, self.warmup, self.match_batch) < 1:
            raise ValueError('V13.1 limits must be positive')
        if not (1 <= self.free_per_ten <= 10 and 0 <= self.diagnostic_per_ten <= 10-self.free_per_ten
                and 0 <= self.max_conditions <= 3 and 0 < self.lineage_decay < 1
                and all(math.isfinite(x) and x >= 0 for x in
                        (self.diagnostic_threshold, self.improvement_delta))):
            raise ValueError('invalid V13.1 policy')
