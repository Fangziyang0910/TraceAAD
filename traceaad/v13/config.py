"""V13 policy; evaluation settings remain the shared task protocol."""

from dataclasses import dataclass
from traceaad.config import Config as BaseConfig


@dataclass
class Config(BaseConfig):
    output_tokens: int = 8192
    prompt_tokens: int = 24000
    max_model_calls: int | None = None
    open_floor: float = 0.35
    route_floor: float = 0.25
    recent_window: int = 20
    max_routes: int = 8
    library_capacity: int = 64
    retrieval_size: int = 8
    revision_interval: int = 4
    library_path: str | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.max_model_calls is None:
            self.max_model_calls = 8 * self.budget
        if min(self.prompt_tokens, self.max_model_calls, self.recent_window,
               self.max_routes, self.library_capacity, self.retrieval_size, self.revision_interval) < 1:
            raise ValueError('V13 limits must be positive')
        if not 0 < self.open_floor <= 1 or not 0 < self.route_floor <= 1:
            raise ValueError('exploration floors must be in (0, 1]')
