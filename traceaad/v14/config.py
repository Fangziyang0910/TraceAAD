"""V14 policy; the base search loop and shared evaluation constraints are unchanged."""

from dataclasses import dataclass

from traceaad.config import Config as BaseConfig


@dataclass
class Config(BaseConfig):
    output_tokens: int = 8192
    # Reject a reply whose code is identical to an evaluated program and ask once more.
    revisit: bool = True
    # Start a design without parent code after this many attempts without a new best.
    diverge: bool = True
    stagnation_window: int = 40
    # Attempts reserved for developing each new design before it competes (0: none).
    mature_steps: int = 4
    map_size: int = 8
    map_distance: float = 0.3
    diverge_shown: int = 4

    def __post_init__(self):
        super().__post_init__()
        if min(self.stagnation_window, self.map_size) < 1 or min(self.mature_steps, self.diverge_shown) < 0:
            raise ValueError("V14 limits must be positive")
        if not 0 < self.map_distance < 1:
            raise ValueError("map distance must lie in (0, 1)")
