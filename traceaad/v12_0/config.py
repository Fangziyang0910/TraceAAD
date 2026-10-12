"""V12.0 parameters: V10.17 search under the current protocol, plus re-proposal linking in Explore."""

from dataclasses import dataclass

from traceaad.v10_21.config import Config as V1021Config

EXPERIMENT = "traceaad_v12_0"


@dataclass
class Config(V1021Config):
    linking: bool = True
    # Nearest earlier Designs shown to the match step; the model judges which share the computation.
    match_candidates: int = 20
    match_output_tokens: int = 1024
    # Share of linked proposals whose implementation step shows the earlier program (diagnostic batches < 1).
    display_probability: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if self.match_candidates < 1 or self.match_output_tokens < 1:
            raise ValueError("match limits must be positive")
        if not 0 <= self.display_probability <= 1:
            raise ValueError("display probability must be within [0, 1]")


@dataclass
class BaseConfig(Config):
    """The same search without linking: the base arm, and the runs E0 replays."""
    linking: bool = False


@dataclass
class DisplayConfig(Config):
    """Diagnostic arm: a linked proposal sees the earlier program at random, half of the time."""
    display_probability: float = 0.5
