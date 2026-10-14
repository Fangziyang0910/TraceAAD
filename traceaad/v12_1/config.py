"""V12.1 parameters: V12.0 with design verification, and completion in place of random development."""

from dataclasses import dataclass

from traceaad.v12_0.config import Config as LinkedConfig

EXPERIMENT = "traceaad_v12_1"


@dataclass
class Config(LinkedConfig):
    # Completion replaces the random development of V10.17.
    development_steps: int = 0
    development_probability: float = 0.0
    # Qwen's itemized containment readings run to several thousand tokens; a cut reply has no judgment.
    verify_output_tokens: int = 8192
    # Completion steps after an exploration's first version, while its latest version does not carry out the Design.
    completion_steps: int = 2
    # Share of explorations that may be completed (the randomized arm uses 0.5).
    completion_probability: float = 1.0
    # "read": complete from what verification read; "resample": implement the proposal again in the start
    # algorithm, as at first (the arm that spends the same candidates on an independent implementation).
    completion: str = "read"

    def __post_init__(self):
        super().__post_init__()
        if self.development_steps or self.development_probability:
            raise ValueError("V12.1 develops an exploration only by completion")
        if self.verify_output_tokens < 1 or self.completion_steps < 0:
            raise ValueError("invalid verification or completion limits")
        if not 0 <= self.completion_probability <= 1:
            raise ValueError("completion probability must be within [0, 1]")
        if self.completion not in ("read", "resample"):
            raise ValueError("completion must be 'read' or 'resample'")


@dataclass
class DrawConfig(Config):
    """Randomized arm: an exploration that needs completion is completed with probability 0.5."""
    completion_probability: float = 0.5


@dataclass
class ResampleConfig(Config):
    """Same trigger and candidates, but each completion is an independent implementation from the start."""
    completion: str = "resample"
