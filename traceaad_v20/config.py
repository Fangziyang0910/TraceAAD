"""V20 settings: Base settings plus the switches for its three mechanisms."""

from dataclasses import dataclass

from traceaad_base.config import Config as BaseConfig


@dataclass(frozen=True)
class Config(BaseConfig):
    trajectory_scoring: bool = True
    trajectory_context: bool = True
    reference_selection: bool = True
    experience_learning: bool = True
