"""V10.21 mechanism parameters: V10.20's."""

from dataclasses import dataclass

from traceaad.v10_20.config import Config as V1020Config

EXPERIMENT = "traceaad_v10_21"


@dataclass
class Config(V1020Config):
    pass
