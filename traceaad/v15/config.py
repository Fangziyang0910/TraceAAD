"""V15 policy: V14 plus an exposure policy; the base loop and shared evaluation constraints are unchanged."""

from dataclasses import dataclass

from traceaad.v14.config import Config as V14Config

ALLOCATIONS = ("effort", "fitness")
EXPOSURES = ("info", "code")


@dataclass
class Config(V14Config):
    # Who is exposed. "effort": attempts go to lineages, each compared with the
    # champion lineage after the same number of attempts; "fitness": V14.
    allocation: str = "effort"
    # How other programs are exposed. "info": only the program being edited is
    # shown as code, other programs by Design and outcome; "code": V14.
    exposure: str = "info"

    def __post_init__(self):
        super().__post_init__()
        if self.allocation not in ALLOCATIONS or self.exposure not in EXPOSURES:
            raise ValueError(f"allocation must be one of {ALLOCATIONS}, exposure one of {EXPOSURES}")
