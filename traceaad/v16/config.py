"""V16 policy: V15 plus an optimistic estimate at each checkpoint and a context of changes; the base loop and shared evaluation constraints are unchanged."""

from dataclasses import dataclass

from traceaad.v15.config import Config as V15Config


@dataclass
class Config(V15Config):
    # Also keep a challenger whose recent improvement per attempt, continued for as
    # many attempts again, reaches the champion lineage at twice the checkpoint; False: V15.
    projection: bool = True
    # Steps and attempts of the edited lineage are shown by the Change each program
    # stated, and listed attempts are not to be repeated; False: by Design, as V15.
    changes: bool = True
    # A Crossover reference is chosen among programs of other lineages; False: of all, as V15.
    foreign_reference: bool = True
