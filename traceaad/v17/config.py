"""V17 policy: V16 with continuation by a lineage's own progress and with component imports; the base loop and shared evaluation constraints are unchanged."""

from dataclasses import dataclass

from traceaad.v16.config import Config as V16Config


@dataclass
class Config(V16Config):
    # Used only when progress_rule is False: V16 keeps a challenger by its optimistic estimate.
    projection: bool = False
    # A lineage other than the champion stays active while it improves on its own record;
    # attempts outside the champion lineage are bounded by the champion's attempts beyond the
    # stagnation window since the last new best. False: same-age comparison, as V16.
    progress_rule: bool = True
    # A sampled Crossover imports an improving change of another lineage, shown as its diff,
    # when one is untried; a paused lineage is resumed by a new change of a better lineage.
    # False: Crossover with a reference design, as V16.
    imports: bool = True
