"""V17 policy: V16 with V15's same-age comparison and with component imports; the base loop and shared evaluation constraints are unchanged."""

from dataclasses import dataclass

from traceaad.v16.config import Config as V16Config


@dataclass
class Config(V16Config):
    # Used only when progress_rule is False: V16 keeps a challenger by its optimistic estimate.
    # False: a challenger continues only by V15's same-age comparison with the champion.
    projection: bool = False
    # True: a lineage other than the champion stays active while it improves on its own record;
    # attempts outside the champion lineage are bounded by the champion's attempts beyond the
    # stagnation window since the last new best, and a paused lineage is resumed by an import.
    # False: same-age comparison, as V15 (V16 with projection False).
    progress_rule: bool = False
    # A sampled Crossover imports an improving change of another lineage, shown as its diff,
    # when one is untried. False: Crossover with a reference design, as V16.
    imports: bool = True
