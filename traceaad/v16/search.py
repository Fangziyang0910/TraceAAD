"""Develop lineages using matched-age quality and recent improvement."""

import math

from .selection import better
from ._allocation import LineageSearch, best_score
from .config import Config
from .prompts import PromptBuilderV16


def projected(attempts, checkpoint):
    """Best score after ``checkpoint`` attempts, extended by its recent rate over ``checkpoint`` more attempts."""
    first = next(i for i, p in enumerate(attempts, 1) if p.valid)
    start = max(first, math.ceil(checkpoint / 2))
    now = best_score(attempts[:checkpoint])
    if start >= checkpoint:
        return now
    rate = (best_score(attempts[:start]) - now) / (checkpoint - start)
    return now - rate * checkpoint


class TraceAADV16(LineageSearch):
    METHOD = "traceaad_v16"

    def __init__(self, *, config=None, **kwargs):
        super().__init__(config=config or Config(), **kwargs)
        self.prompts = PromptBuilderV16(
            self.llm,
            kwargs.get("task"),
            kwargs["evaluation"],
            self.programs,
            self.config,
            info=self.config.exposure == "info",
            changes=self.config.changes,
        )

    def _references(self, parent):
        if not self.config.foreign_reference:
            return super()._references(parent)
        lineage = self._lineage_of(parent)
        others = {i: p for i, p in self.archive.items() if self._lineage_of(p) != lineage}
        return others or self.archive

    def _allocate(self):
        """The lineage that receives the next attempt, with what decided it."""
        lineages = self._lineages()
        champion = self._lineage_of(self._best())
        first = 1 + self.config.mature_steps
        active = []
        for lineage, attempts in lineages.items():
            if lineage == champion or best_score(attempts) is None:
                continue
            effort, checkpoint, rescued = len(attempts), None, False
            if effort >= first:
                checkpoint = first
                while 2 * checkpoint <= effort:
                    checkpoint *= 2
                own = best_score(attempts[:checkpoint])
                if own is None:
                    continue
                bar = best_score(lineages[champion][:checkpoint])
                if bar is not None and better(bar, own):
                    ahead = best_score(lineages[champion][: 2 * checkpoint])
                    if not self.config.projection or better(ahead, projected(attempts, checkpoint)):
                        continue
                    rescued = True
            active.append((effort, lineage, checkpoint, rescued))
        effort, lineage, checkpoint, rescued = min(
            active, default=(len(lineages[champion]), champion, None, False)
        )
        return lineage, {
            "lineage": lineage,
            "champion": champion,
            "effort": effort,
            "checkpoint": checkpoint,
            "projected": rescued,
        }
