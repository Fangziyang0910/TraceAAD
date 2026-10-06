"""V10.18: its mechanism, using shared search execution."""

from traceaad.common.search import DevelopingSearch
from .config import Config
from .prompts import PromptBuilder


class TraceAADV1018(DevelopingSearch):
    METHOD = "v1018"
    Config = Config
    PromptBuilder = PromptBuilder

    def _ending(self, exploration):
        """Why the exploration ends now, or None while it is still being developed."""
        if exploration["proposed"] is None:
            return "no_program"
        if self._rank(exploration["best"]) < self.config.final_candidates:
            return "competitive"
        steps = len(exploration["development"])
        if steps >= self.config.development_max:
            return "cap"
        if steps >= self.config.development_min and exploration["stalled"] >= self.config.development_stall:
            return "stalled"
        return None
