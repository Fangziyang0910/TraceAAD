"""V10.23: V10.22's search; steps differ in the kind of change and in their material."""

from traceaad.common.prompts import ContextTooLong
from traceaad.v10_22.traceaad import TraceAADV1022
from .config import Config
from .prompts import PromptBuilder


class TraceAADV1023(TraceAADV1022):
    METHOD = "v1023"
    Config = Config
    PromptBuilder = PromptBuilder

    def _develop(self, exploration):
        """Continue an exploration's new computation from its own path (V10.17 used a plain Refine)."""
        if (exploration["proposed"] is None or not self._developed(exploration["id"])
                or len(exploration["development"]) >= self.config.development_steps):
            self._close_exploration(exploration)
            return
        best = exploration["best"]
        try:
            request = self.prompts.build("Develop", best, source=exploration["source"])
        except ContextTooLong:
            self._close_exploration(exploration)
            return
        request.update(sampled_action="Develop", fallbacks=[], parent_id=best["id"], reference_id=None,
                       selection=None, reference_selection=None,
                       exploration={"id": exploration["id"], "step": len(exploration["development"]) + 1})
        self._attempt(request, parent=best)
