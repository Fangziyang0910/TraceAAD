"""V10.17: its mechanism, using shared search execution."""

import random
from traceaad.common.search import Search
from traceaad.common.prompts import ContextTooLong
from traceaad.common.history import final_attempt
from .config import Config
from .prompts import PromptBuilder


class TraceAADV1017(Search):
    RECORD_EXPLORATIONS = True
    METHOD = "v1017"
    Config = Config
    PromptBuilder = PromptBuilder

    def _search(self):
        if self.progress.repair_id is not None:
            self._repair()
            return
        opened = self._open_exploration()
        if self.attempts >= self.config.budget:
            if opened is not None:
                self._close_exploration(opened)
            self.progress.phase = "freeze"
            self._save()
            return
        if opened is not None:
            self._develop(opened)
            return
        self._ordinary_search()

    def _developed(self, exploration_id):
        """Whether an exploration's new program is developed: a fixed draw per exploration,
        independent of every other random choice, so a resumed run draws the same."""
        draw = random.Random(f"v10.17:{self.config.seed}:develop:{exploration_id}").random()
        return draw < self.config.development_probability

    def _develop(self, exploration):
        if (exploration["proposed"] is None or not self._developed(exploration["id"])
                or len(exploration["development"]) >= self.config.development_steps):
            self._close_exploration(exploration)
            return
        best = exploration["best"]
        try:
            request = self.prompts.build("Refine", best)
        except ContextTooLong:
            self._close_exploration(exploration)
            return
        request.update(sampled_action="Refine", fallbacks=[], parent_id=best["id"], reference_id=None,
                       selection=None, reference_selection=None,
                       exploration={"id": exploration["id"], "step": len(exploration["development"]) + 1})
        self._attempt(request, parent=best)

    def _close_exploration(self, exploration):
        proposal, proposed, best = exploration["proposal"], exploration["proposed"], exploration["best"]
        search_best = min(self.archive.values(), key=lambda n: (n["fitness"], n["id"]))
        record = {
            "id": exploration["id"], "start_id": proposal["parent_id"], "proposal_attempt": proposal["id"],
            "idea": proposal["idea"], "first_status": final_attempt(proposal, self.attempts_table)["status"],
            "proposed_id": proposed["id"] if proposed else None,
            "developed": proposed is not None and self._developed(exploration["id"]),
            "first_score": proposed["score"] if proposed else None,
            "development_attempts": [a["id"] for a in exploration["development"]],
            "best_id": best["id"] if best else None, "best_score": best["score"] if best else None,
            "search_best_id": search_best["id"], "search_best_score": search_best["score"],
            "closed_after": self.attempts}
        self._save(exploration=record)
