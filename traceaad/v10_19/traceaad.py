"""V10.19: its mechanism, using shared search execution."""

from traceaad.common.search import DevelopingSearch
from traceaad.common.history import final_attempt
from traceaad.common.selection import better
from .config import Config
from .prompts import PromptBuilder
from .selection import sample_parent


class TraceAADV1019(DevelopingSearch):
    METHOD = "v1019"
    Config = Config
    PromptBuilder = PromptBuilder

    def _ending(self, exploration):
        """Why the exploration ends now, or None while the change is still being made to work."""
        if exploration["proposed"] is None:
            return "no_program"
        if better(exploration["best"]["fitness"], exploration["source"]["fitness"]):
            return "improved"
        if len(exploration["development"]) >= self.config.development_steps:
            return "cap"
        return None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompts.explorations = self.facts.explorations

    def _choose_parent(self, eligible):
        return sample_parent(eligible, self.attempts_table, self.programs, self.parent_rng, self.facts.explorations)

    def _close_exploration(self, exploration, reason):
        proposal, proposed, best = exploration["proposal"], exploration["proposed"], exploration["best"]
        source = exploration["source"]
        search_best = max(self.archive.values(), key=lambda n: (n["fitness"], -n["id"]))
        record = {
            "id": exploration["id"], "start_id": proposal["parent_id"], "proposal_attempt": proposal["id"],
            "idea": proposal["idea"], "first_status": final_attempt(proposal, self.attempts_table)["status"],
            "proposed_id": proposed["id"] if proposed else None,
            "first_score": proposed["score"] if proposed else None,
            "first_rank": self._rank(proposed) if proposed else None,
            "start_score": source["score"] if source.get("valid") else None,
            "development_attempts": [a["id"] for a in exploration["development"]],
            "best_id": best["id"] if best else None, "best_score": best["score"] if best else None,
            "best_rank": self._rank(best) if best else None, "reason": reason,
            "improved_start": bool(best and source.get("valid") and better(best["fitness"], source["fitness"])),
            "search_best_id": search_best["id"], "search_best_score": search_best["score"],
            "closed_after": self.attempts}
        self._save(exploration=record)
