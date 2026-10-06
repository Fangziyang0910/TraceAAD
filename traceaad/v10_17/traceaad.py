"""V10.17: its mechanism, using shared search execution."""

import random
from traceaad.common.search import Search
from traceaad.common.prompts import ContextTooLong
from traceaad.common.history import final_attempt
from traceaad.common.selection import better, choose_reference
from .config import Config
from .prompts import PromptBuilder


class TraceAADV1017(Search):
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
        eligible = [node for node in self.archive.values() if node["id"] not in self.progress.too_long]
        if not eligible:
            self.progress.phase = "freeze"
            self._save()
            return
        parent, selection = self._choose_parent(eligible)
        sampled = self.action_rng.choices(list(self.config.operators), list(self.config.operators.values()))[0]
        action = sampled
        reference, reference_selection = None, None
        flags = []
        if action == "Crossover":
            reference, reference_selection = choose_reference(parent, self.archive, self.reference_rng)
            if reference is None:
                action = "Refine"
                flags.append("crossover_fallback")
        try:
            request = self.prompts.build(action, parent, reference=reference)
        except ContextTooLong:
            self.progress.too_long.append(parent["id"])
            self._save()
            return
        if request["action"] == "Refine" and action == "Crossover":
            reference = None
            flags.append("crossover_context_fallback")
        exploration = ({"id": len(self.facts.explorations) + 1, "step": 0}
                       if request["action"] == "Explore" else None)
        request.update(sampled_action=sampled, fallbacks=flags, parent_id=parent["id"],
                       reference_id=reference["id"] if reference else None, selection=selection,
                       reference_selection=reference_selection, exploration=exploration)
        self._attempt(request, parent=parent, reference=reference)

    def _open_exploration(self):
        """The exploration still in progress, read from the attempts (so a resumed run continues it).

        Returns the exploration's state: its proposal, the new program the proposal
        produced (None if it produced none), its development attempts and the best
        program reached so far, which is where the next development step starts.
        """
        tagged = [a for a in self.attempts_table.values() if a.get("exploration")]
        if not tagged:
            return None
        eid = max(a["exploration"]["id"] for a in tagged)
        if eid in self.facts.explorations:
            return None
        attempts = sorted((a for a in tagged if a["exploration"]["id"] == eid), key=lambda a: a["id"])
        proposal, development = attempts[0], attempts[1:]
        final = final_attempt(proposal, self.attempts_table)
        proposed = self.archive.get(final["program_id"]) if final["status"] == "valid" else None
        best = proposed
        for attempt in development:
            final = final_attempt(attempt, self.attempts_table)
            reached = self.archive.get(final["program_id"]) if final["status"] == "valid" else None
            if reached is not None and better(reached["fitness"], best["fitness"]):
                best = reached
        return {"id": eid, "proposal": proposal, "proposed": proposed, "development": development, "best": best}

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
            self.progress.too_long.append(best["id"])
            self._close_exploration(exploration)
            return
        request.update(sampled_action="Refine", fallbacks=[], parent_id=best["id"], reference_id=None,
                       selection=None, reference_selection=None,
                       exploration={"id": exploration["id"], "step": len(exploration["development"]) + 1})
        self._attempt(request, parent=best)

    def _close_exploration(self, exploration):
        proposal, proposed, best = exploration["proposal"], exploration["proposed"], exploration["best"]
        search_best = max(self.archive.values(), key=lambda n: (n["fitness"], -n["id"]))
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
