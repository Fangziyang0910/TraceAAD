"""V10.15: its mechanism, using shared search execution."""

from traceaad.common.search import Search
from traceaad.common.prompts import ContextTooLong
from traceaad.common.selection import choose_reference
from .config import Config
from .prompts import PromptBuilder
from .selection import sample_parent, choose_explore_references


class TraceAADV1015(Search):
    METHOD = "v1015"
    Config = Config
    PromptBuilder = PromptBuilder
    MEASURE_CALLS = False
    REPLACE_FAILED_FINALISTS = False

    def _choose_parent(self, eligible):
        return sample_parent(eligible, self.parent_rng)

    def _search(self):
        if self.progress.repair_id is not None:
            self._repair()
            return
        if self.attempts >= self.config.budget:
            self.progress.phase = "freeze"
            self._save()
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
        explore_references, explore_reference_selection = [], None
        flags = []
        if action == "Crossover":
            reference, reference_selection = choose_reference(parent, self.archive, self.reference_rng)
            if reference is None:
                action = "Refine"
                flags.append("crossover_fallback")
        elif action == "Explore" and self.config.explore_cards:
            explore_references, explore_reference_selection = choose_explore_references(
                parent, self.archive, self.reference_rng, self.config.explore_cards)
        best_score = max(self.archive.values(), key=lambda n: n["fitness"])["score"]
        try:
            request = self.prompts.build(action, parent, reference=reference,
                                         references=explore_references, best_score=best_score)
        except ContextTooLong:
            self.progress.too_long.append(parent["id"])
            self._save()
            return
        if request["action"] == "Refine" and action == "Crossover":
            reference = None
            flags.append("crossover_context_fallback")
        request["sampled_action"] = sampled
        request["fallbacks"] = flags
        request["parent_id"] = parent["id"]
        request["reference_id"] = reference["id"] if reference else None
        request["selection"] = selection
        request["reference_selection"] = reference_selection
        request["explore_reference_selection"] = explore_reference_selection
        self._attempt(request, parent=parent, reference=reference)
