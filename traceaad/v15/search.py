"""V15: V14 with an exposure policy.

An LLM operator writes from what its prompt shows, so selection acts twice: it
picks the program to edit, and it decides what the operator sees. V15 sets both.

Who is exposed (allocation="effort"). Attempts go to lineages: a program written
without a parent and every program derived from it. A lineage below the first
checkpoint, 1 + mature_steps attempts, is developed from its best program. At
each later checkpoint (5, 10, 20, ... by default) it is compared with the
champion lineage, the one holding the best program, after the same number of
attempts: it continues while it is not worse and pauses otherwise. The next
attempt goes to the active lineage with the fewest attempts, else to the
champion, whose parent is sampled among its own programs. A new lineage starts
after stagnation_window attempts without a new best since the last start.

How others are exposed (exposure="info"): see prompts.PromptBuilderV15.

Everything is derived from the program records, so a resumed run continues
the same allocation. allocation="fitness" with exposure="code" is V14.
"""

from collections import defaultdict

from traceaad.selection import better
from traceaad.v14.search import TraceAADV14
from .config import Config
from .prompts import PromptBuilderV15


def best_score(attempts):
    scores = [p.fitness for p in attempts if p.valid]
    return min(scores) if scores else None


class TraceAADV15(TraceAADV14):
    METHOD = "traceaad_v15"

    def __init__(self, *, config=None, **kwargs):
        super().__init__(config=config or Config(), **kwargs)
        self.prompts = PromptBuilderV15(
            self.llm,
            kwargs.get("task"),
            kwargs["evaluation"],
            self.programs,
            self.config,
            info=self.config.exposure == "info",
        )
        self._lineage = {}

    # Lineages

    def _lineage_of(self, program):
        """The first program of the lineage ``program`` belongs to.

        A retry after a revisit has no parent when the rejected reply had none;
        it stays in the rejected reply's lineage.
        """
        chain = []
        while program.id not in self._lineage:
            link = program.parent_id if program.parent_id is not None else program.repair_of
            if link is None:
                break
            chain.append(program.id)
            program = self.programs[link]
        root = self._lineage.get(program.id, program.id)
        for pid in chain + [program.id]:
            self._lineage[pid] = root
        return root

    def _lineages(self):
        """Each lineage's attempts, oldest first, keyed by its first program."""
        lineages = defaultdict(list)
        for program in sorted(self.programs.values(), key=lambda p: p.id):
            lineages[self._lineage_of(program)].append(program)
        return lineages

    def _allocate(self):
        """The lineage that receives the next attempt, with what decided it."""
        lineages = self._lineages()
        champion = self._lineage_of(self._best())
        first = 1 + self.config.mature_steps
        active = []
        for lineage, attempts in lineages.items():
            if lineage == champion or best_score(attempts) is None:
                continue
            effort, checkpoint = len(attempts), None
            if effort >= first:
                checkpoint = first
                while 2 * checkpoint <= effort:
                    checkpoint *= 2
                own = best_score(attempts[:checkpoint])
                bar = best_score(lineages[champion][:checkpoint])
                if own is None or (bar is not None and better(bar, own)):
                    continue
            active.append((effort, lineage, checkpoint))
        effort, lineage, checkpoint = min(active, default=(len(lineages[champion]), champion, None))
        return lineage, {"lineage": lineage, "champion": champion, "effort": effort, "checkpoint": checkpoint}

    # Search

    def _search(self, sampled=None):
        if self.config.allocation == "fitness" or self.budget_used >= self.config.budget or not self.archive:
            return super()._search(sampled)
        if self.config.diverge and self._stalled():
            return self._create_program(self._diverge_request())
        lineage, allocation = self._allocate()
        members = [p for p in self.archive.values() if self._lineage_of(p) == lineage]
        if allocation["effort"] < 1 + self.config.mature_steps:
            node = min(members, key=lambda p: (p.fitness, p.id))
            request = self.prompts.mature(node, self._best().fitness)
            request.update(parent_id=node.id, allocation=allocation)
            return self._create_program(request, parent=node)
        return self._step(members, sampled, allocation=allocation)

    def _stalled(self):
        if self.config.allocation == "fitness":
            return super()._stalled()
        last = max(
            (p.generation for p in self.programs.values() if p.action == "Diverge" and p.repair_of is None),
            default=0,
        )
        return self.progress.generations - max(self._best().generation, last) >= self.config.stagnation_window

    def _tried(self):
        if self.config.allocation == "fitness":
            return super()._tried()
        tried = []
        for attempts in self._lineages().values():
            if attempts[0].action != "Diverge":
                continue
            first = next((p for p in attempts if p.action == "Diverge" and p.status != "revisit"), None)
            if first is not None:
                tried.append((first, best_score(attempts)))
        return tried[-self.config.diverge_shown :] if self.config.diverge_shown else []

    def _summary(self, status, error=None):
        save = self.facts.save_summary

        def with_policy(summary):
            lineages = self._lineages()
            developed = [a for a in lineages.values() if best_score(a) is not None]
            champion = self._lineage_of(self._best()) if self.archive else None
            total = sum(len(a) for a in lineages.values())
            summary["v15"] = {
                "allocation": self.config.allocation,
                "exposure": self.config.exposure,
                "lineages": len(developed),
                "lineages_past_first_checkpoint": sum(len(a) > 1 + self.config.mature_steps for a in developed),
                "top_lineage_share": max(map(len, lineages.values()), default=0) / total if total else 0.0,
                "champion_lineage": champion,
                "champion_origin": self.programs[champion].action if champion else None,
                "champion_attempts": len(lineages[champion]) if champion else 0,
            }
            save(summary)

        self.facts.save_summary = with_policy
        try:
            return super()._summary(status, error)
        finally:
            self.facts.save_summary = save
