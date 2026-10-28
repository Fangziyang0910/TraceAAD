"""V17: V16 with continuation by a lineage's own progress and with component imports.

Allocation (progress_rule). A lineage below the first checkpoint, 1 +
mature_steps attempts, is developed first, the one with the fewest attempts
first; the champion lineage, the one holding the best program, is no exception.
Otherwise the champion receives the attempt unless exploration is open: since
the last new best, the attempts outside the champion lineage are fewer than the
champion's attempts minus stagnation_window. So after each new best the
champion first spends stagnation_window attempts, and later at most every
other attempt goes elsewhere.

An exploration attempt starts a new lineage by Diverge when V15 would: after
stagnation_window attempts without a new best since the last new best or the
last Diverge. Otherwise it goes to an active lineage: one whose attempts since
its own last improvement are fewer than max(first checkpoint, half its attempts).
The lineage that improved most recently goes first, then the one with fewer
attempts, then the older one. A lineage is not compared with the champion and
its potential is not estimated: it continues while its own record improves.
Without an active lineage, a paused lineage is resumed by an import when a
lineage scoring better has made a new component since its last attempt, the
paused lineage scoring best first; otherwise a new lineage starts by Diverge.

Components and imports (imports). A component is a change that set a new best
of its lineage, made by Refine, Explore, Crossover, Mature or Import from a
valid program of the lineage, that changed at most COMPONENT_LINES lines and
gained at least the median relative gain of all such changes. When Crossover is
sampled and another lineage has a component this lineage has not imported, the
attempt imports the one whose program scores best. The prompt gives its Change
and diff but not the program it came from (prompts.PromptBuilderV17).
Otherwise the attempt is V16's Crossover with a reference from another lineage.

Everything is derived from the program records. With progress_rule and imports
both False, the search is V16 with the configured projection.
"""

from collections import Counter
import difflib
import statistics

from traceaad.selection import better, choose_reference, sample_parent
from traceaad.v14 import novelty
from traceaad.v15.search import best_score
from traceaad.v16.search import TraceAADV16
from .config import Config
from .prompts import PromptBuilderV17, step_of

COMPONENT_LINES = 60
CHANGES = ("Refine", "Explore", "Crossover", "Mature", "Import")


def stall(attempts):
    """Attempts made after the one that first reached the lineage's best score."""
    best, at = None, 0
    for index, program in enumerate(attempts, 1):
        if program.valid and (best is None or better(program.fitness, best)):
            best, at = program.fitness, index
    return len(attempts) - at


class TraceAADV17(TraceAADV16):
    METHOD = "traceaad_v17"

    def __init__(self, *, config=None, **kwargs):
        super().__init__(config=config or Config(), **kwargs)
        self.prompts = PromptBuilderV17(
            self.llm,
            kwargs.get("task"),
            kwargs["evaluation"],
            self.programs,
            self.config,
            info=self.config.exposure == "info",
            changes=self.config.changes,
        )
        self._changed = {}

    # Components

    def _changed_lines(self, start, program):
        if program.id not in self._changed:
            diff = list(difflib.unified_diff(start.code.splitlines(), program.code.splitlines(), n=0, lineterm=""))
            self._changed[program.id] = sum(line[:1] in "+-" for line in diff[2:])
        return self._changed[program.id]

    def _components(self):
        """The programs whose change is a component, oldest first."""
        best, steps = {}, []
        for program in sorted(self.archive.values(), key=lambda p: p.id):
            lineage = self._lineage_of(program)
            before = best.get(lineage)
            if before is not None and not better(program.fitness, before):
                continue
            best[lineage] = program.fitness
            start, action = step_of(program, self.programs)
            if before is None or start is None or action not in CHANGES:
                continue
            if self._changed_lines(start, program) <= COMPONENT_LINES:
                steps.append(((before - program.fitness) / max(abs(before), 1e-12), program))
        if not steps:
            return []
        median = statistics.median(gain for gain, _ in steps)
        return [program for gain, program in steps if gain >= median]

    def _imported(self, lineage):
        return {
            p.reference_id for p in self.programs.values() if p.action == "Import" and self._lineage_of(p) == lineage
        }

    def _component(self, parent):
        """The component of another lineage, not yet imported into ``parent``'s lineage, whose program scores best."""
        lineage = self._lineage_of(parent)
        imported = self._imported(lineage)
        offered = [p for p in self._components() if self._lineage_of(p) != lineage and p.id not in imported]
        return min(offered, key=lambda p: (p.fitness, -p.id), default=None)

    def _resumable(self, lineages, paused):
        """The paused lineage scoring best for which a lineage scoring better made a component since its last attempt."""
        components = self._components()
        for lineage in sorted(paused, key=lambda l: (best_score(lineages[l]), l)):
            last, own = lineages[lineage][-1].id, best_score(lineages[lineage])
            offered = [
                p
                for p in components
                if p.id > last and self._lineage_of(p) != lineage and better(p.fitness, own)
            ]
            if offered:
                return lineage, min(offered, key=lambda p: (p.fitness, -p.id))
        return None, None

    # Allocation

    def _allocate(self):
        """The lineage that receives the next attempt (None: a new one), with what decided it."""
        if not self.config.progress_rule:
            return super()._allocate()
        lineages = {l: a for l, a in self._lineages().items() if best_score(a) is not None}
        best = self._best()
        champion = self._lineage_of(best)
        first = 1 + self.config.mature_steps

        def allocation(lineage, reason, **facts):
            effort = len(lineages[lineage]) if lineage is not None else 0
            return lineage, {"lineage": lineage, "champion": champion, "effort": effort, "reason": reason, **facts}

        young = min(((len(a), l) for l, a in lineages.items() if len(a) < first), default=None)
        if young is not None:
            return allocation(young[1], "young")
        recent = [self._lineage_of(p) == champion for p in self.programs.values() if p.generation > best.generation]
        account = {"champion": sum(recent), "other": len(recent) - sum(recent)}
        if account["other"] >= account["champion"] - self.config.stagnation_window:
            return allocation(champion, "champion", account=account)
        if self.config.diverge and self._stalled():
            # A lineage less developed than the champion improves on its own record easily, so one is
            # almost always active; without this rule a new lineage would start almost never.
            return allocation(None, "diverge", account=account)
        active, paused = [], []
        for lineage, attempts in lineages.items():
            if lineage == champion:
                continue
            idle = stall(attempts)
            if idle < max(first, len(attempts) / 2):
                active.append((idle, len(attempts), lineage))
            else:
                paused.append(lineage)
        if active:
            idle, _, lineage = min(active)
            return allocation(lineage, "active", stall=idle, account=account)
        if self.config.imports:
            lineage, component = self._resumable(lineages, paused)
            if lineage is not None:
                return allocation(lineage, "resume", component=component.id, account=account)
        if self.config.diverge:
            return allocation(None, "diverge", account=account)
        return allocation(champion, "champion", account=account)

    # Search

    def _search(self, sampled=None):
        if (
            not self.config.progress_rule
            or self.config.allocation == "fitness"
            or self.budget_used >= self.config.budget
            or not self.archive
        ):
            return super()._search(sampled)
        lineage, allocation = self._allocate()
        if allocation["reason"] == "diverge":
            request = self._diverge_request()
            request["allocation"] = allocation
            return self._create_program(request)
        members = [p for p in self.archive.values() if self._lineage_of(p) == lineage]
        node = min(members, key=lambda p: (p.fitness, p.id))
        if allocation["reason"] == "young":
            request = self.prompts.mature(node, self._best().fitness)
            request.update(parent_id=node.id, allocation=allocation)
            return self._create_program(request, parent=node)
        if allocation["reason"] == "resume":
            return self._import(node, self.programs[allocation["component"]], allocation=allocation)
        return self._step(members, sampled, allocation=allocation)

    def _import(self, parent, component, **record):
        request = self.prompts.import_component(parent, component)
        request.update(parent_id=parent.id, reference_id=component.id, **record)
        return self._create_program(request, parent=parent, reference=component)

    def _step(self, eligible, sampled=None, **record):
        """TraceAAD._step, except that a sampled Crossover imports an untried component when there is one."""
        if not self.config.imports:
            return super()._step(eligible, sampled, **record)
        parent, selection = sample_parent(eligible, self.programs, self.progress.rng["parent"])
        if sampled is None:
            sampled = self.progress.rng["action"].choices(
                list(self.config.operators), list(self.config.operators.values())
            )[0]
        component = self._component(parent) if sampled == "Crossover" else None
        if component is not None:
            return self._import(
                parent, component, sampled_action=sampled, fallbacks=[], selection=selection, **record
            )
        action, reference, reference_selection, flags = sampled, None, None, []
        if action == "Crossover":
            reference, reference_selection = choose_reference(
                parent, self._references(parent), self.progress.rng["reference"]
            )
            if reference is None:
                action = "Refine"
                flags.append("crossover_fallback")
        request = self.prompts.build(action, parent, reference=reference)
        request.update(
            sampled_action=sampled,
            fallbacks=flags,
            parent_id=parent.id,
            reference_id=reference.id if reference else None,
            selection=selection,
            reference_selection=reference_selection,
            **record,
        )
        self._create_program(request, parent=parent, reference=reference)

    def _rebuild(self, failed, parent, reference):
        if failed.action == "Import":
            return self.prompts.import_component(parent, reference)
        return super()._rebuild(failed, parent, reference)

    def _summary(self, status, error=None):
        save = self.facts.save_summary

        def with_policy(summary):
            programs = list(self.programs.values())
            outcome = {p.repair_of: p for p in programs if p.repair_of is not None}
            first = [p for p in programs if p.repair_of is None]
            imports = [p for p in first if p.action == "Import"]
            improved = [
                p
                for p in imports
                if outcome.get(p.id, p).valid and better(outcome.get(p.id, p).fitness, self.programs[p.parent_id].fitness)
            ]
            resumed = [p for p in imports if (p.request.get("allocation") or {}).get("reason") == "resume"]
            distances = [
                novelty.distance(self._shingle(outcome.get(p.id, p)), self._shingle(self.programs[p.reference_id]))
                for p in imports
                if outcome.get(p.id, p).valid
            ]
            summary["v17"] = {
                "progress_rule": self.config.progress_rule,
                "imports": self.config.imports,
                "attempts_by_reason": dict(Counter((p.request.get("allocation") or {}).get("reason", "init") for p in first)),
                "components": len(self._components()) if self.archive else 0,
                "import_attempts": len(imports),
                "imports_improved": len(improved),
                "resumed": len(resumed),
                "resumed_improved": sum(p in improved for p in resumed),
                "import_source_distance_median": statistics.median(distances) if distances else None,
                "imports_near_source": sum(d < self.config.map_distance for d in distances),
            }
            save(summary)

        self.facts.save_summary = with_policy
        try:
            return super()._summary(status, error)
        finally:
            self.facts.save_summary = save
