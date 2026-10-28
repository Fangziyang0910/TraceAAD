"""V17 prompts: V16 with an Import request.

An Import shows the current algorithm as a Refine does (its formation path, the
programs already generated from it, its Design and code) and one component: a
change that improved another lineage, given by its Change, the score of that
algorithm before and after it, and its code diff. The program the component
came from is not shown, neither as code nor as Design, so the reply adds a
computation to the current algorithm rather than rebuilding the other one.
"""

from traceaad.history import code_diff, path, score_text
from traceaad.prompts import idea_view
from traceaad.v16.prompts import PromptBuilderV16

SEARCH_ACTIONS = ("Refine", "Explore", "Crossover", "Mature", "Import")
COMPONENT = "[Component From Another Algorithm]\nA change that improved another algorithm of this search. That algorithm follows a different design and is not shown; the diff gives only the lines the change removed and added, with a little context.\nScore of that algorithm: {before} → {after} (improved)"
IMPORT = "[Your Task: Import]\nImplement the computation that the component above adds or changes in the current algorithm, adapted to its code, so that it improves the current algorithm's solution quality. Keep the current algorithm's main decision rule; do not rebuild the other algorithm."
IMPORT_ANALYSIS = "what the component computes and why it improved the other algorithm, and how you adapt it to the current algorithm"


def step_of(program, programs):
    """The program a change started from (None unless valid) and the action that made it.

    A repaired program started where its failed version started.
    """
    action, source = program.action, program
    if action == "Repair":
        source = programs[program.repair_of]
        action = source.action
    start = programs.get(source.parent_id)
    return (start if start is not None and start.valid else None), action


class PromptBuilderV17(PromptBuilderV16):
    def children_from(self, node):
        return [
            p
            for p in sorted(self.programs.values(), key=lambda p: p.id)
            if p.parent_id == node.id and p.repair_of is None and p.action in SEARCH_ACTIONS
        ]

    def component(self, program):
        start, _ = step_of(program, self.programs)
        text = COMPONENT.format(before=score_text(start.fitness), after=score_text(program.fitness))
        if program.change:
            text += f"\nChange: {idea_view(program.change)}"
        return text + f"\n```diff\n{code_diff(start.code, program.code)}\n```"

    def import_component(self, parent, component):
        """Add to ``parent`` the change that ``component`` made in another lineage."""
        sequence = path(parent, self.programs)
        depth = len(self.steps(sequence)) - 1
        history, ids = self._formation(sequence, min(self.config.history_depth, depth))
        tried, tried_ids = self._children_section(parent)
        sections = self.common + [
            self.records,
            history,
            tried,
            self._current(parent),
            self.component(component),
            IMPORT,
            self._format(IMPORT_ANALYSIS, edit=True),
        ]
        result = self._result(sections, "Import", ids)
        result["child_ids"] = tried_ids
        return result
