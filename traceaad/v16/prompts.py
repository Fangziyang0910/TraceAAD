"""V16 prompts: V15 with the edited lineage shown as changes and their outcomes.

Each modifying reply also states its Change: what the returned code changed
from the algorithm it was produced from, before and after. Formation steps,
earlier attempts from the current algorithm and the improvements of the best
score are shown by Change and outcome, so each reads as a change and its effect
rather than a whole program again. A whole program is still shown by its
Design: the current algorithm, the start of a formation path, a Crossover
reference, a program written without a parent, and the programs of the
initialization and new-design requests. The attempts already made from the
current algorithm are marked as evaluated, and the Crossover task keeps the
current main decision rule.
"""

from traceaad.prompts import idea_view
from traceaad.v15.prompts import PromptBuilderV15

RECORDS_CHANGES = "Scores and failures are evaluation results. Code is shown only for the algorithm you are asked to change. Design describes a whole program; Change states what a program changed from the algorithm it was produced from. Both were written when the program was generated. A repaired program inherits the Design and Change of the failed program."
FORMATION_INTRO_CHANGES = "The steps that produced the current algorithm, oldest first. Each step gives its evaluated outcome and the Change that produced it. If a step required repair, the first failure is stated separately."
TRIED_NOTE = "These programs are already evaluated, so repeating one of them gives no new result. Make a change that none of them made; you may keep a change that improved the score and add to it."
CROSSOVER_CHANGES = "[Your Task: Crossover]\nImprove the current algorithm with computations from the reference design. Keep the current algorithm's main decision rule. The reference is described, not shown: implement the parts you take from it in the current code, so that they work with its existing computations."
DESIGN_CHANGES = "describe the code you returned: its main computations, naming the quantities, formulas and parameter values that decide its output; at most 200 words"
CHANGE = "state what the returned code changed from the current algorithm, naming each changed computation or value before and after; at most 60 words"


class PromptBuilderV16(PromptBuilderV15):
    def __init__(self, *args, info=True, changes=True):
        super().__init__(*args, info=info)
        self.changes = info and changes
        if self.changes:
            self.records = RECORDS_CHANGES
            self.formation_intro = FORMATION_INTRO_CHANGES
            self.tried_note = TRIED_NOTE
            self.crossover = CROSSOVER_CHANGES

    def _what(self, program, missing=""):
        if not (self.changes and program.change):
            return super()._what(program, missing)
        return f"Change: {idea_view(program.change)}"

    def _formation(self, sequence, count, **kwargs):
        text, ids = super()._formation(sequence, count, **kwargs)
        if self.changes and len(self.steps(sequence)) == 1:
            # An initial design: its Design is shown with the program itself.
            text = text[: text.rindex("\nDesign: ")]
        return text, ids

    def _current(self, node):
        if not self.changes:
            return super()._current(node)
        return f"[Current Algorithm]\n{self.measured(node)}\nDesign: {idea_view(node.idea)}\n```python\n{node.code.rstrip()}\n```"

    def _format(self, analysis, *, design=True, edit=False):
        if not (self.changes and design):
            return super()._format(analysis, design=design)
        fields = [
            f'"analysis": explain {analysis}.',
            '"code": the complete Python program. Write no comments or docstrings in the code.',
            f'"design": {DESIGN_CHANGES}.',
        ]
        if edit:
            fields.append(f'"change": {CHANGE}.')
        count = "four" if edit else "three"
        return f"[Output Format]\nReply with one JSON object with {count} string fields, in this order:\n" + "\n".join(fields)
